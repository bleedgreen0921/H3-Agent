"""Optional, explicit local knowledge settings."""

from pathlib import Path
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
    field_validator,
)
from urllib.parse import urlsplit
from typing import Literal


class KnowledgeSource(BaseModel):
    """只有显式配置的目录与云空间允许被递归扫描。"""
    model_config = ConfigDict(extra='forbid', frozen=True)

    # 本地来源使用 path；飞书来源使用 token 和明确的访问身份。
    kind: Literal['local_directory', 'feishu_wiki', 'feishu_folder']
    path: Path | None = None
    token: str | None = None
    node_token: str | None = None
    identity: Literal['user', 'bot'] = 'user'

    # 来源扫描的包含和排除模式，默认包含所有匹配项。
    include: list[str] = Field(default_factory=lambda: ['*'])
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def target(self):
        """根据来源类型检查必需的本地路径或飞书 token。"""
        if self.kind == 'local_directory' and self.path is None:
            raise ValueError('local_directory requires path')
        if self.kind != 'local_directory' and not self.token:
            raise ValueError('Feishu source requires token')
        return self


class ModelService(BaseModel):
    """Embedding 或 Reranker HTTP 服务的连接与批处理配置。"""
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    base_url: str
    model: str = Field(min_length=1)
    api_key: SecretStr = SecretStr("local")
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=4, gt=0, le=256)

    @field_validator("base_url")
    @classmethod
    def service_url(cls, value):
        """拒绝非 HTTP 地址、内嵌凭据及查询或片段参数。"""
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Use an HTTP(S) service URL without embedded credentials")
        return value.rstrip("/")


class KnowledgeConfig(BaseModel):
    """知识库存储、模型、索引任务和检索的配置与约束。"""
    backend: str = Field(default='pgvector', min_length=1)

    @field_validator('backend')
    @classmethod
    def supported_backend(cls, value):
        # 后端必须在存储适配层中明确实现。
        from ai_assistant.storage.backends import require_backend
        require_backend('knowledge', value)
        return value

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    database_url: SecretStr
    sources: dict[str, KnowledgeSource] = Field(default_factory=dict)

    # 分别限制索引、解析和向量化任务的并发及同步时限。
    index_concurrency: int = Field(default=8, gt=0)
    parse_concurrency: int = Field(default=8, gt=0)
    embedding_concurrency: int = Field(default=8, gt=0)
    sync_deadline_seconds: int = Field(default=14400, gt=0)

    # 证据长度、知识库标识及本地资料和诊断目录。
    evidence_chars: int = Field(default=24000, ge=1000, le=64000)
    knowledge_base_id: str = Field(default="default", min_length=1)
    materials_dir: Path = Path(".rag/materials")
    diagnostics_dir: Path = Path(".rag/diagnostics")

    # 分别配置向量化和重排服务；默认地址指向本机 HTTP 服务。
    embedding: ModelService = Field(
        default_factory=lambda: ModelService(
            base_url="http://127.0.0.1:1236/v1", model="qwen3-embedding-0.6b"
        )
    )
    reranker: ModelService = Field(
        default_factory=lambda: ModelService(
            base_url="http://127.0.0.1:1235/v1", model="qwen3-reranker-0.6b"
        )
    )
    query_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    # 文本分块参数及向量、关键词候选和最终结果数量。
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    vector_k: int = Field(default=40, gt=0)
    keyword_k: int = Field(default=40, gt=0)
    top_k: int = Field(default=8, gt=0)

    # 后台索引任务的租约、心跳、时限、重试和解析超时。
    lease_seconds: int = Field(default=60, ge=30)
    heartbeat_seconds: int = Field(default=10, gt=0)
    deadline_seconds: int = Field(default=14400, gt=0)
    max_attempts: int = Field(default=3, gt=0)
    retry_seconds: int = Field(default=10, ge=0)
    parse_timeout: int = Field(default=600, gt=0)

    @field_validator('sources')
    @classmethod
    def source_names(cls, value):
        """限制来源 ID 的长度与字符，确保配置名称格式稳定。"""
        import re
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', key) for key in value):
            raise ValueError('Source IDs require 1-64 ASCII letters, digits, underscores or hyphens')
        return value

    @model_validator(mode="after")
    def limits(self):
        """检查分块、租约心跳和数据库 URL 的跨字段约束。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.heartbeat_seconds * 2 >= self.lease_seconds:
            raise ValueError("heartbeat must be shorter than half the lease")
        if not self.database_url.get_secret_value().startswith("postgresql+asyncpg://"):
            raise ValueError("Use a postgresql+asyncpg URL")
        return self

    def fingerprint(self) -> str:
        """根据索引相关配置生成稳定指纹，用于区分不同构建设置。"""
        import hashlib, json

        # 指纹包含解析器、向量维数、模型名和分块方式，不包含凭据。
        data = dict(
            schema=3,
            parser="native-files-and-feishu-v1",
            dimensions=1024,
            embedding_model=self.embedding.model,
            query_instruction=self.query_instruction,
            splitter="recursive-characters",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def resolve_knowledge_paths(settings, base):
    """将资料与诊断目录相对 base 解析，返回更新后的配置副本。"""
    return settings.model_copy(
        update={
            name: (
                getattr(settings, name)
                if getattr(settings, name).is_absolute()
                else base / getattr(settings, name)
            ).resolve()
            for name in ("materials_dir", "diagnostics_dir")
        }
    )
