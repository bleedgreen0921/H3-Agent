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
    kind: Literal['local_directory', 'feishu_wiki', 'feishu_folder']
    path: Path | None = None
    token: str | None = None
    node_token: str | None = None
    identity: Literal['user', 'bot'] = 'user'
    include: list[str] = Field(default_factory=lambda: ['*'])
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def target(self):
        if self.kind == 'local_directory' and self.path is None:
            raise ValueError('local_directory requires path')
        if self.kind != 'local_directory' and not self.token:
            raise ValueError('Feishu source requires token')
        return self


class ModelService(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    base_url: str
    model: str = Field(min_length=1)
    api_key: SecretStr = SecretStr("local")
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=4, gt=0, le=256)

    @field_validator("base_url")
    @classmethod
    def service_url(cls, value):
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
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    database_url: SecretStr
    sources: dict[str, KnowledgeSource] = Field(default_factory=dict)
    sync_deadline_seconds: int = Field(default=14400, gt=0)
    evidence_chars: int = Field(default=8000, ge=1000, le=20000)
    knowledge_base_id: str = Field(default="default", min_length=1)
    materials_dir: Path = Path(".rag/materials")
    diagnostics_dir: Path = Path(".rag/diagnostics")
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
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    vector_k: int = Field(default=20, gt=0)
    keyword_k: int = Field(default=20, gt=0)
    top_k: int = Field(default=5, gt=0)
    lease_seconds: int = Field(default=60, ge=30)
    heartbeat_seconds: int = Field(default=10, gt=0)
    deadline_seconds: int = Field(default=14400, gt=0)
    max_attempts: int = Field(default=3, gt=0)
    retry_seconds: int = Field(default=10, ge=0)
    parse_timeout: int = Field(default=600, gt=0)

    @field_validator('sources')
    @classmethod
    def source_names(cls, value):
        import re
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', key) for key in value):
            raise ValueError('Source IDs require 1-64 ASCII letters, digits, underscores or hyphens')
        return value

    @model_validator(mode="after")
    def limits(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.heartbeat_seconds * 2 >= self.lease_seconds:
            raise ValueError("heartbeat must be shorter than half the lease")
        if not self.database_url.get_secret_value().startswith("postgresql+asyncpg://"):
            raise ValueError("Use a postgresql+asyncpg URL")
        return self

    def fingerprint(self) -> str:
        import hashlib, json

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
    """Resolve all local RAG paths relative to the configuration file."""
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
