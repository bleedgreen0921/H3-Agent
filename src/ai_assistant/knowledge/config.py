"""Optional, explicit local knowledge settings."""
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class KnowledgeConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    database_url: SecretStr
    knowledge_base_id: str = Field(default='default', min_length=1)
    materials_dir: Path = Path('.rag/materials')
    diagnostics_dir: Path = Path('.rag/diagnostics')
    embedding_model: Path = Path('/home/qjj/models/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master')
    reranker_model: Path = Path('/home/qjj/models/models/BAAI--bge-reranker-v2-m3/snapshots/master')
    device: str = 'cuda'
    embedding_batch: int = Field(default=4, gt=0)
    reranker_batch: int = Field(default=4, gt=0)
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

    @model_validator(mode='after')
    def limits(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('chunk_overlap must be smaller than chunk_size')
        if self.heartbeat_seconds * 2 >= self.lease_seconds:
            raise ValueError('heartbeat must be shorter than half the lease')
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError('device must explicitly select cuda or cpu')
        if not self.database_url.get_secret_value().startswith('postgresql+asyncpg://'):
            raise ValueError('Use a postgresql+asyncpg URL')
        return self

    def fingerprint(self) -> str:
        import hashlib, json
        fields = ('embedding_model', 'chunk_size', 'chunk_overlap')
        data = {key: str(getattr(self, key)) for key in fields}
        data.update(schema=1, parser='unstructured-local', dimensions=1024, tokenizer='embedding')
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def resolve_knowledge_paths(settings, base):
    """Resolve all local RAG paths relative to the configuration file."""
    return settings.model_copy(update={
        name: (getattr(settings, name) if getattr(settings, name).is_absolute()
               else base / getattr(settings, name)).resolve()
        for name in ('materials_dir', 'diagnostics_dir', 'embedding_model', 'reranker_model')
    })
