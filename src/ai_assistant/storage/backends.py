"""存储后端的显式创建入口；新后端仍须适配 repository、checkpoint 和 SQL。"""
from importlib import import_module

BACKENDS = {
    'sessions': {'postgresql': ('ai_assistant.storage.sessions', 'postgres_session_storage')},
    'operations': {'postgresql': ('ai_assistant.storage.integrations', 'postgres_operation_storage')},
    'usage': {'postgresql': ('ai_assistant.storage.usage', 'postgres_usage_storage')},
    'knowledge': {'pgvector': ('ai_assistant.storage.rag', 'PgvectorStore')},
    'vectors': {'pgvector': ('ai_assistant.knowledge.ingestion', 'pgvector_store')},
}


def require_backend(kind, backend):
    if backend not in BACKENDS[kind]:
        raise ValueError('该存储后端尚未实现；未创建连接或回退。')


def backend_factory(kind, backend):
    from ai_assistant.config import ConfigError
    try:
        require_backend(kind, backend)
    except ValueError:
        raise ConfigError('该存储后端尚未实现；未创建连接或回退。') from None
    module, name = BACKENDS[kind][backend]
    return getattr(import_module(module), name)
