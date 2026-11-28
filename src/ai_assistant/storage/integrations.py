"""外部操作回执仓储；与 checkpoint 和 Web 消息使用不同事务边界。"""
from contextlib import asynccontextmanager
from hashlib import sha256
import json
from uuid import uuid4

from ai_assistant.config import ConfigError

# 只保留外部效果回执；不创建渠道投递队列。
DDL = '''CREATE TABLE IF NOT EXISTS assistant_operations (
 id text PRIMARY KEY, input_id text NOT NULL, principal text NOT NULL,
 tool text NOT NULL, args_digest text NOT NULL, source_key text,
 status text NOT NULL CHECK(status IN ('pending','succeeded','failed','unknown')),
 result jsonb, created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(input_id,principal,tool,args_digest)
);
CREATE UNIQUE INDEX IF NOT EXISTS assistant_operations_source_key ON assistant_operations(source_key)
 WHERE source_key IS NOT NULL;
'''


class OperationStore:
    """当前进程使用的操作回执仓储。"""

    def __init__(self, conn):
        self.conn = conn
        from ai_assistant.storage.operations import OperationRepository
        self.operations = OperationRepository(conn)

    async def initialize(self):
        """只创建当前回执结构；不触碰旧渠道表。"""
        async with self.conn.transaction():
            await self.conn.execute("SELECT pg_advisory_xact_lock(714160002)")
            await self.conn.execute(DDL)


@asynccontextmanager
async def postgres_operation_storage(settings, *, initialize=False):
    """打开 PostgreSQL 操作存储；仅在显式要求时初始化结构。"""
    if settings is None:
        raise ConfigError("External operations require [sessions]; run init first.")
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    conn = await AsyncConnection.connect(settings.database_url.get_secret_value(),
                                          autocommit=True, row_factory=dict_row,
                                          connect_timeout=10)
    async with conn:
        store = OperationStore(conn)
        if initialize:
            await store.initialize()
        await conn.execute("SELECT id,source_key FROM assistant_operations LIMIT 0")
        yield store


class MemoryOperationStore:
    """仅当前会话去重；不提供崩溃恢复保证，也不保存原始参数。"""

    def __init__(self):
        self.rows = {}

    @property
    def operations(self):
        return self

    async def begin(self, input_id, principal, tool, arguments, source_key=None):
        """根据规范化参数摘要或来源键复用已有操作。"""
        digest = sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if source_key:
            existing = await self.by_source(source_key)
            if existing:
                return False, existing
        key = (input_id, principal, tool, digest)
        if key in self.rows:
            return False, dict(self.rows[key])
        row = dict(id=str(uuid4()), input_id=input_id, principal=principal,
                   tool=tool, args_digest=digest, source_key=source_key,
                   status='pending', result=None)
        self.rows[key] = row
        return True, dict(row)

    async def by_source(self, source_key):
        return next((dict(row) for row in self.rows.values()
                     if row.get('source_key') == source_key), None)

    async def finish(self, ident, status, result=None):
        row = next(row for row in self.rows.values() if row['id'] == ident)
        if row['status'] != 'pending' and row['status'] != status:
            return
        row['status'] = status
        if result is not None:
            merged = dict(row['result'] or {})
            merged.update(result)
            row['result'] = merged

    async def list(self, input_id=None):
        return [dict(row) for row in self.rows.values()
                if input_id is None or row['input_id'] == input_id]


def operation_storage(settings, *, initialize=False):
    """根据显式配置选择操作存储后端。"""
    from ai_assistant.storage.backends import backend_factory
    if settings is None:
        raise ConfigError('存储需要显式配置 [sessions]。')
    return backend_factory('operations', settings.backend)(settings, initialize=initialize)
