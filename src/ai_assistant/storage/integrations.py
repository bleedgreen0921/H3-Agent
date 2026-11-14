"""Durable external effects, separate from checkpoint transactions."""
import hashlib
import json
from contextlib import asynccontextmanager
from uuid import uuid4

from ai_assistant.config import ConfigError

# 显式初始化时创建操作、命令、聊天绑定及输入输出队列表。
DDL = '''
CREATE TABLE IF NOT EXISTS assistant_chat_commands (
 input_id text PRIMARY KEY, binding text NOT NULL, action text NOT NULL,
 target text, result text, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS assistant_operations (
 id text PRIMARY KEY, input_id text NOT NULL, principal text NOT NULL,
 tool text NOT NULL, args_digest text NOT NULL, source_key text,
 status text NOT NULL CHECK(status IN ('pending','succeeded','failed','unknown')),
 result jsonb, created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(input_id,principal,tool,args_digest)
);
CREATE UNIQUE INDEX IF NOT EXISTS assistant_operations_source_key ON assistant_operations(source_key)
 WHERE source_key IS NOT NULL;
CREATE TABLE IF NOT EXISTS assistant_channel_sessions (
 binding text PRIMARY KEY, session_id uuid NOT NULL
);
CREATE TABLE IF NOT EXISTS assistant_inbox (
 id text PRIMARY KEY, binding text NOT NULL, body jsonb,
 status text NOT NULL DEFAULT 'received', received_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS assistant_outbox (
 id text PRIMARY KEY, input_id text NOT NULL REFERENCES assistant_inbox(id),
 body jsonb, status text NOT NULL DEFAULT 'pending', remote_id text,
 created_at timestamptz NOT NULL DEFAULT now(), part integer NOT NULL DEFAULT 0,
 next_attempt_at timestamptz NOT NULL DEFAULT now()
);
'''


class OperationStore:
    """在同一连接上组合外部操作、渠道队列和诊断仓储。"""

    def __init__(self, conn):
        self.conn = conn
        from ai_assistant.storage.commands import CommandRepository
        from ai_assistant.storage.diagnostics import DiagnosticsRepository
        from ai_assistant.storage.operations import OperationRepository
        from ai_assistant.storage.queues import InboxRepository, OutboxRepository
        self.operations = OperationRepository(conn)
        self.inbox = InboxRepository(conn)
        self.outbox = OutboxRepository(conn)
        self.command_records = CommandRepository(conn)
        self.diagnostics = DiagnosticsRepository(conn)

    async def initialize(self):
        """在事务锁保护下显式初始化存储结构。"""
        async with self.conn.transaction():
            await self.conn.execute('SELECT pg_advisory_xact_lock(714160002)')
            present = await (await self.conn.execute("SELECT to_regclass('assistant_outbox') AS name")).fetchone()
            if present['name']:
                await self.conn.execute('SELECT next_attempt_at,part FROM assistant_outbox LIMIT 0')
            await self.conn.execute(DDL)



@asynccontextmanager
async def postgres_operation_storage(settings, *, initialize=False):
    """打开 PostgreSQL 操作存储；仅在显式要求时初始化结构。"""
    if settings is None:
        raise ConfigError('External operations require [sessions]; run init first.')
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    conn = await AsyncConnection.connect(settings.database_url.get_secret_value(),
        autocommit=True, row_factory=dict_row, connect_timeout=10)
    async with conn:
        store = OperationStore(conn)
        if initialize:
            await store.initialize()

        # 连接后检查所需结构；缺失时失败，不自动迁移。
        await conn.execute('SELECT id,source_key FROM assistant_operations LIMIT 0')
        await conn.execute('SELECT last_input_id FROM assistant_sessions LIMIT 0')
        await conn.execute('SELECT input_id FROM assistant_chat_commands LIMIT 0')
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
        digest = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if source_key:
            existing = await self.by_source(source_key)
            if existing:
                return False, existing

        # 内存版只记录参数摘要，不保存原始参数。
        key = (input_id, principal, tool, digest)
        if key in self.rows:
            return False, dict(self.rows[key])
        row = dict(id=str(uuid4()), input_id=input_id, principal=principal, tool=tool,
                   args_digest=digest, source_key=source_key, status='pending', result=None)
        self.rows[key] = row
        return True, dict(row)

    async def by_source(self, source_key):
        """按来源键查找已登记操作。"""
        return next((dict(row) for row in self.rows.values() if row.get('source_key') == source_key), None)

    async def finish(self, ident, status, result=None):
        """更新操作状态或为相同终态补充回执。"""
        row = next(row for row in self.rows.values() if row['id'] == ident)
        # 与数据库版同规则：终态只接受同状态补充回执并合并 result，跨终态改判直接忽略。
        if row['status'] != 'pending' and row['status'] != status:
            return
        row['status'] = status
        if result is not None:
            merged = dict(row['result'] or {})
            merged.update(result)
            row['result'] = merged

    async def list(self, input_id=None):
        """列出当前内存存储中的操作副本。"""
        return [dict(row) for row in self.rows.values() if input_id is None or row['input_id'] == input_id]


def operation_storage(settings, *, initialize=False):
    """根据显式配置选择操作存储后端。"""
    from ai_assistant.storage.backends import backend_factory
    if settings is None:
        from ai_assistant.config import ConfigError
        raise ConfigError('存储需要显式配置 [sessions]。')
    return backend_factory('operations', settings.backend)(settings, initialize=initialize)
