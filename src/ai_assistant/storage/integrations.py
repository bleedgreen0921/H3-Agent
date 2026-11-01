"""Durable external effects, separate from checkpoint transactions."""
import hashlib
import json
from contextlib import asynccontextmanager
from uuid import uuid4

from ai_assistant.config import ConfigError

DDL = '''
CREATE TABLE IF NOT EXISTS assistant_chat_commands (
 input_id text PRIMARY KEY, binding text NOT NULL, action text NOT NULL,
 target text, result text, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS assistant_operations (
 id text PRIMARY KEY, input_id text NOT NULL, principal text NOT NULL,
 tool text NOT NULL, args_digest text NOT NULL,
 status text NOT NULL CHECK(status IN ('pending','succeeded','failed','unknown')),
 result jsonb, created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(input_id,principal,tool,args_digest)
);
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
 kind text NOT NULL DEFAULT 'text', display_version bigint NOT NULL DEFAULT 0,
 delivered_version bigint NOT NULL DEFAULT 0, next_attempt_at timestamptz NOT NULL DEFAULT now(),
 final_state text NOT NULL DEFAULT 'stage'
);
'''


class OperationStore:
    def __init__(self, conn):
        self.conn = conn

    async def initialize(self):
        async with self.conn.transaction():
            await self.conn.execute('SELECT pg_advisory_xact_lock(714160002)')
            present = await (await self.conn.execute("SELECT to_regclass('assistant_outbox') AS name")).fetchone()
            if present['name']:
                await self.conn.execute('SELECT kind,display_version,delivered_version,next_attempt_at,final_state,part FROM assistant_outbox LIMIT 0')
            await self.conn.execute(DDL)

    async def begin(self, input_id, principal, tool, arguments):
        digest = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False,
                                          separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        ident = str(uuid4())
        row = await (await self.conn.execute('''INSERT INTO assistant_operations
            (id,input_id,principal,tool,args_digest,status) VALUES(%s,%s,%s,%s,%s,'pending')
            ON CONFLICT DO NOTHING RETURNING *''', (ident, input_id, principal, tool, digest))).fetchone()
        if row:
            return True, row
        row = await (await self.conn.execute('''SELECT * FROM assistant_operations
            WHERE input_id=%s AND principal=%s AND tool=%s AND args_digest=%s''',
            (input_id, principal, tool, digest))).fetchone()
        return False, row

    async def finish(self, ident, status, result=None):
        from psycopg.types.json import Jsonb
        # pending 可转任意终态；已终态仅允许同状态补充回执（jsonb 顶层合并，原键保留）。
        # 跨终态改判（含迟到 unknown/failed 降级 succeeded）条件不满足、0 行更新、静默忽略。
        payload = Jsonb(result) if result is not None else None
        await self.conn.execute('''UPDATE assistant_operations SET status=%s,
            result=COALESCE(result || %s, %s, result), updated_at=now()
            WHERE id=%s AND (status='pending' OR status=%s)''',
            (status, payload, payload, ident, status))

    async def list(self, input_id=None):
        if input_id:
            cursor = await self.conn.execute('SELECT * FROM assistant_operations WHERE input_id=%s ORDER BY created_at', (input_id,))
        else:
            cursor = await self.conn.execute('SELECT * FROM assistant_operations ORDER BY created_at DESC LIMIT 100')
        return await cursor.fetchall()

    async def replies(self):
        return await (await self.conn.execute('''SELECT id,input_id,status,remote_id,part,created_at,
            display_version,delivered_version,final_state,body->'delivery' AS delivery,
            (body->'delivery'->>'version'='1') IS NOT TRUE AS unsupported_format,'channel_reply' AS kind
            FROM assistant_outbox ORDER BY created_at DESC LIMIT 100''')).fetchall()

    async def commands(self, ident=None):
        return await (await self.conn.execute("""SELECT input_id,binding,action,target,
            CASE WHEN result IS NULL THEN 'incomplete' ELSE 'completed' END AS status,
            created_at,result AS receipt,'management_command' AS kind FROM assistant_chat_commands
            WHERE (%s::text IS NULL OR input_id=%s) ORDER BY created_at DESC LIMIT 100""",
            (ident, ident))).fetchall()

    async def inspect(self, ident):
        row = await (await self.conn.execute("""SELECT id,input_id,tool,principal,status,
            created_at,updated_at,result->>'version' AS receipt_version,
            COALESCE(result->>'verification','unverified') AS verification,
            (COALESCE(result->>'verification','unverified')!='passed') AS needs_verification,
            'tool_operation' AS kind FROM assistant_operations WHERE id=%s""", (ident,))).fetchone()
        if row is not None:
            return row
        row = await (await self.conn.execute("""SELECT id,input_id,status,remote_id,part,created_at,
            display_version,delivered_version,final_state,body->'delivery' AS delivery,
            (body->'delivery'->>'version'='1') IS NOT TRUE AS unsupported_format,
            'channel_reply' AS kind FROM assistant_outbox WHERE id=%s""", (ident,))).fetchone()
        return row if row is not None else await self.commands(ident)


@asynccontextmanager
async def postgres_operation_storage(settings, *, initialize=False):
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
        await conn.execute('SELECT id FROM assistant_operations LIMIT 0')
        await conn.execute('SELECT last_input_id FROM assistant_sessions LIMIT 0')
        await conn.execute('SELECT input_id FROM assistant_chat_commands LIMIT 0')
        yield store


class MemoryOperationStore:
    """仅当前会话去重；不提供崩溃恢复保证，也不保存原始参数。"""

    def __init__(self):
        self.rows = {}

    async def begin(self, input_id, principal, tool, arguments):
        digest = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        key = (input_id, principal, tool, digest)
        if key in self.rows:
            return False, dict(self.rows[key])
        row = dict(id=str(uuid4()), input_id=input_id, principal=principal, tool=tool,
                   args_digest=digest, status='pending', result=None)
        self.rows[key] = row
        return True, dict(row)

    async def finish(self, ident, status, result=None):
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
        return [dict(row) for row in self.rows.values() if input_id is None or row['input_id'] == input_id]


def operation_storage(settings, *, initialize=False):
    from ai_assistant.storage.backends import backend_factory
    if settings is None:
        from ai_assistant.config import ConfigError
        raise ConfigError('存储需要显式配置 [sessions]。')
    return backend_factory('operations', settings.backend)(settings, initialize=initialize)
