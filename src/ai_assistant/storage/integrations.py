"""Durable external effects, separate from checkpoint transactions."""
import hashlib
import json
from contextlib import asynccontextmanager
from uuid import uuid4

from ai_assistant.config import ConfigError

DDL = '''
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
 created_at timestamptz NOT NULL DEFAULT now(), part integer NOT NULL DEFAULT 0
);
ALTER TABLE assistant_outbox ADD COLUMN IF NOT EXISTS part integer NOT NULL DEFAULT 0;
ALTER TABLE assistant_sessions ADD COLUMN IF NOT EXISTS last_input_id text;
ALTER TABLE assistant_outbox ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'text';
ALTER TABLE assistant_outbox ADD COLUMN IF NOT EXISTS display_version bigint NOT NULL DEFAULT 0;
ALTER TABLE assistant_outbox ADD COLUMN IF NOT EXISTS delivered_version bigint NOT NULL DEFAULT 0;
ALTER TABLE assistant_outbox ADD COLUMN IF NOT EXISTS next_attempt_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE assistant_outbox ADD COLUMN IF NOT EXISTS final_state text NOT NULL DEFAULT 'stage';
'''


class OperationStore:
    def __init__(self, conn):
        self.conn = conn

    async def initialize(self):
        async with self.conn.transaction():
            await self.conn.execute('SELECT pg_advisory_xact_lock(714160002)')
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
        await self.conn.execute('''UPDATE assistant_operations SET status=%s,result=COALESCE(%s,result),
            updated_at=now() WHERE id=%s''', (status, Jsonb(result) if result is not None else None, ident))

    async def list(self, input_id=None):
        if input_id:
            cursor = await self.conn.execute('SELECT * FROM assistant_operations WHERE input_id=%s ORDER BY created_at', (input_id,))
        else:
            cursor = await self.conn.execute('SELECT * FROM assistant_operations ORDER BY created_at DESC LIMIT 100')
        return await cursor.fetchall()

    async def replies(self):
        return await (await self.conn.execute('''SELECT id,input_id,status,remote_id,part,created_at
            FROM assistant_outbox ORDER BY created_at DESC LIMIT 100''')).fetchall()

    async def inspect(self, ident):
        row = await (await self.conn.execute('SELECT * FROM assistant_operations WHERE id=%s', (ident,))).fetchone()
        if row is not None:
            return row
        return await (await self.conn.execute('''SELECT id,input_id,status,remote_id,part,created_at,
            'channel_reply' AS kind FROM assistant_outbox WHERE id=%s''', (ident,))).fetchone()


@asynccontextmanager
async def operation_storage(settings, *, initialize=False):
    if settings is None:
        raise ConfigError('External operations require [sessions]; run mcp init first.')
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
        yield store
