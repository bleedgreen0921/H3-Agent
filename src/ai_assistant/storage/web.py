"""本地 Web 对话展示消息；checkpoint 仍是业务事实来源。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from ai_assistant.config import ConfigError


DDL = '''CREATE TABLE IF NOT EXISTS assistant_web_messages (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    input_id text,
    role text NOT NULL CHECK(role IN ('user','assistant','system')),
    text text NOT NULL,
    status text NOT NULL DEFAULT 'complete',
    attachment_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
)'''


@asynccontextmanager
async def web_store(settings):
    try:
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row
    except ImportError:
        raise ConfigError("安装 sessions 依赖以使用 Web 会话。") from None
    if settings is None:
        raise ConfigError("Web 入口需要配置 [sessions]。")
    conn = await AsyncConnection.connect(
        settings.database_url.get_secret_value(), autocommit=True,
        row_factory=dict_row, connect_timeout=10)
    try:
        await conn.execute("SELECT id FROM assistant_web_messages LIMIT 0")
        yield conn
    finally:
        await conn.close()


async def initialize(settings):
    from psycopg import AsyncConnection
    conn = await AsyncConnection.connect(settings.database_url.get_secret_value(),
                                         autocommit=True, connect_timeout=10)
    try:
        await conn.execute(DDL)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS assistant_web_messages_session "
            "ON assistant_web_messages(session_id, created_at, id)")
    finally:
        await conn.close()


class WebMessageStore:
    def __init__(self, settings):
        self.settings = settings

    async def append(self, session_id, role, text, *, input_id=None,
                     status="complete", attachment_ids=()):
        from psycopg.types.json import Jsonb
        async with web_store(self.settings) as conn:
            await conn.execute(
                '''INSERT INTO assistant_web_messages
                (id,session_id,input_id,role,text,status,attachment_ids)
                VALUES(%s,%s,%s,%s,%s,%s,%s)''',
                (str(uuid4()), session_id, input_id, role, text, status,
                 Jsonb(list(attachment_ids or ()))))

    async def list(self, session_id, limit=200):
        async with web_store(self.settings) as conn:
            return await (await conn.execute(
                '''SELECT * FROM assistant_web_messages WHERE session_id=%s
                ORDER BY created_at,id LIMIT %s''', (session_id, limit))).fetchall()
