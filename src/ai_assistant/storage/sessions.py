"""Small session index next to a native Saver; never stores message history."""
from contextlib import asynccontextmanager, suppress
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from ai_assistant.config import ConfigError


class SessionError(RuntimeError):
    """Safe application errors; underlying database diagnostics stay private."""


class SessionStorageError(SessionError):
    pass


DDL = '''CREATE TABLE IF NOT EXISTS assistant_sessions (
    id uuid PRIMARY KEY, workspace text NOT NULL, thread_id text NOT NULL,
    thread_ids text[] NOT NULL, complete jsonb,
    status text NOT NULL CHECK(status IN ('ready','running','interrupted','deleting')),
    format_version integer NOT NULL DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
)'''
SCHEMA_LOCK = 714160001


def session_uuid(value):
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise SessionError('Invalid session ID.') from None


def lock_key(value):
    return int.from_bytes(sha256(('ai-assistant:session:' + value).encode()).digest()[:8],
                          'big', signed=True)


class SessionRepository:
    def __init__(self, conn, saver):
        self.conn, self.saver = conn, saver
        self.current = None
        self.broken = False

    async def execute(self, sql, params=()):
        if self.broken:
            raise SessionStorageError('Session connection lost; reopen the conversation.')
        try:
            return await self.conn.execute(sql, params)
        except Exception:
            self.broken = True
            raise SessionStorageError('Session database unavailable; check connection and run session init.') from None

    async def check(self):
        await self.execute('SELECT 1')

    async def check_schema(self):
        row = await (await self.execute(
            'SELECT version FROM assistant_session_schema WHERE singleton = true')).fetchone()
        if not row or row['version'] != 1:
            raise SessionError('Unsupported session schema; run session init with a compatible application.')
        # Also fail before any model calls if native Saver initialization is missing.
        for table in ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes', 'checkpoint_migrations'):
            await self.execute(f'SELECT 1 FROM {table} LIMIT 0')

    async def acquire(self, ident):
        ident = session_uuid(ident)
        row = await (await self.execute('SELECT pg_try_advisory_lock(%s) AS locked',
                                       (lock_key(ident),))).fetchone()
        if not row['locked']:
            raise SessionError('Session is already in use.')
        self.current = ident
        return ident

    async def release(self):
        if self.current is not None:
            ident, self.current = self.current, None
            if not self.broken:
                await self.execute('SELECT pg_advisory_unlock(%s)', (lock_key(ident),))

    async def open(self, workspace, ident=None):
        await self.release()
        creating = ident is None
        ident = await self.acquire(str(uuid4()) if creating else ident)
        if creating:
            thread = str(uuid4())
            await self.execute('''INSERT INTO assistant_sessions
                (id,workspace,thread_id,thread_ids,status) VALUES(%s,%s,%s,%s,'ready')''',
                (ident, str(workspace), thread, [thread]))
        row = await (await self.execute('SELECT * FROM assistant_sessions WHERE id=%s', (ident,))).fetchone()
        if not row:
            raise SessionError('Session not found.')
        if row['format_version'] != 1:
            raise SessionError('Unsupported session format.')
        if row['status'] == 'deleting':
            raise SessionError('Session deletion is incomplete; retry session delete.')
        if row['workspace'] != str(Path(workspace).resolve()):
            raise SessionError('Session belongs to a different workspace.')
        return row

    async def list(self):
        return await (await self.execute('''SELECT id,workspace,status,created_at,updated_at
            FROM assistant_sessions ORDER BY updated_at DESC,id''')).fetchall()

    async def update_current(self, sql, params):
        cursor = await self.execute(sql, params)
        if cursor.rowcount != 1:
            self.broken = True
            raise SessionStorageError('Session index disappeared; reopen the conversation.')

    async def start(self, thread):
        await self.update_current('''UPDATE assistant_sessions SET status='running', thread_id=%s,
            thread_ids=CASE WHEN %s=ANY(thread_ids) THEN thread_ids ELSE array_append(thread_ids,%s) END,
            updated_at=clock_timestamp() WHERE id=%s''', (thread, thread, thread, self.current))

    async def complete(self, config, *, input_id=None):
        from psycopg.types.json import Jsonb
        # Keep only the native state locator, never callbacks or runtime resources.
        c = config['configurable']
        locator = {'configurable': {k: c[k] for k in ('thread_id', 'checkpoint_ns', 'checkpoint_id')}}
        if input_id is not None:
            await self.update_current('''UPDATE assistant_sessions SET complete=%s,status='ready',
                last_input_id=%s, updated_at=clock_timestamp() WHERE id=%s''',
                (Jsonb(locator), input_id, self.current))
        else:
            await self.update_current('''UPDATE assistant_sessions SET complete=%s,status='ready',
                updated_at=clock_timestamp() WHERE id=%s''', (Jsonb(locator), self.current))

    async def interrupted(self):
        await self.update_current("UPDATE assistant_sessions SET status='interrupted', updated_at=clock_timestamp() WHERE id=%s",
                           (self.current,))

    async def delete(self, ident):
        await self.release()
        ident = await self.acquire(ident)
        row = await (await self.execute('SELECT thread_ids FROM assistant_sessions WHERE id=%s', (ident,))).fetchone()
        if not row:
            raise SessionError('Session not found.')
        await self.execute("UPDATE assistant_sessions SET status='deleting' WHERE id=%s", (ident,))
        try:
            for thread in row['thread_ids']:
                await self.saver.adelete_thread(thread)
        except Exception:
            raise SessionStorageError('Session deletion incomplete; retry session delete.') from None
        await self.execute('DELETE FROM assistant_sessions WHERE id=%s', (ident,))


@asynccontextmanager
async def session_storage(settings, *, initialize=False):
    try:
        from psycopg import AsyncConnection, Error as DatabaseError
        from psycopg.rows import dict_row
        from psycopg.conninfo import make_conninfo
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        raise ConfigError('Install the sessions extra to use persistent conversations.') from None
    try:
        url = make_conninfo(settings.database_url.get_secret_value(), connect_timeout=10)
    except Exception:
        raise ConfigError('Invalid sessions database URL.') from None
    try:
        conn = await AsyncConnection.connect(url, autocommit=True, prepare_threshold=0,
                                             row_factory=dict_row, connect_timeout=10)
    except Exception:
        raise SessionStorageError('Unable to connect to session database.') from None
    async with conn:
        try:
            async with AsyncPostgresSaver.from_conn_string(url) as saver:
                repo = SessionRepository(conn, saver)
                try:
                    if initialize:
                        await repo.execute('SELECT pg_advisory_lock(%s)', (SCHEMA_LOCK,))
                        try:
                            await saver.setup()
                            async with conn.transaction():
                                await repo.execute('''CREATE TABLE IF NOT EXISTS assistant_session_schema (
                                    singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton), version integer NOT NULL)''')
                                row = await (await repo.execute('SELECT version FROM assistant_session_schema')).fetchone()
                                if row and row['version'] != 1:
                                    raise SessionError('Unsupported session schema version.')
                                await repo.execute(DDL)
                                await repo.execute('INSERT INTO assistant_session_schema VALUES(true,1) ON CONFLICT DO NOTHING')
                        finally:
                            await repo.execute('SELECT pg_advisory_unlock(%s)', (SCHEMA_LOCK,))
                    await repo.check_schema()
                    yield repo
                finally:
                    # Closing the connection releases its locks even if explicit
                    # unlock fails. Cleanup must not mask a caller exception.
                    with suppress(SessionStorageError):
                        await repo.release()
        except (SessionError, ConfigError):
            raise
        except DatabaseError:
            raise SessionStorageError('Session database operation failed; check connection and schema.') from None
