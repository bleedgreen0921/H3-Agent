"""Small session index next to a native Saver; never stores message history."""
from contextlib import asynccontextmanager, suppress
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from ai_assistant.config import ConfigError


class SessionError(RuntimeError):
    """Safe application errors; underlying database diagnostics stay private."""

    code = 'session_error'
    safe_message = '会话状态不可用，请稍后重试；本轮未执行。'
    retryable = False

    def __init__(self, message, *, code=None, safe_message=None, retryable=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        if safe_message is not None:
            self.safe_message = safe_message
        if retryable is not None:
            self.retryable = retryable


class SessionStorageError(SessionError):
    code = 'session_storage'
    safe_message = '会话存储暂不可用，请稍后重试；本轮未执行。'
    retryable = True


class SessionBusyError(SessionError):
    code = 'session_busy'
    safe_message = '会话正被另一个执行器占用，请稍后重试；本轮未执行。'
    retryable = True

    def __init__(self, message='Session is already in use.'):
        super().__init__(message)


class SessionNotFoundError(SessionError):
    code = 'session_not_found'
    safe_message = '目标会话不存在；本轮未执行。'


class SessionBindingError(SessionError):
    code = 'session_binding_mismatch'
    safe_message = '目标会话不属于当前聊天；本轮未执行。'


class SessionWorkspaceError(SessionError):
    code = 'session_workspace_mismatch'
    safe_message = '目标会话属于其他工作区；本轮未执行。'


class SessionFormatError(SessionError):
    code = 'session_format_unsupported'
    safe_message = '会话记录格式不兼容；请使用 /new 或按文档检查存储结构。'


class SessionDeletingError(SessionError):
    code = 'session_deleting'
    safe_message = '会话正在删除，尚未完成；本轮未执行。'



# 会话索引只保存状态、绑定和 checkpoint 定位，不保存完整对话正文。
DDL = '''CREATE TABLE IF NOT EXISTS assistant_sessions (
    id uuid PRIMARY KEY, workspace text NOT NULL, thread_id text NOT NULL,
    thread_ids text[] NOT NULL, complete jsonb,
    status text NOT NULL CHECK(status IN ('ready','running','interrupted','deleting')),
    format_version integer NOT NULL DEFAULT 1,
    binding text, title text NOT NULL DEFAULT '', last_input_id text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
)'''
SCHEMA_LOCK = 714160001


def session_uuid(value):
    """校验并规范化会话 UUID。"""
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise SessionFormatError('Invalid session ID.') from None


def lock_key(value):
    """将会话 ID 映射为 PostgreSQL advisory lock 的整数键。"""
    return int.from_bytes(sha256(('ai-assistant:session:' + value).encode()).digest()[:8],
                          'big', signed=True)


class SessionRepository:
    """管理会话索引、独占锁和原生 LangGraph Saver 定位。"""
    def __init__(self, conn, saver, *, lock_timeout_seconds=0.0):
        self.conn, self.saver = conn, saver
        self.current = None
        self.broken = False
        self.lock_timeout_seconds = max(0.0, float(lock_timeout_seconds))

    async def execute(self, sql, params=()):
        """统一执行 SQL；连接失败后标记仓储不可继续使用。"""
        if self.broken:
            raise SessionStorageError('Session connection lost; reopen the conversation.')
        try:
            return await self.conn.execute(sql, params)
        except Exception:
            self.broken = True
            raise SessionStorageError('Session database unavailable; check connection and run init.') from None

    async def check(self):
        await self.execute('SELECT 1')

    async def check_schema(self):
        """检查当前会话结构版本及原生 Saver 必需表。"""
        row = await (await self.execute(
            'SELECT version FROM assistant_session_schema WHERE singleton = true')).fetchone()
        if not row or row['version'] != 5:
            raise SessionFormatError('Unsupported session schema; run init with a compatible application.')
        await self.execute('SELECT binding,title,last_input_id FROM assistant_sessions LIMIT 0')
        # Also fail before any model calls if native Saver initialization is missing.
        for table in ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes', 'checkpoint_migrations'):
            await self.execute(f'SELECT 1 FROM {table} LIMIT 0')

    async def acquire(self, ident, *, wait_seconds=None):
        """尝试获取指定会话的连接级独占锁。"""
        ident = session_uuid(ident)
        from ai_assistant.storage.locks import try_advisory_lock
        try:
            locked = await try_advisory_lock(
                self.conn, lock_key(ident),
                wait_seconds=self.lock_timeout_seconds if wait_seconds is None else wait_seconds)
        except Exception:
            self.broken = True
            raise SessionStorageError('Session database unavailable; check connection and run init.') from None
        if not locked:
            raise SessionBusyError()
        self.current = ident
        return ident

    async def release(self):
        """释放当前连接持有的会话锁。"""
        if self.current is not None:
            ident, self.current = self.current, None
            if not self.broken:
                await self.execute('SELECT pg_advisory_unlock(%s)', (lock_key(ident),))

    async def open(self, workspace, ident=None, *, binding=None):
        """创建或打开会话，并检查聊天归属、格式与工作区。"""
        await self.release()
        creating = ident is None
        ident = await self.acquire(str(uuid4()) if creating else ident)
        if creating:
            thread = str(uuid4())
            await self.execute('''INSERT INTO assistant_sessions
                (id,workspace,thread_id,thread_ids,status,binding) VALUES(%s,%s,%s,%s,'ready',%s)''',
                (ident, str(workspace), thread, [thread], binding))
        row = await (await self.execute('SELECT * FROM assistant_sessions WHERE id=%s', (ident,))).fetchone()
        if not row:
            raise SessionNotFoundError('Session not found.')
        if binding is not None and row['binding'] != binding:
            raise SessionBindingError('会话不属于当前聊天。')
        if row['format_version'] != 1:
            raise SessionFormatError('Unsupported session format.')
        if row['status'] == 'deleting':
            raise SessionDeletingError('Session deletion is incomplete; retry session delete.')
        if row['workspace'] != str(Path(workspace).resolve()):
            raise SessionWorkspaceError('Session belongs to a different workspace.')
        return row

    async def binding_session(self, binding):
        return await (await self.execute(
            'SELECT s.* FROM assistant_sessions s JOIN assistant_channel_sessions c '
            'ON c.session_id=s.id WHERE c.binding=%s', (binding,))).fetchone()

    async def binding_session_id(self, binding):
        return await (await self.execute(
            'SELECT session_id FROM assistant_channel_sessions WHERE binding=%s', (binding,))).fetchone()

    async def session_binding(self, ident):
        return await (await self.execute(
            'SELECT binding FROM assistant_sessions WHERE id=%s', (str(ident),))).fetchone()

    async def list_binding(self, binding, offset=0, limit=11):
        return await (await self.execute('''SELECT s.id,s.title,s.updated_at,s.status,
            (c.session_id=s.id) AS current FROM assistant_sessions s
            LEFT JOIN assistant_channel_sessions c ON c.binding=s.binding
            WHERE s.binding=%s ORDER BY s.updated_at DESC,s.id LIMIT %s OFFSET %s''',
            (binding, limit, offset))).fetchall()

    async def list_admin(self, binding=None, offset=0, limit=10):
        return await (await self.execute('''SELECT s.id,s.title,s.binding,s.updated_at,s.status,
            EXISTS(SELECT 1 FROM assistant_channel_sessions c WHERE c.binding=s.binding AND c.session_id=s.id) AS current
            FROM assistant_sessions s WHERE (%s::text IS NULL OR s.binding=%s)
            ORDER BY s.updated_at DESC,s.id LIMIT %s OFFSET %s''',
            (binding, binding, limit, offset))).fetchall()

    async def status_for_binding(self, ident, binding):
        return await (await self.execute(
            'SELECT status FROM assistant_sessions WHERE id=%s AND binding=%s',
            (str(ident), binding))).fetchone()

    async def status(self, ident):
        return await (await self.execute(
            'SELECT status FROM assistant_sessions WHERE id=%s', (str(ident),))).fetchone()

    async def bind(self, binding, session_id):
        """绑定当前聊天与会话，拒绝覆盖不匹配的既有归属。"""
        cursor = await self.execute('''UPDATE assistant_sessions SET binding=%s
            WHERE id=%s AND (binding IS NULL OR binding=%s)''', (binding, session_id, binding))
        if cursor.rowcount != 1:
            raise SessionBindingError('会话不属于当前聊天。')
        await self.execute('''INSERT INTO assistant_channel_sessions(binding,session_id) VALUES(%s,%s)
            ON CONFLICT(binding) DO UPDATE SET session_id=EXCLUDED.session_id''', (binding, session_id))

    async def list(self):
        return await (await self.execute('''SELECT id,workspace,status,created_at,updated_at
            FROM assistant_sessions ORDER BY updated_at DESC,id''')).fetchall()

    async def update_current(self, sql, params):
        cursor = await self.execute(sql, params)
        if cursor.rowcount != 1:
            self.broken = True
            raise SessionStorageError('Session index disappeared; reopen the conversation.')

    async def start(self, thread):
        """标记会话运行中并登记使用过的原生线程 ID。"""
        await self.update_current('''UPDATE assistant_sessions SET status='running', thread_id=%s,
            thread_ids=CASE WHEN %s=ANY(thread_ids) THEN thread_ids ELSE array_append(thread_ids,%s) END,
            updated_at=clock_timestamp() WHERE id=%s''', (thread, thread, thread, self.current))

    async def complete(self, config, *, input_id=None, title=None):
        """将成功 checkpoint 的定位和可选标题提交到会话索引。"""
        from psycopg.types.json import Jsonb
        # Keep only the native state locator, never callbacks or runtime resources.
        c = config['configurable']
        locator = {'configurable': {k: c[k] for k in ('thread_id', 'checkpoint_ns', 'checkpoint_id')}}
        # 主题与成功指针同事务，避免失败轮次的主题提前发布。
        async with self.conn.transaction():
            if title is not None:
                await self.update_current('UPDATE assistant_sessions SET title=%s WHERE id=%s',
                                          (title, self.current))
            if input_id is not None:
                await self.update_current('''UPDATE assistant_sessions SET complete=%s,status='ready',
                    last_input_id=%s, updated_at=clock_timestamp() WHERE id=%s''',
                    (Jsonb(locator), input_id, self.current))
            else:
                await self.update_current('''UPDATE assistant_sessions SET complete=%s,status='ready',
                    updated_at=clock_timestamp() WHERE id=%s''', (Jsonb(locator), self.current))

    async def interrupted(self):
        """标记当前会话执行中断。"""
        await self.update_current("UPDATE assistant_sessions SET status='interrupted', updated_at=clock_timestamp() WHERE id=%s",
                           (self.current,))

    async def delete(self, ident, *, binding=None):
        """持锁标记删除，清理 Saver 线程后删除索引和聊天绑定。"""
        ident = session_uuid(ident)
        if self.current != ident:
            await self.release()
            ident = await self.acquire(ident)
        row = await (await self.execute('SELECT thread_ids,status,binding FROM assistant_sessions WHERE id=%s', (ident,))).fetchone()
        if not row or (binding is not None and row['binding'] != binding):
            raise SessionNotFoundError('未找到可删除的会话。')
        if binding is not None and row['status'] == 'running':
            raise SessionBusyError('会话正在执行，请先 /cancel 并等待清理结束。')

        # 先持久化 deleting；Saver 清理失败时保留状态供后续重试。
        await self.execute("UPDATE assistant_sessions SET status='deleting' WHERE id=%s", (ident,))
        try:
            for thread in row['thread_ids']:
                await self.saver.adelete_thread(thread)
        except Exception:
            raise SessionStorageError('Session deletion incomplete; retry session delete.') from None
        async with self.conn.transaction():
            exists = await (await self.execute("SELECT to_regclass('assistant_channel_sessions') AS name")).fetchone()
            if exists['name']:
                await self.execute('DELETE FROM assistant_channel_sessions WHERE session_id=%s', (ident,))
            await self.execute('DELETE FROM assistant_sessions WHERE id=%s', (ident,))

@asynccontextmanager
async def postgres_session_storage(settings, *, initialize=False):
    """打开会话连接和原生 Saver；仅显式初始化时创建当前结构。"""
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
                repo = SessionRepository(
                    conn, saver,
                    lock_timeout_seconds=getattr(settings, 'open_lock_timeout_seconds', 0.0))
                try:
                    if initialize:
                        await repo.execute('SELECT pg_advisory_lock(%s)', (SCHEMA_LOCK,))
                        try:
                            # 先拒绝旧结构；不在失败前升级原生 saver 或补建应用表。
                            present = await (await repo.execute("SELECT to_regclass('assistant_sessions') AS name")).fetchone()
                            if present['name']:
                                await repo.check_schema()
                            await saver.setup()
                            async with conn.transaction():
                                await repo.execute('''CREATE TABLE IF NOT EXISTS assistant_session_schema (
                                    singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton), version integer NOT NULL)''')
                                row = await (await repo.execute('SELECT version FROM assistant_session_schema')).fetchone()
                                if row and row['version'] != 5:
                                    raise SessionFormatError('Unsupported session schema version.')
                                await repo.execute(DDL)
                                await repo.execute('CREATE INDEX IF NOT EXISTS assistant_sessions_binding ON assistant_sessions(binding,updated_at DESC,id)')
                                await repo.execute('INSERT INTO assistant_session_schema VALUES(true,5) ON CONFLICT(singleton) DO NOTHING')
                        finally:
                            await repo.execute('SELECT pg_advisory_unlock(%s)', (SCHEMA_LOCK,))
                    await repo.check_schema()
                    if initialize:
                        # 核心结构已成功初始化；统计连接和建表失败单独提示。
                        try:
                            from ai_assistant.storage.usage import initialize_usage
                            await initialize_usage(settings)
                        except Exception as error:
                            import logging
                            logging.getLogger(__name__).warning(
                                '核心会话初始化成功；累计用量统计初始化失败：%s', type(error).__name__)
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


def session_storage(settings, *, initialize=False):
    """按显式配置选择会话存储后端。"""
    from ai_assistant.storage.backends import backend_factory
    if settings is None:
        from ai_assistant.config import ConfigError
        raise ConfigError('存储需要显式配置 [sessions]。')
    return backend_factory('sessions', settings.backend)(settings, initialize=initialize)
