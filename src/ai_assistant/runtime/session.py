"""会话身份、checkpoint 恢复、并发互斥及连接状态。"""
import asyncio
from uuid import uuid4

from ai_assistant.config import ConfigError
from ai_assistant.storage.sessions import SessionError, SessionStorageError
from ai_assistant.runtime.turns import execute_turn
from ai_assistant.runtime.maintenance import maintain_session_memory
from ai_assistant.observability import event, span


class AssistantSession:
    def __init__(self, agent, context, settings, *, model_name='', repository=None):
        self.agent, self.context, self.settings = agent, context, settings
        self.model_name = model_name
        self.repository = repository
        self._running = False
        self._turn_lock = asyncio.Lock()
        self._closed = False
        self._active_task = None
        self._storage_lost = False
        self._reset()

    @property
    def session_id(self):
        return self._session_id

    def _reset(self):
        self._session_id = str(uuid4())
        self._thread = str(uuid4())
        self._threads = [self._thread]
        self._complete = None
        self._failed = False
        self._maintenance = None

    async def open(self, session_id=None):
        if self._closed:
            raise RuntimeError('Session is closed.')
        if self.repository is None:
            if session_id is not None:
                raise ConfigError('Opening a session requires [sessions].')
            self._reset()
            return
        caller = self.context.caller
        binding = caller.binding if caller and caller.scope != 'local' else None
        try:
            row = await self.repository.open(self.context.workspace.root, session_id, binding=binding)
        except SessionError as error:
            event('session.open_failed', status='failed',
                  code=getattr(error, 'code', 'session_error'),
                  retryable=getattr(error, 'retryable', False))
            raise
        self._session_id = str(row['id'])
        self._thread, self._complete = row['thread_id'], row['complete']
        self._threads = row.get('thread_ids') or [self._thread]
        self._failed = row['status'] in {'running', 'interrupted'}
        if self._complete is not None:
            state = await self.agent.aget_state(self._complete)
            from ai_assistant.runtime.history import saved_turn
            if not state.values or state.next or saved_turn(state.values, threads=self._threads) is None:
                event('session.open_failed', status='failed',
                      code='session_checkpoint_invalid', retryable=False)
                raise SessionError('不支持的 checkpoint 格式或正文缺失；未恢复或重新执行。')

    async def new_conversation(self):
        if self._running or self._turn_lock.locked():
            raise RuntimeError('A turn is already running.')
        await self.open()

    async def watch_connection(self):
        """Loss of the advisory-lock connection invalidates this open session."""
        while True:
            try:
                await self.repository.check()
            except SessionStorageError:
                self._storage_lost = True
                if self._active_task is not None and not self._active_task.cancelling():
                    self._active_task.cancel()
                return
            await asyncio.sleep(1)

    async def run_turn(self, text, **kwargs):
        kwargs["input_id"] = kwargs.get("input_id") or str(uuid4())
        with span("agent.turn", input_id=kwargs["input_id"], session_id=self.session_id, execution_thread_id=self._thread):
            return await self._locked_turn(text, **kwargs)

    async def _locked_turn(self, text, **kwargs):
        # 输入校验也会读取 checkpoint，必须纳入同一轮互斥范围。
        if self._turn_lock.locked():
            raise RuntimeError('A turn is already running.')
        async with self._turn_lock:
            return await self._run_turn(text, **kwargs)

    async def _run_turn(self, text, **kwargs):
        # 会话拥有状态与互斥锁，轮次模块负责执行和提交，不另建模型调度循环。
        return await execute_turn(self, text, **kwargs)

    async def after_reply(self):
        """入口完成回复处理后，按配置更新本会话记忆。"""
        return await maintain_session_memory(self)
