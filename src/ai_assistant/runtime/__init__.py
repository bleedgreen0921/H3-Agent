"""Native turns, durable completion pointers and resource ownership."""
import asyncio
import inspect
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent
from ai_assistant.config import ConfigError
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.requests import model_session
from ai_assistant.tools.builtin import create_builtin_tools
from ai_assistant.tools.context import tool_context
from ai_assistant.tools.terminal import _drain_worker
from ai_assistant.storage.sessions import SessionError, SessionStorageError, session_storage
from ai_assistant.runtime.observation import TurnObserver, TurnUsage

RECOVERY_NOTICE = ('应用状态：上一轮未完成，已发生的文件或命令效果不会回滚。'
                   '请先核实可能变化的工作区；不要恢复执行上一轮请求。')
REOPEN_NOTICE = ('应用状态：本会话已重新打开，工作区及工具权限以当前配置为准。'
                 '请重新核实工作区和当前可用工具，不恢复执行旧请求。')


@dataclass(frozen=True)
class AgentTurn:
    messages: list[BaseMessage]
    sources: list[Document]
    usage: TurnUsage = field(default_factory=TurnUsage)


def final_text(messages):
    if not messages or not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
        return ''
    return messages[-1].text.strip()


class AssistantSession:
    def __init__(self, agent, context, settings, *, model_name='', tools=(), repository=None):
        self.agent, self.context, self.settings = agent, context, settings
        self.model_name, self.tools = model_name, tuple(tools)
        self.repository = repository
        self._running = False
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
        self._complete = None
        self._failed = False
        self._reopened = False

    async def open(self, session_id=None):
        if self._closed:
            raise RuntimeError('Session is closed.')
        if self.repository is None:
            if session_id is not None:
                raise ConfigError('Opening a session requires [sessions].')
            self._reset()
            return
        row = await self.repository.open(self.context.workspace.root, session_id)
        self._session_id = str(row['id'])
        self._thread, self._complete = row['thread_id'], row['complete']
        self._failed = row['status'] in {'running', 'interrupted'}
        self._reopened = session_id is not None
        if self._complete is not None:
            state = await self.agent.aget_state(self._complete)
            if not state.values or state.next or not final_text(state.values.get('messages', [])):
                raise SessionError('Stored successful checkpoint is missing or incompatible.')

    async def new_conversation(self):
        if self._running:
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

    async def run_turn(self, text, *, on_event=None):
        if self._closed:
            raise RuntimeError('Session is closed.')
        if self._running:
            raise RuntimeError('A turn is already running.')
        if self._storage_lost:
            raise SessionStorageError('Session connection lost; reopen the conversation.')
        self._running = True
        self._active_task = asyncio.current_task()
        try:
            if self._complete is None and self._failed:
                self._thread = str(uuid4())
            if self.repository is not None:
                await self.repository.start(self._thread)
            observer = TurnObserver(self.settings.model_call_limit, self.model_name)
            locator = self._complete or {'configurable': {'thread_id': self._thread}}
            config = {**locator, 'callbacks': [observer],
                      'metadata': {'assistant_model_profile': self.model_name}}
            inputs = []
            if self._failed:
                inputs.append(HumanMessage(content=RECOVERY_NOTICE))
            if self._reopened:
                inputs.append(HumanMessage(content=REOPEN_NOTICE))
            inputs.append(HumanMessage(content=text))
            result = None
            root_id = None
            async with asyncio.timeout(self.settings.turn_timeout_seconds):
                async for event in self.agent.astream_events(
                    {'messages': inputs}, config=config, context=self.context,
                    version='v2', durability='sync'
                ):
                    if root_id is None:
                        root_id = event['run_id']
                    if event['event'] in {'on_tool_start', 'on_tool_end', 'on_tool_error'} and on_event:
                        returned = on_event(event)
                        if inspect.isawaitable(returned):
                            await returned
                    if event['event'] == 'on_chain_end' and event['run_id'] == root_id:
                        result = event['data']['output']
            if not result or not final_text(result['messages']):
                raise RuntimeError('Agent returned no final answer.')
            if self._storage_lost:
                raise SessionStorageError('Session connection lost; completion was not confirmed.')
            # Exclusive session lock makes latest unambiguous ONLY after this
            # successful invocation. Failed runs always branch from _complete.
            state = await self.agent.aget_state({'configurable': {'thread_id': self._thread}})
            if state.next:
                raise RuntimeError('Agent has pending execution.')
            if self.repository is not None:
                await self.repository.complete(state.config)
            self._complete = state.config
            self._failed = self._reopened = False
            return AgentTurn(result['messages'], observer.sources, observer.usage)
        except BaseException as exc:
            self._failed = True
            if self.repository is not None:
                with suppress(SessionStorageError):
                    await self.repository.interrupted()
            if self.repository is not None and (self.repository.broken or
                    self.repository.saver.conn.closed or self.repository.saver.conn.broken):
                self._storage_lost = True
            if self._storage_lost:
                raise SessionStorageError('Session connection lost; current turn did not complete.') from None
            raise
        finally:
            self._running = False
            self._active_task = None


def validate_context_config(config, selected):
    if config.context is None:
        return
    summary = config.context.summary_model or selected
    if any(config.models[name].max_input_tokens is None for name in {selected, summary}):
        raise ConfigError('Chat and summary profiles require explicit max_input_tokens.')


@asynccontextmanager
async def assistant_session(config, model_name=None, *, retriever=None, session_id=None):
    selected = model_name or config.default_model
    if selected not in config.models:
        raise ConfigError('Unknown model profile.')
    if session_id is not None and config.sessions is None:
        raise ConfigError('Opening a session requires [sessions].')
    validate_context_config(config, selected)
    if retriever is None and config.knowledge is not None:
        try:
            from ai_assistant.knowledge.config import resolve_knowledge_paths
            from ai_assistant.knowledge.retrieval import create_knowledge_retriever
            base = config.config_path.parent if config.config_path else Path.cwd()
            retriever = create_knowledge_retriever(resolve_knowledge_paths(config.knowledge, base))
        except ImportError:
            raise ConfigError('Knowledge integration is unavailable; install the rag extras.') from None
    async with AsyncExitStack() as stack:
        repository = await stack.enter_async_context(session_storage(config.sessions)) if config.sessions else None
        saver = repository.saver if repository else InMemorySaver()
        context = await stack.enter_async_context(tool_context(config, retriever=retriever))
        model = await stack.enter_async_context(model_session(config, selected,
            config.models[selected].parameters.get('timeout', 300), factory=create_model))
        summary_name = (config.context.summary_model or selected) if config.context else selected
        summary_model = model
        if summary_name != selected:
            summary_model = await stack.enter_async_context(model_session(config, summary_name,
                config.models[summary_name].parameters.get('timeout', 300), factory=create_model))
        tools = create_builtin_tools(include_web=config.tools.web is not None,
            include_write=config.tools.files.enabled,
            include_terminal=config.tools.terminal is not None, retriever=retriever)
        agent = create_assistant_agent(model, tools, checkpointer=saver, settings=config.agent,
            context_settings=config.context, summary_model=summary_model,
            max_input_tokens=config.models[selected].max_input_tokens,
            summary_max_input_tokens=config.models[summary_name].max_input_tokens,
            summary_profile=summary_name)
        session = AssistantSession(agent, context, config.agent, model_name=selected,
                                   tools=tools, repository=repository)
        await session.open(session_id)
        watcher = asyncio.create_task(session.watch_connection()) if repository else None
        try:
            yield session
        finally:
            if session._active_task is not None:
                task = session._active_task
                if not task.cancelling():
                    task.cancel()
                await _drain_worker(task)
            session._closed = True
            if watcher:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
            if repository is None:
                saver.storage.clear()
                saver.writes.clear()
                saver.blobs.clear()
