"""Native turns, durable completion pointers and resource ownership."""
import asyncio
import inspect
import logging
from time import monotonic
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent, load_persona
from ai_assistant.config import ConfigError
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.requests import model_session
from ai_assistant.tools.builtin import create_builtin_tools
from ai_assistant.tools.context import tool_context
from ai_assistant.cancellation import _drain_worker
from ai_assistant.storage.sessions import SessionError, SessionStorageError, session_storage
from ai_assistant.runtime.observation import TurnObserver, TurnUsage

logger = logging.getLogger(__name__)

RECOVERY_NOTICE = ('应用状态：上一轮未完成，已发生的文件或命令效果不会回滚。'
                   '不要自动重放上一轮动作；直接处理本轮请求，仅在本轮需要操作相关资源时核实状态。')


class TurnIncompleteError(SessionError):
    """仅包含运行时生成的安全状态说明，可直接交付各入口。"""



@dataclass(frozen=True)
class AgentTurn:
    messages: list[BaseMessage]
    sources: list[Document]
    usage: TurnUsage = field(default_factory=TurnUsage)
    skill_ids: list[str] = field(default_factory=list)
    input_id: str = ""
    operations: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class WaitingTurn:
    sync_id: str
    input_id: str


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
        self._pending = None
        self._pending_input = None

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
        self._pending = row.get('pending')
        self._pending_input = row.get('pending_input')
        self._reopened = session_id is not None
        if self._complete is not None:
            state = await self.agent.aget_state(self._complete)
            if not state.values or state.next or not final_text(state.values.get('messages', [])):
                raise SessionError('Stored successful checkpoint is missing or incompatible.')

    async def new_conversation(self):
        if self._running:
            raise RuntimeError('A turn is already running.')
        if self._pending:
            await self.cancel_waiting()
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

    async def run_turn(self, text, *, skill_ids=None, on_event=None, on_output=None,
                       input_id=None, resume=False, user_source=None):
        if self._closed:
            raise RuntimeError('Session is closed.')
        if self._running:
            raise RuntimeError('A turn is already running.')
        if self._storage_lost:
            raise SessionStorageError('Session connection lost; reopen the conversation.')
        if self._pending and not resume:
            raise SessionError('本会话正在等待知识同步，请先继续或取消等待。')
        if resume and not self._pending:
            raise SessionError('本会话没有可续接的知识请求。')
        input_id = self._pending_input if resume else (input_id or str(uuid4()))
        turn_context = replace(self.context, input_id=input_id, resolved_datetimes=set(), task_outcomes={})
        started = monotonic()
        timings = {}
        observer = None

        async def emit_output(kind, **data):
            if kind == 'stage':
                timings.setdefault('first_status_seconds', monotonic() - started)
            if kind == 'delta' and data.get('text'):
                timings.setdefault('first_body_seconds', monotonic() - started)
            if on_output:
                returned = on_output({'kind': kind, **data})
                if inspect.isawaitable(returned):
                    await returned

        self._running = True
        self._active_task = asyncio.current_task()
        try:
            if self._complete is None and self._failed:
                self._thread = str(uuid4())
            if self.repository is not None:
                if resume:
                    # 续接一旦开始便不再属于可安全自动恢复的等待状态。
                    await self.repository.clear_waiting()
                await self.repository.start(self._thread)
            await emit_output('stage', text='正在处理')
            if turn_context.memory is not None:
                await turn_context.memory.recall(text)
            observer = TurnObserver(self.settings.model_call_limit, self.model_name,
                self.context.usage_store, self.session_id)
            locator = self._pending['checkpoint'] if resume else (self._complete or {'configurable': {'thread_id': self._thread}})
            # astream_events may insert LangGraph's generic 25-step default,
            # overriding create_agent's 9_999-step setting. Middleware nodes
            # consume steps too; model/tool/deadline limits remain authoritative.
            config = {**locator, 'recursion_limit': 9_999, 'callbacks': [observer],
                      'metadata': {'assistant_model_profile': self.model_name}}
            inputs = []
            if self._failed:
                inputs.append(HumanMessage(content=RECOVERY_NOTICE))
            # 飞书每条消息均重新打开持久会话，重开不是用户的新任务。
            from datetime import datetime, timezone
            inputs.append(HumanMessage(content=text, id=str(uuid4()), additional_kwargs={
                'user_source': text if user_source is None else user_source,
                'user_timestamp': datetime.now(timezone.utc).isoformat()}))
            activated_skills = set()
            graph_input = {'messages': inputs, 'turn_outcome': None}
            if not resume:
                snapshot = await self.agent.aget_state(locator)
                graph_input['active_mcp_tools'] = {key: None for key in snapshot.values.get('active_mcp_tools', {})}
            if resume:
                from langgraph.types import Command
                if self.context.operations:
                    operations = await self.context.operations.list(input_id)
                    if any(o['status'] in ('pending','unknown') for o in operations):
                        raise SessionError('存在未确认的外部写入，请先核实操作回执，不自动续接。')
                graph_input = Command(resume=True)
            if skill_ids and not resume:
                from ai_assistant.skills.catalog import SkillError
                if self.context.skills is None:
                    raise SkillError('Skills are not configured.')
                chosen = [self.context.skills.resolve(identity) for identity in skill_ids]
                graph_input['active_skills'] = {s.id: s.digest for s in chosen}
                activated_skills.update(s.id for s in chosen)
            result = None
            root_id = None
            async with asyncio.timeout(self.settings.turn_timeout_seconds):
                async for event in self.agent.astream_events(
                    graph_input, config=config, context=turn_context,
                    version='v2', durability='sync'
                ):
                    if root_id is None:
                        root_id = event['run_id']
                    metadata = event.get('metadata', {})
                    if metadata.get('langgraph_node') == 'model' and metadata.get('lc_source') != 'summarization':
                        if event['event'] == 'on_chat_model_start':
                            await emit_output('replace', text='')
                        elif event['event'] == 'on_chat_model_stream':
                            chunk = event.get('data', {}).get('chunk')
                            if isinstance(chunk, AIMessage) and chunk.text:
                                await emit_output('delta', text=chunk.text)
                        elif event['event'] == 'on_chat_model_end':
                            answer = event.get('data', {}).get('output')
                            if isinstance(answer, AIMessage) and answer.tool_calls:
                                await emit_output('stage', text='正在执行工具')
                    if event['event'] == 'on_tool_start':
                        await emit_output('stage', text='正在执行工具')
                    if event['event'] == 'on_tool_end'  and event.get('name') == 'load_skill' and self.context.skills:
                        from langgraph.types import Command
                        output = event.get('data', {}).get('output')
                        if isinstance(output, Command) and isinstance(output.update, dict):
                            activated_skills.update(s.id for s in self.context.skills.active(output.update.get('active_skills', {})))
                    if event['event'] in {'on_tool_start', 'on_tool_end', 'on_tool_error'} and on_event:
                        returned = on_event(event)
                        if inspect.isawaitable(returned):
                            await returned
                    if event['event'] == 'on_chain_end' and event['run_id'] == root_id:
                        result = event['data']['output']
            state = await self.agent.aget_state({'configurable': {'thread_id': self._thread}})
            interrupts = [item for task in state.tasks for item in task.interrupts]
            if interrupts:
                value = interrupts[0].value
                if not isinstance(value, dict) or value.get('kind') != 'knowledge_sync':
                    raise RuntimeError('Unsupported graph interruption.')
                if self.repository is None:
                    raise SessionError('后台知识检索需要持久会话。')
                self._pending = {'sync_id': value['sync_id'], 'checkpoint': state.config}
                self._pending_input = input_id
                await self.repository.waiting(self._pending, input_id)
                await emit_output('stage', text='正在同步知识来源，完成后继续回答')
                return WaitingTurn(value['sync_id'], input_id)
            if not result or not final_text(result['messages']):
                raise RuntimeError('Agent returned no final answer.')
            if self._storage_lost:
                raise SessionStorageError('Session connection lost; completion was not confirmed.')
            # Exclusive session lock makes latest unambiguous ONLY after this
            # successful invocation. Failed runs always branch from _complete.
            state = await self.agent.aget_state({'configurable': {'thread_id': self._thread}})
            if state.next:
                raise RuntimeError('Agent has pending execution.')
            operations = await self.context.operations.list(input_id) if self.context.operations else []
            if self.repository is not None:
                # 成功指针与待提交记忆任务共用事务；失败分支不会登记候选。
                async with self.repository.conn.transaction() if turn_context.memory is not None else AsyncExitStack():
                    if turn_context.memory is not None:
                        await turn_context.memory.register(state.values.get('pending_memories', []), self.repository.conn)
                    if self.context.mcp is not None or self.context.caller is not None:
                        await self.repository.complete(state.config, input_id=input_id)
                    else:
                        await self.repository.complete(state.config)
            self._complete = state.config
            self._pending = self._pending_input = None
            self._failed = self._reopened = False
            with suppress(Exception):
                await emit_output('failed' if result.get('turn_outcome') else 'complete', text=final_text(result['messages']))
            return AgentTurn(result['messages'], observer.sources, observer.usage,
                sorted(activated_skills | {s.id for s in self.context.skills.active(result.get('active_skills', {}))}) if self.context.skills else [],
                input_id, operations)
        except BaseException as error:
            from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
            if isinstance(error, ModelCallLimitExceededError):
                try:
                    operations = await turn_context.operations.list(input_id) if turn_context.operations else []
                except Exception:
                    operations = None
                lines = ['未完成：本轮主模型调用次数已达上限。']
                if operations is None:
                    lines.append('操作记录读取失败，远端写入状态未知；请核实，未自动重放。')
                elif not operations:
                    lines.append('本轮未记录远端写入。')
                for operation in operations or []:
                    stored = operation.get('result') or {}
                    detail = stored.get('business_outcome')
                    status = operation['status']
                    lines.append(f"操作 {operation['id']}：{status}；" + (
                        '业务核对未通过' if detail else '远端回执成功，本轮整体仍未完成' if status == 'succeeded'
                        else '需要核实，未自动重放'))
                if any(o['status'] == 'succeeded' and not (o.get('result') or {}).get('business_outcome')
                       for o in operations or []):
                    lines.insert(1, '部分完成：已有成功回执，其他步骤未完成或未确认。')
                error = TurnIncompleteError('\n'.join(lines))
            with suppress(Exception):
                await emit_output('cancelled' if isinstance(error, asyncio.CancelledError) else 'failed', text=str(error) if isinstance(error, TurnIncompleteError) else '本轮未完成')
            if resume:
                self._pending = self._pending_input = None
            self._failed = True
            if self.repository is not None:
                with suppress(SessionStorageError):
                    await self.repository.interrupted()
            if self.repository is not None and (self.repository.broken or
                    self.repository.saver.conn.closed or self.repository.saver.conn.broken):
                self._storage_lost = True
            if self._storage_lost:
                raise SessionStorageError('Session connection lost; current turn did not complete.') from None
            raise error
        finally:
            if observer is not None:
                observer.usage.total_seconds = monotonic() - started
                observer.usage.first_status_seconds = timings.get('first_status_seconds')
                observer.usage.first_body_seconds = timings.get('first_body_seconds')
                from dataclasses import asdict
                logger.info('Turn metrics: input_id=%s metrics=%s', input_id, asdict(observer.usage))
            self._running = False
            self._active_task = None

    async def cancel_waiting(self):
        if self.repository:
            await self.repository.clear_waiting()
        self._pending = self._pending_input = None

    async def continue_waiting(self, *, on_event=None, on_output=None):
        if not self._pending:
            raise SessionError('没有待续接请求。')
        from ai_assistant.knowledge.sync import wait_sync
        await wait_sync(self.context.knowledge_config, self._pending['sync_id'])
        return await self.run_turn('', resume=True, on_event=on_event, on_output=on_output)


def validate_context_config(config, selected):
    if config.context is None:
        return
    summary = config.context.summary_model or selected
    if any(config.models[name].max_input_tokens is None for name in {selected, summary}):
        raise ConfigError('Chat and summary profiles require explicit max_input_tokens.')


@asynccontextmanager
async def assistant_session(config, model_name=None, *, retriever=None, session_id=None, caller=None, resources=None):
    persona = load_persona(config)
    from ai_assistant.integrations.policy import Caller
    caller = caller or Caller()
    if caller.channel == 'feishu_group':
        # Group sessions cannot inherit private local capabilities.
        config = config.model_copy(update={'knowledge': None, 'skills': None,
            'tools': config.tools.model_copy(update={'web': None, 'files': config.tools.files.model_copy(update={'enabled': False}), 'terminal': None})})
        retriever = None
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
        usage_store = None
        if repository:
            from ai_assistant.storage.usage import usage_storage
            usage_store = await stack.enter_async_context(usage_storage(config.sessions))
        saver = repository.saver if repository else InMemorySaver()
        context = await stack.enter_async_context(tool_context(config, retriever=retriever,
            skill_snapshot=getattr(resources, "skills", None) if caller.channel != "feishu_group" else None))
        context = replace(context, usage_store=usage_store, timezone=config.agent.timezone,
                          owner_open_id=config.integrations.feishu.owner_open_id if config.integrations.feishu else None,
                          default_task_assignee=config.integrations.feishu.default_task_assignee if config.integrations.feishu else 'none',
                          caller=caller if caller.channel != "cli" else None,
                          mcp_schema_tokens=config.context.mcp_schema_tokens if config.context else 6000)
        if config.memory.enabled and caller.channel == 'feishu_private' and caller.owner:
            from ai_assistant.storage.memory import memory_storage
            memory = await stack.enter_async_context(memory_storage(config))
            context = replace(context, memory=memory)
        if config.knowledge is not None:
            context = replace(context, knowledge_config=config)
        if config.mcp is not None and config.mcp.servers:
            from ai_assistant.integrations.mcp import mcp_resources
            from ai_assistant.storage.integrations import operation_storage
            directory = getattr(resources, "catalog", resources) if resources is not None else await stack.enter_async_context(
                mcp_resources(config, context.workspace, caller=caller))
            operations = await stack.enter_async_context(operation_storage(config.sessions)) if config.sessions else None
            context = replace(context, mcp=directory.for_caller(caller), caller=caller, operations=operations)
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
        if config.knowledge is not None:
            from ai_assistant.knowledge.tools import knowledge_tools
            tools = [t for t in tools if t.name != 'search_knowledge'] + knowledge_tools()
        if config.context is not None:
            from ai_assistant.agents.results import read_tool_result
            tools.append(read_tool_result)
        if context.memory is not None:
            from ai_assistant.tools.memory import remember_memory, manage_memory
            tools.extend([remember_memory, manage_memory])
        if context.skills is not None:
            from ai_assistant.tools.skills import create_skill_tools
            tools.extend(create_skill_tools())
        if caller.channel == 'feishu_group':
            tools = []
        from ai_assistant.tools.datetime import resolve_datetime, get_current_identity
        if caller.channel != 'feishu_group' or (context.mcp and any(
                e.original_name.startswith('task_v2_task_') for e in context.mcp.entries.values())):
            tools.append(resolve_datetime.model_copy(update={'handle_tool_error': True}))
        if caller.channel == 'feishu_private' and caller.owner:
            tools.append(get_current_identity.model_copy(update={'handle_tool_error': True}))
        if context.mcp is not None:
            from ai_assistant.tools.mcp import discovery_tools
            tools.extend(discovery_tools())
            tools.extend(e.tool for e in context.mcp.entries.values())
        agent = create_assistant_agent(model, tools, checkpointer=saver, settings=config.agent, persona=persona,
            context_settings=config.context, summary_model=summary_model,
            max_input_tokens=config.models[selected].max_input_tokens,
            summary_max_input_tokens=config.models[summary_name].max_input_tokens,
            summary_profile=summary_name, skills_enabled=context.skills is not None, mcp_enabled=context.mcp is not None)
        context = replace(context, checkpoint_history=agent.aget_state_history)
        session = AssistantSession(agent, context, config.agent, model_name=selected,
                                   tools=tools, repository=repository)
        await session.open(session_id)
        if usage_store:
            await usage_store.abandon(session.session_id)
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
