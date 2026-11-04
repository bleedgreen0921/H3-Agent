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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent, load_persona, load_project_rules
from ai_assistant.config import ConfigError
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.requests import model_session
from ai_assistant.tools.builtin import create_builtin_tools
from ai_assistant.tools.context import tool_context
from ai_assistant.cancellation import _drain_worker
from ai_assistant.storage.sessions import SessionError, SessionStorageError, session_storage
from ai_assistant.runtime.observation import TurnObserver, TurnUsage

from ai_assistant.observability import observed, span, event as observation_event, current_service, ObservationService

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
    result: dict = field(default_factory=dict)


def request_preview(text):
    """只保留用户原文的单行预览，附件优先显示文件名。"""
    import json
    if text.startswith('File reference: '):
        try:
            text = json.loads(text.removeprefix('File reference: ')).get('file_name') or '文件请求'
        except (ValueError, TypeError):
            text = '文件请求'
    return ' '.join(text.split())[:40]


def final_text(messages):
    if not messages or not isinstance(messages[-1], AIMessage) or messages[-1].tool_calls:
        return ''
    return messages[-1].text.strip()


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
        row = await self.repository.open(self.context.workspace.root, session_id, binding=binding)
        self._session_id = str(row['id'])
        self._thread, self._complete = row['thread_id'], row['complete']
        self._threads = row.get('thread_ids') or [self._thread]
        self._failed = row['status'] in {'running', 'interrupted'}
        if self._complete is not None:
            state = await self.agent.aget_state(self._complete)
            from ai_assistant.runtime.history import saved_turn
            if not state.values or state.next or saved_turn(state.values, threads=self._threads) is None:
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

    async def _run_turn(self, text, *, skill_ids=None, on_event=None,
                       input_id=None, user_source=None):
        if self._closed:
            raise RuntimeError('Session is closed.')
        if self._running:
            raise RuntimeError('A turn is already running.')
        if self._storage_lost:
            raise SessionStorageError('Session connection lost; reopen the conversation.')
        input_id = input_id or str(uuid4())
        from ai_assistant.runtime.inputs import metadata, user_message, recorded_messages
        by_id = {}
        for thread in self._threads:
            for message in await recorded_messages(self.agent, thread, input_id=input_id):
                by_id.setdefault((thread, message.id), message)
        records = sorted(by_id.values(), key=lambda m: metadata(m).get('received_at', ''))
        business_text = text
        if user_source is None:
            user_source = text
        for message in records:
            if metadata(message).get('input_id') == input_id:
                if message.text != business_text:
                    raise SessionError('同一输入 ID 的内容不同，拒绝覆盖已接收输入。')
                raise SessionError('该输入已接收，未重复执行；请查看会话状态。')
        project_rules = load_project_rules(self.context)
        turn_context = replace(self.context, project_rules=project_rules, input_id=input_id,
                               resolved_datetimes=set(), effect_facts=[], inflight_tools=set(), executed_tool_calls=set())
        started = monotonic()
        timings = {}
        observer = None
        phase = 'preparation'
        saved_result = None
        self._maintenance = None
        event_enabled = True

        self._running = True
        self._active_task = asyncio.current_task()
        try:
            from ai_assistant.runtime.history import latest_state, recovery_values
            snapshot = await latest_state(self.agent, self._thread)
            recovered = None
            if self._failed:
                # 只复制已保存的对话事实；新线程没有旧任务或 pending writes。
                recovered = recovery_values(snapshot.values)
                branch = str(uuid4())
                await self.agent.aupdate_state({'configurable': {'thread_id': branch}},
                                               recovered, as_node='__start__')
                self._thread = branch
                self._threads.append(branch)
                observation_event('request.branch', execution_thread_id=branch)
            if self.repository is not None:
                await self.repository.start(self._thread)
            timings.setdefault('first_status_seconds', monotonic() - started)
            observer = TurnObserver(self.settings.summary_call_limit, self.model_name,
                self.context.usage_store, self.session_id, input_id=input_id, execution_thread_id=self._thread)
            config = {'configurable': {'thread_id': self._thread},
                      'recursion_limit': 9_999, 'callbacks': [observer],
                      'metadata': {'assistant_model_profile': self.model_name, 'input_id': input_id,
                          'session_id': self.session_id, 'execution_thread_id': self._thread}}
            inputs = []
            graph_input = recovered or {}
            if recovered is not None:
                inputs.extend(graph_input.pop('messages', []))
                inputs.append(SystemMessage(content=RECOVERY_NOTICE))
            inputs.append(user_message(self.session_id, input_id, business_text, source=user_source))
            activated_skills = set()
            graph_input.update(messages=inputs, turn_result=None,
                               execution=None, diagnostics=[], project_rules_snapshot=project_rules)
            if skill_ids:
                from ai_assistant.skills.catalog import SkillError
                if self.context.skills is None:
                    raise SkillError('Skills are not configured.')
                chosen = [self.context.skills.prepare(identity) for identity in skill_ids]
                graph_input['active_skills'] = {s.id: s.digest for s in chosen}
                activated_skills.update(s.id for s in chosen)
            result = None
            root_id = None
            phase = 'execution'
            agent_started = monotonic()
            with span('agent.execute', execution_thread_id=self._thread):
                async with asyncio.timeout(self.settings.turn_timeout_seconds):
                    async for event in self.agent.astream_events(
                        graph_input, config=config, context=turn_context,
                        version='v2', durability='sync'
                    ):
                        if root_id is None:
                            root_id = event['run_id']
                        # 流式回调已删除；仅记录首个正文块的耗时指标。
                        if event['event'] == 'on_chat_model_stream':
                            metadata = event.get('metadata', {})
                            chunk = event.get('data', {}).get('chunk')
                            if (metadata.get('langgraph_node') == 'model' and metadata.get('lc_source') != 'summarization'
                                    and isinstance(chunk, AIMessage) and not timings.get('first_body_seconds')
                                    and any(block.get('text') for block in chunk.content_blocks if block.get('type') == 'text')):
                                timings['first_body_seconds'] = monotonic() - started
                                observation_event('model.first_body', model_first_body_seconds=monotonic() - agent_started)
                        if event['event'] == 'on_tool_end'  and event.get('name') == 'load_skill' and self.context.skills:
                            from langgraph.types import Command
                            output = event.get('data', {}).get('output')
                            if isinstance(output, Command) and isinstance(output.update, dict):
                                activated_skills.update(s.id for s in self.context.skills.active(output.update.get('active_skills', {})))
                        if event['event'] in {'on_tool_start', 'on_tool_end', 'on_tool_error'} and on_event and event_enabled:
                            try:
                                returned = on_event(event)
                                if inspect.isawaitable(returned):
                                    await returned
                            except Exception as callback_error:
                                event_enabled = False
                                logger.warning('Turn auxiliary failure: input_id=%s phase=on_event category=%s', input_id, type(callback_error).__name__)
                        if event['event'] == 'on_chain_end' and event['run_id'] == root_id:
                            result = event['data']['output']
            observation_event('agent.finished', execution_thread_id=self._thread, agent_seconds=monotonic() - agent_started)
            if not result or not final_text(result['messages']):
                raise RuntimeError('Agent returned no final answer.')
            if self._storage_lost:
                raise SessionStorageError('Session connection lost; completion was not confirmed.')
            # 会话锁保护最新状态；完成指针只表示已确认提交的位置。
            state = await self.agent.aget_state({'configurable': {'thread_id': self._thread}})
            if state.next:
                raise RuntimeError('Agent has pending execution.')
            operations = await (self.context.operations or self.context.terminal_operations).list(input_id)
            from ai_assistant.runtime.outcomes import turn_result
            execution_state = {**state.values, 'execution': state.values.get('execution') or
                               {'status': 'complete', 'phase': 'execution'}}
            unified = turn_result(execution_state, operations, executed_tool_calls=turn_context.executed_tool_calls).to_dict() | {'input_id': input_id}
            body_message = state.values['messages'][-1]
            unified['body_locator'] = {'thread_id': self._thread, 'checkpoint_ns': '', 'message_id': body_message.id}
            update = {'turn_result': unified}
            if not state.values.get('conversation_title'):
                first = next((m for m in state.values['messages'] if isinstance(m, HumanMessage) and m.content != RECOVERY_NOTICE and not m.additional_kwargs.get('lc_source')), None)
                if first:
                    update['conversation_title'] = request_preview(first.additional_kwargs.get('user_source', first.text))
            if state.values.get('last_counted_input_id') != input_id:
                completed = state.values.get('completed_turns', 0) + 1
                update.update(completed_turns=completed, last_counted_input_id=input_id)
            commit_started = monotonic()
            phase = 'result_commit'
            with span('result.commit'):
                state_config = await self.agent.aupdate_state(state.config, update)
            saved_result = unified
            phase = 'session_index'
            if self.repository is not None:
                with span('session.index'):
                    await self.repository.complete(state_config, input_id=input_id,
                        title=update.get('conversation_title') or state.values.get('conversation_title') or request_preview(user_source))
            state = await latest_state(self.agent, self._thread)
            observation_event('result.committed', status=unified['status'], result_commit_seconds=monotonic() - commit_started)
            self._complete = state.config
            self._failed = False
            self._maintenance = (input_id, turn_context, config, observer)
            return AgentTurn(result['messages'], observer.sources, observer.usage,
                sorted(activated_skills | {s.id for s in self.context.skills.active(result.get('active_skills', {}))}) if self.context.skills else [],
                input_id, operations, unified)
        except BaseException as error:
            if isinstance(error, SessionStorageError):
                self._storage_lost = True
            from ai_assistant.runtime.execution import drain_tools, ToolCleanupError
            try:
                await drain_tools(turn_context)
            except BaseException as cleanup_error:
                self._storage_lost = True
                logger.error('Turn cleanup failure: input_id=%s phase=tools category=%s', input_id, type(cleanup_error).__name__)
            observation_event('turn.stopped', phase=phase, category=type(error).__name__, status='failed')
            logger.error('Turn stopped: input_id=%s phase=%s category=%s', input_id, phase, type(error).__name__)
            self._failed = True
            self._maintenance = None
            if self.repository is not None and not self._storage_lost:
                try:
                    await self.repository.interrupted()
                except Exception as marking_error:
                    self._storage_lost = True
                    logger.error('Turn cleanup failure: input_id=%s phase=interrupted category=%s', input_id, type(marking_error).__name__)
            from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
            if isinstance(error, ModelCallLimitExceededError):
                error = TurnIncompleteError('本轮主模型调用次数已达上限；已发生效果保留，请核实操作回执。')
            if isinstance(error, ToolCleanupError):
                self._storage_lost = True
            # 已保存回答由入口只读恢复，索引或交付错误不能改写它。
            if saved_result is not None:
                error.turn_result = saved_result
            if not isinstance(error, asyncio.CancelledError):
                error.turn_message = ('会话存储或清理未确认，请重新打开后核实。' if self._storage_lost else
                    '本轮超时。' if isinstance(error, TimeoutError) else
                    '本轮未完成（' + type(error).__name__ + '）。') + '已发生的操作不会自动回滚，未自动重试。'
            if self._storage_lost and isinstance(error, asyncio.CancelledError):
                raise SessionStorageError('Session connection lost; cleanup finished, reopen the conversation.') from None
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

    @observed("session.maintenance", lambda self: {"session_id": self.session_id,
        "execution_thread_id": self._thread, "input_id": self._maintenance[0] if self._maintenance else None}, root=True)
    async def after_reply(self):
        """入口成功处理最终回复后，在同一任务、会话锁内整理；不改写回答。"""
        maintenance_started = monotonic()
        pending = getattr(self, '_maintenance', None)
        self._maintenance = None
        if pending is None or self._storage_lost or self._closed:
            observation_event('maintenance.skipped', status='skipped')
            return
        async with self._turn_lock:
            input_id, context, config, observer = pending
            self._active_task = asyncio.current_task()
            try:
                if self.repository is not None:
                    await self.repository.check()
                from ai_assistant.runtime.history import latest_state
                state = await latest_state(self.agent, self._thread)
                every = getattr(self.agent, 'summary_every_turns', 0)
                summarizer = getattr(self.agent, 'context_summary', None)
                completed = state.values.get('completed_turns', 0)
                if (state.values.get('turn_result', {}).get('input_id') != input_id
                        or state.values['turn_result']['status'] != 'complete'
                        or not every or not summarizer
                        or completed - state.values.get('summarized_turns', 0) < every
                        or observer.usage.summary_model_calls >= self.settings.summary_call_limit):
                    observation_event('maintenance.skipped', status='skipped')
                    return
                update = await summarizer.periodic_update(state.values, context, config)
                if update:
                    await self.agent.aupdate_state(state.config, {**update, 'summarized_turns': completed})
                observation_event('maintenance.finished', status='complete', maintenance_seconds=monotonic() - maintenance_started)
            except Exception as error:
                if isinstance(error, SessionStorageError):
                    self._storage_lost = True
                observation_event('maintenance.failed', status='failed', category=type(error).__name__, maintenance_seconds=monotonic() - maintenance_started)
                logger.warning('Turn auxiliary failure: input_id=%s phase=periodic_summary category=%s', input_id, type(error).__name__)
            finally:
                self._active_task = None


def validate_context_config(config, selected):
    if config.context is None:
        return
    summary = config.context.summary_model or selected
    if any(config.models[name].max_input_tokens is None for name in {selected, summary}):
        raise ConfigError('Chat and summary profiles require explicit max_input_tokens.')


@asynccontextmanager
async def assistant_session(config, model_name=None, *, retriever=None, session_id=None, caller=None, resources=None):
    from ai_assistant.integrations.policy import Caller, shared_service_allowed
    caller = caller or Caller()
    persona = load_persona(config) if caller.personal else ''
    source_config = config
    if not caller.personal:
        # 共享和未验证私人入口不能继承个人能力。
        config = config.model_copy(update={'knowledge': None, 'skills': None,
            'tools': config.tools.model_copy(update={'web': None, 'files': config.tools.files.model_copy(update={'enabled': False}), 'terminal': None})})
        retriever = None
    selected = model_name or config.default_model
    if selected not in config.models:
        raise ConfigError('Unknown model profile.')
    if session_id is not None and config.sessions is None:
        raise ConfigError('Opening a session requires [sessions].')
    validate_context_config(config, selected)
    if config.mcp and any(s.tools and caller.owner and (caller.scope != 'shared' or (
            name in caller.resources and shared_service_allowed(s))) for name, s in config.mcp.servers.items()):
        if config.models[selected].max_input_tokens is None:
            raise ConfigError('启用 MCP 工具的模型必须显式配置 max_input_tokens。')
    async with AsyncExitStack() as stack:
        if current_service() is None:
            from ai_assistant.observability import configuration_secrets
            await stack.enter_async_context(ObservationService(config.observability, configuration_secrets(config)))
        repository = await stack.enter_async_context(session_storage(config.sessions)) if config.sessions else None
        usage_store = None
        if repository:
            from ai_assistant.storage.usage import usage_storage
            try:
                usage_store = await stack.enter_async_context(usage_storage(config.sessions))
            except Exception as error:
                logger.warning('Usage unavailable: phase=connect category=%s', type(error).__name__)
        saver = repository.saver if repository else InMemorySaver()
        context = await stack.enter_async_context(tool_context(config, retriever=retriever,
            skill_snapshot=getattr(resources, "skills", None) if caller.personal else None))
        if not caller.personal:
            from ai_assistant.skills.catalog import SkillCatalog, configured_roots, skills_for_caller
            source = getattr(resources, 'skills', None)
            if caller.skill_ids and source is None and source_config.skills is not None:
                source = SkillCatalog(configured_roots(source_config), disabled=source_config.skills.disabled, config_files=source_config.config_files)
                stack.callback(source.close)
            context = replace(context, skills=skills_for_caller(source_config, source, caller))
        context = replace(context, usage_store=usage_store, timezone=config.agent.timezone if caller.personal else 'UTC',
                          owner_open_id=config.integrations.feishu.owner_open_id if caller.personal and config.integrations.feishu else None,
                          default_task_assignee=config.integrations.feishu.default_task_assignee if caller.personal and config.integrations.feishu else 'none',
                          caller=caller,
                          skills_directory_tokens=config.skills.directory_tokens if config.skills else 6000)
        if config.knowledge is not None:
            context = replace(context, knowledge_config=config)
        from ai_assistant.tools.mail import MailService, allowed as mail_allowed, email_tools
        mail_enabled = config.mail.enabled and mail_allowed(caller)
        from ai_assistant.storage.integrations import operation_storage
        operations = (await stack.enter_async_context(operation_storage(config.sessions))
                      if config.sessions and (mail_enabled or config.tools.terminal is not None or
                          (config.mcp is not None and config.mcp.servers)) else None)
        context = replace(context, operations=operations)
        if config.mcp is not None and config.mcp.servers:
            from ai_assistant.integrations.mcp import mcp_resources
            directory = getattr(resources, "catalog", resources) if resources is not None else await stack.enter_async_context(
                mcp_resources(config, context.workspace, caller=caller))
            context = replace(context, mcp=directory.for_caller(caller), caller=caller, operations=operations)
        if mail_enabled:
            if operations is None:
                raise ConfigError('启用邮箱需要持久回执 [sessions]，请配置并显式 init。')
            try:
                await operations.by_source('mail:schema-check')
            except Exception:
                raise ConfigError('邮箱来源回执结构不可用，请检查存储并显式 init 后重启。') from None
            mail = MailService(config.mail, config.agent.timezone)
            stack.push_async_callback(mail.aclose)
            context = replace(context, mail=mail)
        pdf_enabled = (config.tools.pdf.enabled and caller.owner and caller.scope == 'private'
                       and caller.channel == 'feishu_private')
        if pdf_enabled:
            from ai_assistant.tools.pdf import PDFAttachments
            from ai_assistant.integrations.feishu_files import FeishuAPI
            import httpx
            client = await stack.enter_async_context(httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=60))
            pdf = PDFAttachments(FeishuAPI(config.integrations.feishu, client), config.tools.pdf.timeout)
            stack.callback(pdf.close)
            context = replace(context, pdf=pdf)
        model = await stack.enter_async_context(model_session(config, selected,
            config.models[selected].parameters.get('timeout', 180), factory=create_model))
        summary_name = (config.context.summary_model or selected) if config.context else selected
        summary_model = model
        if summary_name != selected:
            summary_model = await stack.enter_async_context(model_session(config, summary_name,
                config.models[summary_name].parameters.get('timeout', 180), factory=create_model))
        tools = create_builtin_tools(include_web=config.tools.web is not None,
            include_write=config.tools.files.enabled,
            include_terminal=config.tools.terminal is not None)
        if mail_enabled:
            tools.extend(email_tools())
        if pdf_enabled:
            from ai_assistant.tools.pdf import read_pdf_attachment, read_document_attachment
            tools.extend([read_pdf_attachment, read_document_attachment])
        if config.knowledge is not None or retriever is not None:
            from ai_assistant.knowledge.tools import knowledge_tools
            tools.extend(knowledge_tools(include_management=config.knowledge is not None))
        if context.skills is not None:
            from ai_assistant.tools.skills import create_skill_tools
            tools.extend(create_skill_tools())
        if not caller.personal:
            tools = create_skill_tools() if context.skills is not None else []
        from ai_assistant.tools.datetime import resolve_datetime
        if caller.personal or (context.mcp and any(
                e.original_name.startswith('task_v2_task_') for e in context.mcp.entries.values())):
            tools.append(resolve_datetime.model_copy(update={'handle_tool_error': True}))
        if config.integrations.feishu is not None:
            from ai_assistant.integrations.feishu_identity import identity_tools
            tools.extend(identity_tools(caller))
        if context.mcp is not None:
            tools.extend(e.tool for e in context.mcp.entries.values())
        agent = create_assistant_agent(model, tools, checkpointer=saver, settings=config.agent, persona=persona,
            context_settings=config.context, summary_model=summary_model,
            max_input_tokens=config.models[selected].max_input_tokens,
            summary_max_input_tokens=config.models[summary_name].max_input_tokens,
            summary_profile=summary_name, skills_enabled=context.skills is not None, mcp_enabled=bool(context.mcp and (context.mcp.entries or context.mcp.unavailable)))
        session = AssistantSession(agent, context, config.agent, model_name=selected,
                                   repository=repository)
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
