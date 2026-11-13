"""单轮请求去重、失败恢复、Agent 执行、结果提交与异常清理。"""
import asyncio
import inspect
import logging
from dataclasses import replace
from time import monotonic
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_assistant.agents import load_project_rules
from ai_assistant.storage.sessions import SessionError, SessionStorageError
from ai_assistant.runtime.observation import TurnObserver
from ai_assistant.runtime.types import AgentTurn, RECOVERY_NOTICE, TurnIncompleteError, final_text, request_preview
from ai_assistant.observability import span, event as observation_event, csv_ids

logger = logging.getLogger(__name__)


async def execute_turn(session, text, *, skill_ids=None, on_event=None,
                   input_id=None, user_source=None, planning_action=None,
                   planning_goal=None):
    if session._closed:
        raise RuntimeError('Session is closed.')
    if session._running:
        raise RuntimeError('A turn is already running.')
    if session._storage_lost:
        raise SessionStorageError('Session connection lost; reopen the conversation.')
    input_id = input_id or str(uuid4())
    from ai_assistant.runtime.inputs import metadata, user_message, recorded_messages
    by_id = {}
    for thread in session._threads:
        for message in await recorded_messages(session.agent, thread, input_id=input_id):
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
    project_rules = load_project_rules(session.context)
    from ai_assistant.capabilities.model import CapabilityTurnState
    previous = getattr(session.context, 'capabilities', None)
    registry = getattr(previous, 'registry', None)
    turn_context = replace(session.context, project_rules=project_rules, input_id=input_id, session_id=session.session_id,
                           execution_thread_id=session._thread,
                           resolved_datetimes=set(), effect_facts=[], resource_bindings={},
                           inflight_tools=set(), executed_tool_calls=set(),
                           capabilities=CapabilityTurnState(registry=registry))
    started = monotonic()
    timings = {}
    observer = None
    phase = 'preparation'
    saved_result = None
    session._maintenance = None
    event_enabled = True

    async def emit_aux(item):
        """转发可选的进度事件；回调故障只停用本轮辅助能力，不影响业务结果。"""
        nonlocal event_enabled
        if on_event is None or not event_enabled:
            return
        try:
            returned = on_event(item)
            if inspect.isawaitable(returned):
                await returned
        except Exception as callback_error:
            event_enabled = False
            logger.warning('Turn auxiliary failure: input_id=%s phase=on_event category=%s',
                           input_id, type(callback_error).__name__)

    session._running = True
    session._active_task = asyncio.current_task()
    try:
        from ai_assistant.runtime.history import latest_state, recovery_values
        snapshot = await latest_state(session.agent, session._thread)
        recovered = None
        if session._failed:
            # 只复制已保存的对话事实；新线程没有旧任务或 pending writes。
            recovered = recovery_values(snapshot.values)
            branch = str(uuid4())
            await session.agent.aupdate_state({'configurable': {'thread_id': branch}},
                                           recovered, as_node='__start__')
            session._thread = branch
            session._threads.append(branch)
            turn_context = replace(turn_context, execution_thread_id=branch)
            observation_event('request.branch', execution_thread_id=branch)
        if session.repository is not None:
            await session.repository.start(session._thread)
        timings.setdefault('first_status_seconds', monotonic() - started)
        observer = TurnObserver(session.settings.summary_call_limit, session.model_name,
            session.context.usage_store, session.session_id, input_id=input_id, execution_thread_id=session._thread)
        config = {'configurable': {'thread_id': session._thread},
                  'recursion_limit': 9_999, 'callbacks': [observer],
                  'metadata': {'assistant_model_profile': session.model_name, 'input_id': input_id,
                      'session_id': session.session_id, 'execution_thread_id': session._thread}}
        inputs = []
        graph_input = recovered or {}
        if recovered is not None:
            inputs.extend(graph_input.pop('messages', []))
            inputs.append(SystemMessage(content=RECOVERY_NOTICE))
        inputs.append(user_message(session.session_id, input_id, business_text, source=user_source))
        activated_skills = set()
        graph_input.update(messages=inputs, turn_result=None,
                           execution=None, diagnostics=[], project_rules_snapshot=project_rules,
                           input_id=input_id, planning_action=planning_action,
                           planning_goal=planning_goal, planning_note=None,
                           dispatch=[], effect_facts=[])
        if skill_ids:
            from ai_assistant.skills.catalog import SkillError
            if session.context.services.get('skills') is None:
                raise SkillError('Skills are not configured.')
            chosen = [session.context.services.get('skills').prepare(identity) for identity in skill_ids]
            graph_input['active_skills'] = {s.id: s.digest for s in chosen}
            activated_skills.update(s.id for s in chosen)
        result = None
        root_id = None
        phase = 'execution'
        agent_started = monotonic()
        with span('agent.execute', execution_thread_id=session._thread):
            async with asyncio.timeout(session.settings.turn_timeout_seconds):
                async for event in session.agent.astream_events(
                    graph_input, config=config, context=turn_context,
                    version='v2', durability='sync'
                ):
                    if root_id is None:
                        root_id = event['run_id']
                    # 模型原生流式仍用于首字耗时与工作卡预览，不参与答案组装。
                    if event['event'] == 'on_chat_model_stream':
                        metadata = event.get('metadata', {})
                        chunk = event.get('data', {}).get('chunk')
                        if (metadata.get('langgraph_node') == 'model' and metadata.get('lc_source') != 'summarization'
                                and metadata.get('assistant_phase') != 'task'
                                and isinstance(chunk, AIMessage) and not timings.get('first_body_seconds')
                                and any(block.get('text') for block in chunk.content_blocks if block.get('type') == 'text')):
                            timings['first_body_seconds'] = monotonic() - started
                            observation_event('model.first_body', model_first_body_seconds=monotonic() - agent_started)
                        if (metadata.get('langgraph_node') == 'model' and metadata.get('lc_source') != 'summarization'
                                and metadata.get('assistant_phase') != 'task'
                                and isinstance(chunk, AIMessage)):
                            delta = ''.join(block.get('text', '') for block in chunk.content_blocks
                                            if isinstance(block, dict) and block.get('type') == 'text')
                            if delta:
                                await emit_aux({'event': 'on_text', 'text': delta,
                                                'run_id': event.get('run_id')})
                    if event['event'] == 'on_chat_model_start':
                        metadata = event.get('metadata', {})
                        if (metadata.get('langgraph_node') == 'model'
                                and metadata.get('lc_source') != 'summarization'
                                and metadata.get('assistant_phase') != 'task'):
                            await emit_aux({'event': 'on_model_start'})
                    if event['event'] == 'on_tool_end'  and event.get('name') == 'load_skill' and session.context.services.get('skills'):
                        from langgraph.types import Command
                        output = event.get('data', {}).get('output')
                        if isinstance(output, Command) and isinstance(output.update, dict):
                            activated_skills.update(s.id for s in session.context.services.get('skills').active(output.update.get('active_skills', {})))
                    if event['event'] in {'on_tool_start', 'on_tool_end', 'on_tool_error'}:
                        if event['event'] == 'on_tool_start':
                            try:
                                from ai_assistant.agents.failures import tool_label
                                event = {**event, 'label': tool_label(turn_context, event.get('name'))}
                            except Exception:
                                pass
                        await emit_aux(event)
                    if event['event'] == 'on_chain_end' and event['run_id'] == root_id:
                        result = event['data']['output']
        observation_event('agent.finished', execution_thread_id=session._thread, agent_seconds=monotonic() - agent_started)
        if not result or not final_text(result['messages']):
            raise RuntimeError('Agent returned no final answer.')
        if session._storage_lost:
            raise SessionStorageError('Session connection lost; completion was not confirmed.')
        # 会话锁保护最新状态；完成指针只表示已确认提交的位置。
        state = await session.agent.aget_state({'configurable': {'thread_id': session._thread}})
        operations = await session.context.operations.list(input_id) if session.context.operations else []
        from ai_assistant.runtime.outcomes import turn_result, SUCCESS
        execution_state = {**state.values, 'execution': state.values.get('execution') or
                           {'status': 'complete', 'phase': 'execution'}}
        unified = turn_result(execution_state, operations, executed_tool_calls=turn_context.executed_tool_calls).to_dict() | {'input_id': input_id}
        # 只有完整的终态才能作为下一次恢复点；失败轮次允许保留结果，但不能推进 complete。
        if state.next and unified['status'] == 'complete':
            raise RuntimeError('Agent has pending execution.')
        body_message = state.values['messages'][-1]
        unified['body_locator'] = {'thread_id': session._thread, 'checkpoint_ns': '', 'message_id': body_message.id}
        update = {'turn_result': unified}
        if not state.values.get('conversation_title'):
            first = next((m for m in state.values['messages'] if isinstance(m, HumanMessage) and m.content != RECOVERY_NOTICE and not m.additional_kwargs.get('lc_source')), None)
            if first:
                update['conversation_title'] = request_preview(first.additional_kwargs.get('user_source', first.text))
        if state.values.get('last_counted_input_id') != input_id and unified['status'] == 'complete':
            completed = state.values.get('completed_turns', 0) + 1
            update.update(completed_turns=completed, last_counted_input_id=input_id)
        commit_started = monotonic()
        phase = 'result_commit'
        with span('result.commit'):
            state_config = await session.agent.aupdate_state(state.config, update)
        saved_result = unified
        committed_state = await latest_state(session.agent, session._thread)
        resumable = unified['status'] == 'complete' and not committed_state.next
        phase = 'session_index'
        if session.repository is not None and resumable:
            with span('session.index'):
                await session.repository.complete(state_config, input_id=input_id,
                    title=update.get('conversation_title') or state.values.get('conversation_title') or request_preview(user_source))
        elif session.repository is not None:
            await session.repository.interrupted()
        state = committed_state
        effects = unified.get('items') or []
        observation_event('result.committed', status=unified['status'], result_commit_seconds=monotonic() - commit_started,
                          effect_count=len(effects),
                          confirmed_effects=sum(1 for item in effects if item.get('status') in SUCCESS),
                          write_tools=csv_ids(sorted({item.get('tool') for item in effects
                                                      if item.get('kind') == 'local_file' and item.get('tool')})))
        if resumable:
            session._complete = state_config
            session._failed = False
            session._maintenance = (input_id, turn_context, config, observer)
        else:
            session._failed = True
            session._maintenance = None
        return AgentTurn(result['messages'], observer.sources, observer.usage,
            sorted(activated_skills | {s.id for s in session.context.services.get('skills').active(result.get('active_skills', {}))}) if session.context.services.get('skills') else [],
            input_id, operations, unified)
    except BaseException as error:
        if isinstance(error, SessionStorageError):
            session._storage_lost = True
        from ai_assistant.runtime.execution import drain_tools, ToolCleanupError
        try:
            await drain_tools(turn_context)
        except BaseException as cleanup_error:
            session._storage_lost = True
            logger.error('Turn cleanup failure: input_id=%s phase=tools category=%s', input_id, type(cleanup_error).__name__)
        observation_event('turn.stopped', phase=phase, category=type(error).__name__, status='failed')
        logger.error('Turn stopped: input_id=%s phase=%s category=%s', input_id, phase, type(error).__name__)
        session._failed = True
        session._maintenance = None
        if session.repository is not None and not session._storage_lost:
            try:
                await session.repository.interrupted()
            except Exception as marking_error:
                session._storage_lost = True
                logger.error('Turn cleanup failure: input_id=%s phase=interrupted category=%s', input_id, type(marking_error).__name__)
        from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
        if isinstance(error, ModelCallLimitExceededError):
            error = TurnIncompleteError('本轮主模型调用次数已达上限；已发生效果保留，请核实操作回执。')
        if isinstance(error, ToolCleanupError):
            session._storage_lost = True
        # 已保存回答由入口只读恢复，索引或交付错误不能改写它。
        if saved_result is not None:
            error.turn_result = saved_result
        if not isinstance(error, asyncio.CancelledError):
            error.turn_message = ('会话存储或清理未确认，请重新打开后核实。' if session._storage_lost else
                '本轮超时。' if isinstance(error, TimeoutError) else
                '本轮未完成（' + type(error).__name__ + '）。') + '已发生的操作不会自动回滚，未自动重试。'
        if session._storage_lost and isinstance(error, asyncio.CancelledError):
            raise SessionStorageError('Session connection lost; cleanup finished, reopen the conversation.') from None
        raise error
    finally:
        if observer is not None:
            observer.usage.total_seconds = monotonic() - started
            observer.usage.first_status_seconds = timings.get('first_status_seconds')
            observer.usage.first_body_seconds = timings.get('first_body_seconds')
            from dataclasses import asdict
            logger.info('Turn metrics: input_id=%s metrics=%s', input_id, asdict(observer.usage))
        session._running = False
        session._active_task = None

