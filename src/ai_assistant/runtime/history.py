"""有界 checkpoint 读取：复用原生分页，不拥有另一份会话存储。"""
from ai_assistant.storage.sessions import SessionStorageError
from langgraph.errors import GraphBubbleUp


class CheckpointReader:
    """实例仅供一次读取或一轮复用；权限由消费者每次检查。"""

    def __init__(self, history, threads):
        self.history = history
        self.threads = frozenset(threads)
        self.pages = {}

    async def snapshots(self, config, *, before=None, limit=100):
        values = config.get('configurable', {})
        if values.get('thread_id') not in self.threads:
            raise SessionStorageError('历史定位不属于当前会话。')
        key = (values.get('thread_id'), values.get('checkpoint_ns', ''),
               values.get('checkpoint_id'), repr(before), limit)
        if key not in self.pages:
            try:
                self.pages[key] = [snapshot async for snapshot in self.history(
                    config, before=before, limit=limit)]
            except (SessionStorageError, GraphBubbleUp):
                raise
            except Exception as error:
                raise SessionStorageError('历史存储读取失败；未重新执行工具。') from error
        for snapshot in self.pages[key]:
            yield snapshot

    async def walk(self, config, *, before=None, limit=100):
        while True:
            count = 0
            async for snapshot in self.snapshots(config, before=before, limit=limit):
                count += 1
                before = getattr(snapshot, 'config', None)
                yield snapshot
            if count < limit:
                break
            if before is None:
                raise SessionStorageError('历史分页缺少定位。')


async def latest_state(agent, thread_id):
    """读取最新实际状态，失败时禁止回退到旧完成点。"""
    try:
        return await agent.aget_state({'configurable': {'thread_id': thread_id}})
    except Exception as error:
        raise SessionStorageError('最新会话读取失败；未重新执行工具。') from error


def recovery_values(values):
    """只复制对话数据，不复制原生任务、调度或 pending writes。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    messages = list(values.get('messages', []))
    last_user = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
                     and not m.additional_kwargs.get('lc_source')), default=-1)
    body_id = (values.get('turn_result') or {}).get('body_locator', {}).get('message_id')
    messages = [m for i, m in enumerate(messages) if not (
        i > last_user and isinstance(m, AIMessage) and not m.tool_calls and (not body_id or m.id != body_id))]
    results = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    paired = []
    for message in messages:
        paired.append(message)
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call['id'] not in results:
                    paired.append(ToolMessage('结果未确认；此记录不证明成功、失败或未执行，禁止自动续跑。',
                        # 占位只记录效果未知，下一轮不自动续跑。

                        id='unconfirmed:' + call['id'], tool_call_id=call['id'], name='unconfirmed_result',
                        status='error', artifact={'execution_fact': {'control': 'stop', 'error_category': 'unknown'}}))
    keys = ('conversation_title', 'completed_turns', 'session_state',
            'last_counted_input_id', 'active_skills')
    return {**{key: values[key] for key in keys if key in values}, 'messages': paired}


def saved_turn(values, *, input_id=None, threads=None):
    """当前格式统一只读判断；完成索引不决定已保存正文的有效性。"""
    from ai_assistant.runtime.presentation import saved_body
    from ai_assistant.runtime.outcomes import TERMINAL_STATES
    result = values.get('turn_result') or {}
    locator = result.get('body_locator') or {}
    if (not result.get('input_id') or input_id is not None and result['input_id'] != input_id
            or result.get('status') not in TERMINAL_STATES
            or threads is not None and locator.get('thread_id') not in threads):
        return None
    body = saved_body(result, values.get('messages', []))
    return (result, body) if body is not None else None


async def read_saved_turn(saver, session, input_id):
    """渠道只读恢复使用此入口，读取故障不静默换用更旧记录。"""
    async def read(config):
        try:
            checkpoint = await saver.aget_tuple(config)
            return checkpoint.checkpoint['channel_values'] if checkpoint else {}
        except Exception as error:
            raise SessionStorageError('已保存回答读取失败；未重新执行。') from error
    values = await read({'configurable': {'thread_id': session['thread_id']}})
    found = saved_turn(values, input_id=input_id, threads=session['thread_ids'])
    if found is None and session.get('last_input_id') == input_id and session.get('complete'):
        values = await read(session['complete'])
        found = saved_turn(values, input_id=input_id, threads=session['thread_ids'])
    return found
