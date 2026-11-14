"""任务输入的原生消息记录；不保存执行状态，也不推断语义关联。"""
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from langchain_core.messages import AIMessage, HumanMessage

KEY = 'assistant_input'


def metadata(message):
    """仅读取当前版本的输入关联元数据。"""
    value = message.additional_kwargs.get(KEY, {})
    return value if isinstance(value, dict) and value.get('version') == 1 else {}


def user_message(session_id, input_id, text, *, source=None):
    """身份来自可信 runtime；回答正文不包含路由命令。"""
    # 根据会话和输入 ID 生成稳定消息 ID，便于识别重复输入。
    kind = 'request'
    return HumanMessage(
        content=text,
        id=str(uuid5(NAMESPACE_URL, f'{session_id}:{input_id}:{kind}')),
        additional_kwargs={
            'user_source': text if source is None else source,
            KEY: {'version': 1, 'input_id': input_id, 'kind': kind, 'received_at': datetime.now(timezone.utc).isoformat()},
        },
    )


async def recorded_messages(agent, thread_id, *, reader=None, input_id=None):
    """分页由原生 saver 实现；读取失败必须阻止后续执行，不能视作空历史。"""
    from ai_assistant.runtime.history import CheckpointReader
    reader = reader or CheckpointReader(agent.aget_state_history, [thread_id])
    records = {}
    async for snapshot in reader.walk({'configurable': {'thread_id': thread_id}}, limit=20 if input_id else 100):
        for message in snapshot.values.get('messages', []):
            if message.id and (metadata(message) or isinstance(message, AIMessage)):
                records.setdefault(message.id, message)
                if input_id is not None and metadata(message).get('input_id') == input_id:
                    return [message]
    return [] if input_id is not None else sorted(records.values(), key=lambda m: metadata(m).get('received_at', ''))

