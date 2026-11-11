"""飞书 Markdown 批次构造；批次写入由通用 outbox 负责。"""
from uuid import NAMESPACE_URL, uuid5

from ai_assistant.channels.base import OutboundPart
from ai_assistant.channels.feishu.message import markdown_message, split_markdown
from ai_assistant.interfaces.delivery import metadata
from ai_assistant.interfaces.outbox import enqueue_parts
from ai_assistant.observability import observed


@observed("reply.enqueue", lambda conn, settings, row, text, **kw: {"input_id": row["id"]})
async def enqueue_text(conn, settings, row, text, *, purpose='answer', snapshot=None, presentation_ref=None):
    # 接收端之外的第二道边界：历史群请求也不能登记新的回复。
    if (row.get('body') or {}).get('group') is not False:
        from ai_assistant.config import ConfigError
        raise ConfigError('仅支持私聊回复，未登记群消息投递。')
    parts = []
    for part, value in enumerate(split_markdown(text, row['body']['chat_id'])):
        identity = row['id'] + ':reply:' + str(part)
        payload = {'chat_id': row['body']['chat_id'], 'group': False,
                   'message': markdown_message(value), 'presentation': snapshot}
        if presentation_ref:
            payload['presentation_ref'] = presentation_ref
        parts.append(OutboundPart(
            id=str(uuid5(NAMESPACE_URL, identity)), input_id=row['id'], part=part,
            payload=payload, delivery={**metadata(settings), 'purpose': purpose},
            presentation_ref=presentation_ref))
    await enqueue_parts(conn, row['id'], parts)
