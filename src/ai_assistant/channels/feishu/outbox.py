"""飞书 Markdown 批次构造；批次写入由通用 outbox 负责。"""
from uuid import NAMESPACE_URL, uuid5

from ai_assistant.channels.base import OutboundPart
from ai_assistant.channels.feishu.message import markdown_message, split_markdown
from ai_assistant.interfaces.delivery import metadata
from ai_assistant.interfaces.outbox import enqueue_parts
from ai_assistant.observability import observed


# 用输入 ID 关联回复登记的观测事件，不记录正文。
@observed("reply.enqueue.text", lambda conn, settings, row, text, **kw: {"input_id": row["id"]})
async def enqueue_text(conn, settings, row, text, *, purpose='answer', snapshot=None, presentation_ref=None):
    """将飞书私聊回复转换为有序分段，并交给通用 outbox 登记。"""
    # 接收端之外的第二道边界：历史群请求也不能登记新的回复。
    if (row.get('body') or {}).get('group') is not False:
        from ai_assistant.config import ConfigError
        raise ConfigError('仅支持私聊回复，未登记群消息投递。')

    # 按完整请求字节预算拆分 Markdown，并为每段保留稳定顺序。
    parts = []
    for part, value in enumerate(split_markdown(text, row['body']['chat_id'])):
        # 输入 ID 和分段序号决定稳定 UUID，便于重复登记时去重。
        identity = row['id'] + ':reply:' + str(part)

        # 正文、展示快照和可选工作卡引用随分段一并保存。
        payload = {'chat_id': row['body']['chat_id'], 'group': False,
                   'message': markdown_message(value), 'presentation': snapshot}
        if presentation_ref:
            payload['presentation_ref'] = presentation_ref

        # 投递元数据包含版本、尝试次数和截止时间，由发送器消费。
        parts.append(OutboundPart(
            id=str(uuid5(NAMESPACE_URL, identity)), input_id=row['id'], part=part,
            payload=payload, delivery={**metadata(settings), 'purpose': purpose},
            presentation_ref=presentation_ref))

    # 通用登记层负责锁定输入、处理已有批次及写入队列；此处不发送消息。
    await enqueue_parts(conn, row['id'], parts)
