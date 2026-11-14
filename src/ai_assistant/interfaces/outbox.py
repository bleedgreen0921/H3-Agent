"""通用 outbox 批次登记；payload 由渠道适配器构造，core 不解释消息格式。"""
from ai_assistant.observability import observed
from ai_assistant.storage.queues import InboxRepository, OutboxRepository


@observed("reply.enqueue", lambda conn, input_id, parts, **kw: {"input_id": input_id})
async def enqueue_parts(conn, input_id, parts):
    """在调用方事务中幂等登记回复批次并标记输入已处理。"""
    inbox, outbox = InboxRepository(conn), OutboxRepository(conn)

    # 锁定输入后检查既有批次，避免并发或重复登记不同回复。
    await outbox.lock_inbox(input_id)
    existing = await outbox.first(input_id)
    if existing and (existing.get('body') or {}).get('delivery', {}).get('version') != 1:
        from ai_assistant.storage.sessions import SessionError
        raise SessionError('不支持的投递格式；未重新登记或发送。')

    # 已有当前格式批次直接复用；没有批次时才组装并写入分段。
    if not existing:
        records = []
        for part in parts:
            body = {**part.payload, 'delivery': part.delivery}
            if part.presentation_ref:
                body['presentation_ref'] = part.presentation_ref
            records.append((part.part, part.id, body))
        await outbox.enqueue_parts(input_id, records)

    # 这里表示输入处理完成，实际消息投递由渠道发送器负责。
    await inbox.mark_completed(input_id)
