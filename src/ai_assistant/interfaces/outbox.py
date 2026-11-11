"""通用 outbox 批次登记；payload 由渠道适配器构造，core 不解释消息格式。"""
from ai_assistant.observability import observed
from ai_assistant.storage.queues import InboxRepository, OutboxRepository


@observed("reply.enqueue", lambda conn, input_id, parts, **kw: {"input_id": input_id})
async def enqueue_parts(conn, input_id, parts):
    inbox, outbox = InboxRepository(conn), OutboxRepository(conn)
    await outbox.lock_inbox(input_id)
    existing = await outbox.first(input_id)
    if existing and (existing.get('body') or {}).get('delivery', {}).get('version') != 1:
        from ai_assistant.storage.sessions import SessionError
        raise SessionError('不支持的投递格式；未重新登记或发送。')
    if not existing:
        records = []
        for part in parts:
            body = {**part.payload, 'delivery': part.delivery}
            if part.presentation_ref:
                body['presentation_ref'] = part.presentation_ref
            records.append((part.part, part.id, body))
        await outbox.enqueue_parts(input_id, records)
    await inbox.mark_completed(input_id)
