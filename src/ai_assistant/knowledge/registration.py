"""显式知识登记：复用管理事务，恢复只读回执。"""
from hashlib import sha256


async def register_document(config, reference, *, identity='user', tags=None,
                            input_id=None, principal='local'):
    from ai_assistant.knowledge.sync import register, settings, resolve_submission
    if input_id is None:
        # CLI 每次显式调用都是新请求，文档版本和构建仍由存储去重。
        return await register(config, reference, identity=identity, tags=tags)
    from ai_assistant.storage.rag import RagStore
    from ai_assistant.knowledge.management import receipt, apply
    store = RagStore(settings(config))
    principal = principal + ':' + identity + ':ingest'
    request_id = input_id + ':' + sha256(reference.encode()).hexdigest()
    try:
        previous = await receipt(store, principal, request_id, action='add', target=reference)
        if previous is not None:
            return previous
        submission = await resolve_submission(config, reference, identity=identity, tags=tags)
        return await apply(store, principal, request_id, 'add', reference, submission=submission)
    finally:
        await store.close()
