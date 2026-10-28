"""私人知识库管理业务：只读查询与原子操作回执，供终端和渠道复用。"""
from uuid import uuid4
from sqlalchemy import text

from ai_assistant.knowledge import KnowledgeError
from ai_assistant.storage.knowledge import packed, transaction


async def register_submission(store, uri, descriptor, *, restore=True, conn=None):
    """显式提交和渠道 add 共用；恢复必须请求新一轮扫描，不能复用已扫描的批次。"""
    async with transaction(store.engine, conn) as conn:
        source = await store.add_submission(uri, descriptor, restore=restore, conn=conn)
        if source is None:
            return {'status': 'removed', 'message': '已移除，需显式添加。'}
        filters = {'source_id': source}
        if restore:
            filters['submission_id'] = str(uuid4())
        sync = await store.request_sync(filters, conn=conn)
        return dict(source_id=source, sync_id=sync)


async def sources(config, store):
    rows = {row['id']: row for row in await store.source_list()}
    # 配置来源尚未同步时也可展示，查询不登记来源。
    for name, row in rows.items():
        if not name.startswith('submitted:'):
            row['enabled'] = name in config.knowledge.sources
    for name, definition in config.knowledge.sources.items():
        row = rows.setdefault(name, dict(id=name, checked_at=None, error=None))
        row.update(config=definition.model_dump(mode='json'), enabled=True)
    return [rows[name] for name in sorted(rows)]


async def status(store, ident):
    row = await store.sync_status(ident)
    if row is not None:
        return dict(row, kind='sync', coverage='incomplete' if row['errors'] else ('complete' if row['state'] == 'ready' else 'pending'))
    row = await store.status(ident)
    if row is None:
        raise KnowledgeError('任务不存在。')
    return dict(row, kind='index')


async def receipt(store, principal, input_id, *, conn=None):
    if conn is None:
        async with store.engine.connect() as connection:
            return await receipt(store, principal, input_id, conn=connection)
    row = (await conn.execute(text('''SELECT result FROM rag_management_operations
        WHERE kb=:kb AND principal=:p AND input_id=:i'''),
        dict(kb=store.config.knowledge_base_id, p=principal, i=input_id))).scalar()
    return row


async def apply(store, principal, input_id, action, target=None, *, submission=None):
    """解析在外，登记/恢复/同步/移除和回执在同一事务内；重复输入只读回执。"""
    async with store.engine.begin() as conn:
        key = packed([store.config.knowledge_base_id, principal, input_id])
        await conn.execute(text('SELECT pg_advisory_xact_lock(hashtextextended(:key,0))'), dict(key=key))
        previous = await receipt(store, principal, input_id, conn=conn)
        if previous is not None:
            return previous
        if action == 'add':
            uri, descriptor = submission
            result = await register_submission(store, uri, descriptor, conn=conn)
        elif action == 'sync':
            result = dict(sync_id=await store.request_sync({'source_id': target} if target else {}, conn=conn))
        elif action == 'remove':
            result = dict(document_id=target, removed=await store.delete(target, conn=conn))
        else:
            raise KnowledgeError('不支持的管理操作。')
        await conn.execute(text('''INSERT INTO rag_management_operations(kb,principal,input_id,action,target,result)
            VALUES(:kb,:p,:i,:a,:t,CAST(:r AS jsonb))'''),
            dict(kb=store.config.knowledge_base_id, p=principal, i=input_id,
                 a=action, t=target, r=packed(result)))
        return result
