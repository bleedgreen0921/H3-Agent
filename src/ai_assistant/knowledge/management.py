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
        return dict(version=1, source_id=source, sync_id=sync, status='submitted',
                    searchable_confirmed=False, message='来源已登记，后台任务已提交；尚未确认可检索。')


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
        return dict(row, kind='sync', **sync_facts(row))
    row = await store.status(ident)
    if row is None:
        raise KnowledgeError('任务不存在。')
    return dict(row, kind='index')


async def receipt(store, principal, input_id, *, conn=None, action=None, target=None):
    if conn is None:
        async with store.engine.connect() as connection:
            return await receipt(store, principal, input_id, conn=connection, action=action, target=target)
    row = (await conn.execute(text('''SELECT action,target,result FROM rag_management_operations
        WHERE kb=:kb AND principal=:p AND input_id=:i'''),
        dict(kb=store.config.knowledge_base_id, p=principal, i=input_id))).mappings().first()
    if row is not None and action is not None and (row['action'] != action or row['target'] != target):
        raise KnowledgeError('同一输入已有不同的登记目标，拒绝覆盖或重放。')
    return row['result'] if row else None


async def apply(store, principal, input_id, action, target=None, *, submission=None):
    """解析在外，登记/恢复/同步/移除和回执在同一事务内；重复输入只读回执。"""
    async with store.engine.begin() as conn:
        key = packed([store.config.knowledge_base_id, principal, input_id])
        await conn.execute(text('SELECT pg_advisory_xact_lock(hashtextextended(:key,0))'), dict(key=key))
        previous = await receipt(store, principal, input_id, conn=conn, action=action, target=target)
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


def sync_facts(row):
    """ready 只表示处理结束；可检索构建与覆盖缺口分别展示。"""
    builds, errors = row.get('builds') or [], row.get('errors') or []
    finished = row['state'] == 'ready'
    coverage = 'pending' if not finished else 'partial' if builds and errors else 'failed' if errors else 'available' if builds else 'empty'
    descriptions = {'pending': '后台处理中', 'partial': '部分可检索，仍有覆盖缺口',
        'failed': '处理结束，全部失败，无已确认可检索构建', 'available': '处理结束，已有可检索构建',
        'empty': '处理结束，范围为空，无可检索构建'}
    return dict(coverage=coverage, processing_finished=finished, searchable_builds=len(builds),
                coverage_gaps=len(errors), summary=descriptions[coverage])
