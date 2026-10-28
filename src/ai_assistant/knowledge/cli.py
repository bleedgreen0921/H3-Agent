"""Explicit RAG administration, with no automatic migrations or model downloads."""
import asyncio
import json
from pathlib import Path
from ai_assistant.config import ConfigError
from ai_assistant.knowledge import KnowledgeError


def add_commands(commands):
    parser = commands.add_parser('rag', help='Local persistent knowledge retrieval')
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('submit', 'status', 'worker', 'search', 'delete', 'rebuild', 'check', 'sources', 'sync', 'documents'):
        command = actions.add_parser(action)
        command.add_argument('--config', default='assistant.toml')
        if action in ('sources', 'documents'):
            command.add_argument('--page', type=int, default=1)
        if action == 'submit':
            command.add_argument('path')
            command.add_argument('--tag', action='append', default=[])
        if action == 'sync':
            command.add_argument('--source')
            command.add_argument('--wait', action='store_true')
        if action == 'status':
            command.add_argument('job_id')
            command.add_argument('--diagnostics', action='store_true')
        if action == 'worker':
            command.add_argument('--once', action='store_true')
        if action == 'search':
            command.add_argument('query')
            command.add_argument('--document-id')
            command.add_argument('--source-id')
            command.add_argument('--media-type')
            command.add_argument('--source-name')
            command.add_argument('--tag', action='append', default=[])
        if action in ('delete', 'rebuild'):
            command.add_argument('document_id')
        if action == 'check':
            command.add_argument('--database', action='store_true')
            command.add_argument('--services', action='store_true')


async def execute(config, args):
    c = config.knowledge
    if c is None:
        raise ConfigError('未配置 [knowledge]。')
    base = config.config_path.parent if config.config_path else Path.cwd()
    # All configured relative paths resolve against the TOML, not worker cwd.
    from ai_assistant.knowledge.config import resolve_knowledge_paths
    c = resolve_knowledge_paths(c, base)
    if args.action == 'check':
        if not args.database and not args.services:
            args.database = args.services = True
        if args.services:
            from ai_assistant.knowledge.models import embedding, reranker
            async with embedding(c) as model:
                vector = await model.aembed_query('知识检索测试')
            scores = await asyncio.to_thread(reranker(c).score, [('问题', '答案')])
            print(f'模型服务检查通过：维数={len(vector)}，重排数={len(scores)}。')
        if not args.database:
            return 0
    from ai_assistant.storage.rag import RagStore
    store = RagStore(c)
    try:
        if args.action in ('sources', 'documents'):
            from ai_assistant.knowledge.management import sources
            if args.page < 1:
                raise KnowledgeError('页码必须大于 0。')
            rows = await sources(config, store) if args.action == 'sources' else await store.document_list()
            print(json.dumps(rows[(args.page - 1) * 10:args.page * 10], ensure_ascii=False, default=str))
        elif args.action == 'sync':
            from ai_assistant.knowledge.sync import configure, wait_sync
            await configure(config,store)
            ident = await store.request_sync({'source_id': args.source} if args.source else {})
            print(ident, flush=True)
            if args.wait:
                print(json.dumps(await wait_sync(config,ident),ensure_ascii=False,default=str))
        elif args.action == 'submit':
            from ai_assistant.knowledge.sync import register
            print(json.dumps(await register(config, args.path, tags=args.tag), ensure_ascii=False))
        elif args.action == 'status':
            from ai_assistant.knowledge.management import status
            result = await status(store, args.job_id)
            if result is None:
                raise KnowledgeError('任务不存在。')
            print(json.dumps(result, ensure_ascii=False, default=str))
            if args.diagnostics and result['kind'] == 'index':
                import sys
                from ai_assistant.knowledge.diagnostics import read_diagnostic
                detail = await asyncio.to_thread(read_diagnostic, c, result)
                print(detail, file=sys.stderr)
        elif args.action == 'worker':
            from ai_assistant.knowledge.ingestion import worker
            await worker(store, once=args.once, app_config=config)
        elif args.action == 'search':
            from ai_assistant.knowledge.retrieval import create_knowledge_retriever, render_evidence
            filters = {key: getattr(args, key) for key in ('document_id', 'media_type', 'source_name') if getattr(args, key)}
            if args.tag:
                filters['tags'] = args.tag
            from ai_assistant.knowledge.sync import configure,wait_sync
            await configure(config,store)
            scope = {key:getattr(args,key) for key in ('document_id','source_id') if getattr(args,key,None)}
            ident = await store.request_sync(scope)
            state = await wait_sync(config,ident)
            docs = await create_knowledge_retriever(c, filters=filters, builds=state['builds']).ainvoke(args.query)
            print(render_evidence(docs))
            if state['errors']:
                print('本轮覆盖不完整：'+json.dumps(state['errors'],ensure_ascii=False))
        elif args.action == 'delete':
            print(f'退出查询范围的文档数：{await store.delete(args.document_id)}')
        elif args.action == 'rebuild':
            version = await store.version_for_rebuild(args.document_id)
            if version is None:
                raise KnowledgeError('文档不存在或已移除；请先显式添加恢复。')
            print(await store.submit(source_uri=version['source_uri'], digest=version['digest'],
                snapshot=version['snapshot'], metadata=version['metadata'], force=True, restore=False))
        elif args.action == 'check':
            if args.database:
                from sqlalchemy import text
                async with store.engine.connect() as conn:
                    for table in ('rag_documents', 'rag_versions', 'rag_jobs', 'rag_builds', 'rag_chunks', 'upsertion_record', 'rag_sources', 'rag_sync_runs', 'rag_query_runs', 'rag_management_operations'):
                        if not (await conn.execute(text('SELECT to_regclass(:name)'), dict(name=table))).scalar():
                            raise KnowledgeError('缺少 schema，请显式运行 init。')
                    columns = set((await conn.execute(text("""SELECT attname FROM pg_attribute
                        WHERE attrelid=to_regclass('rag_builds') AND attnum>0 AND NOT attisdropped"""))).scalars())
                    if not {'attempt_number', 'error_code', 'error_message', 'diagnostic_path'} <= columns:
                        raise KnowledgeError('知识库 schema 需要更新，请显式运行 init。')
                print('数据库及 schema 检查通过。')

    finally:
        await store.close()
    return 0
