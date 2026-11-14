"""Immutable submissions and lease-supervised native indexing."""
import asyncio
import hashlib
import mimetypes
import os
from pathlib import Path
import stat
from sqlalchemy.exc import DBAPIError, OperationalError
from ai_assistant.knowledge import IngestionError, KnowledgeError
from ai_assistant.knowledge.loading import parse_snapshot, require_content
from ai_assistant.knowledge.diagnostics import write_diagnostic
from ai_assistant.knowledge.metadata import normalize, terms
from ai_assistant.knowledge.models import embedding


def snapshot(workspace, relative, directory):
    """读取工作区普通文件，按内容摘要保存只读快照。"""
    target = workspace._path(relative)

    # 不跟随最终路径的符号链接；非阻塞打开后再检查是否为普通文件。
    descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    import tempfile
    digest = hashlib.sha256()
    temporary = None
    try:
        with os.fdopen(descriptor, 'rb') as source, tempfile.NamedTemporaryFile(dir=directory, delete=False) as dest:
            temporary = Path(dest.name)
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise KnowledgeError('资料必须是普通文件。')
            while data := source.read(1024 * 1024):
                digest.update(data)
                dest.write(data)
            dest.flush()
            os.fsync(dest.fileno())

        # 按内容摘要命名并只读保存；已有同名快照时核对其摘要。
        # Keep suffix only as a loader hint, never to select a custom parser.
        final = directory / (digest.hexdigest() + target.suffix.lower())
        os.chmod(temporary, 0o400)
        try:
            os.link(temporary, final)
        except FileExistsError:
            if hashlib.sha256(final.read_bytes()).hexdigest() != digest.hexdigest():
                raise KnowledgeError('不可变资料快照校验失败。')
        return target.as_uri(), digest.hexdigest(), final.resolve(), target.name
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


async def submit(store, workspace, relative, *, tags=()):
    """在线程中生成快照，然后登记异步索引任务。"""
    uri, digest, path, name = await asyncio.to_thread(snapshot, workspace, relative, store.config.materials_dir)
    return await store.submit(source_uri=uri, digest=digest, snapshot=path, metadata=dict(
        source_name=name, media_type=mimetypes.guess_type(name)[0] or 'application/octet-stream', tags=list(tags)))


async def pgvector_store(config, embeddings):
    """创建本次操作独占的向量存储引擎，不初始化数据库结构。"""
    from langchain_postgres import PGEngine, PGVectorStore
    from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
    # A separate engine owned by this operation; no schema initialization here.
    engine = PGEngine.from_connection_string(config.database_url.get_secret_value(), hide_parameters=True)
    try:
        store = await PGVectorStore.create(engine, embeddings, 'rag_chunks', metadata_columns=[],
            hybrid_search_config=HybridSearchConfig(tsv_column='content_tsv', tsv_lang='pg_catalog.simple'))
        return engine, store
    except BaseException:
        await engine.close()
        raise


async def prepare_chunks(store, job):
    """校验快照和配置，解析正文并生成带来源元数据的片段。"""
    c = store.config

    # 不允许不同索引配置的 worker 接手任务，也不处理摘要不一致的快照。
    if job['config_hash'] != c.fingerprint():
        raise KnowledgeError('任务索引配置与 Worker 不一致；请显式重新提交。')
    path = Path(job['snapshot'])
    digest = await asyncio.to_thread(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
    if digest != job['digest']:
        raise KnowledgeError('资料快照摘要不匹配。')

    # 飞书结构化快照直接恢复 Document，其余格式交给解析器。
    if path.suffix == '.ragjson':
        import json
        from langchain_core.documents import Document
        docs = [Document(**item) for item in json.loads(await asyncio.to_thread(path.read_text, encoding='utf-8'))]
    else:
        docs = await parse_snapshot(path, timeout=c.parse_timeout)
    docs = require_content(docs, 'empty_document', '未提取到有效正文。')

    # 将知识库、文档版本、来源和构建 ID 写入统一元数据。
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    identity = job['metadata'] | dict(knowledge_base_id=job['kb'], document_id=job['document_id'],
        document_version_id=job['version_id'], source_uri=job['source_uri'],
        parse_config_fingerprint=job['config_hash'], index_version=job['build_id'])
    parser = {'.ragjson':'feishu','.txt':'text','.md':'text','.pdf':'pypdf','.csv':'csv','.xlsx':'openpyxl'}.get(path.suffix,'unstructured')
    normalized = [normalize(doc, identity=identity, parser=parser,
        mapping={'page_start':'page_number','page_end':'page_number'} if parser=='pypdf' else None) for doc in docs]
    if path.suffix in ('.ragjson','.csv','.xlsx'):
        for doc, original in zip(normalized, docs):
            doc.metadata.update(original.metadata)

    # 递归分块保留原始空白和起始位置，便于后续定位证据。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c.chunk_size, chunk_overlap=c.chunk_overlap, length_function=len,
        add_start_index=True, strip_whitespace=False,
        separators=['\n\n', '\n', '。', '！', '？', '；', ' ', ''])
    chunks = await asyncio.to_thread(splitter.split_documents, normalized)
    chunks = require_content(chunks, 'empty_chunks', '未生成有效片段。')
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = i
    return chunks


async def build(store, job):
    """完成解析、向量索引、关键词索引并最终发布构建。"""
    c = store.config
    limiter = getattr(store, 'parse_limiter', None)
    from ai_assistant.observability import span

    # 按需受解析并发限制；解析成功后才进入向量化。
    with span('knowledge.index.parse', job_id=job.get('id')):
        if limiter is None:
            chunks = await prepare_chunks(store, job)
        else:
            async with limiter:
                chunks = await prepare_chunks(store, job)
    from langchain_classic.indexes import SQLRecordManager, aindex
    with span('knowledge.index.embed', job_id=job.get('id')):
        async with embedding(c) as model:
            engine, vectors = await vector_store(c, model)
            try:
                # Unique namespace AND build metadata fence all full-cleanup operations.
                manager = SQLRecordManager('rag:' + job['build_id'], engine=store.engine, async_mode=True)
                await aindex(chunks, manager, vectors, batch_size=c.embedding.batch_size,
                             cleanup='full', source_id_key='document_id', key_encoder='sha256')
                # All embedding requests and vector writes finish before keyword preparation/publication.
                rows = await store.chunk_rows(job['build_id'])
                if not any(row.content.strip() for row in rows):
                    raise IngestionError('empty_index', '当前构建未写入有效片段，已停止发布。',
                                         f'build_id={job["build_id"]}; stored_rows={len(rows)}')
                values = await asyncio.to_thread(lambda: [dict(id=r.langchain_id, words=terms(r.content)) for r in rows])
                await store.update_chunk_tsv(values)
            finally:
                await engine.close()
    with span('knowledge.index.publish', job_id=job.get('id')):
        # 只有有效片段和关键词索引准备完成后才发布供检索的构建。
        await store.publish(job)


async def run_once(store):
    """领取并执行一个索引任务；没有任务时返回 False。"""
    job = await store.claim()
    if job is None:
        return False
    from ai_assistant.observability import event, span
    with span('knowledge.index', job_id=job.get('id'), document_id=job.get('document_id'),
              build_id=job.get('build_id')):
        event('knowledge.index.started', job_id=job.get('id'), status='running')
        return await _run_claimed(store, job)


async def _run_claimed(store, job):
    """在租约、心跳和截止时间约束下执行已领取任务。"""
    async def heartbeat():
        while True:
            await asyncio.sleep(store.config.heartbeat_seconds)
            await store.heartbeat(job)

    from ai_assistant.observability import event
    error_code = None
    task = asyncio.create_task(build(store, job))
    pulse = asyncio.create_task(heartbeat())
    try:
        # 并发监督构建和心跳；任一异常或任务截止都会进入失败处理。
        # A database lease is authoritative, including after event-loop stalls.
        remaining = max(0, (job['deadline'] - __import__('datetime').datetime.now(__import__('datetime').timezone.utc)).total_seconds())
        async with asyncio.timeout(remaining):
            done, _ = await asyncio.wait((task, pulse), return_when=asyncio.FIRST_COMPLETED)
            for completed in done:
                completed.result()
    except asyncio.CancelledError:
        # Leave the task leased; another worker recovers after expiry.
        event('knowledge.index.failed', job_id=job.get('id'), status='cancelled', category='CancelledError')
        raise
    except Exception as exc:
        # Resolve an unknown commit outcome before marking/retrying anything.
        status = await store.status(job['id'])
        if status and status['state'] != 'succeeded':
            # 仅将识别出的数据库连接故障标为临时错误，交给存储层处理。
            transient = isinstance(exc, (OperationalError, ConnectionError)) or (
                isinstance(exc, DBAPIError) and exc.connection_invalidated)
            code = exc.code if isinstance(exc, IngestionError) else ('transient_database' if transient else 'ingestion_failed')
            error_code = code
            message = 'transient_database' if transient else (str(exc) if isinstance(exc, KnowledgeError) else type(exc).__name__)
            diagnostic_path = None
            if isinstance(exc, IngestionError):
                # 详细诊断单独写入本地文件，失败不覆盖原始任务错误。
                try:
                    diagnostic_path = await asyncio.to_thread(write_diagnostic, store.config, job,
                                                             code, message, exc.diagnostic_text)
                except (OSError, ValueError):
                    message += '（诊断文件未能保存。）'
            await store.fail(job, transient=transient, error=message,
                             error_code=code, diagnostic_path=diagnostic_path)
        event('knowledge.index.failed', job_id=job.get('id'), status='failed',
              category=type(exc).__name__, reason=error_code)
    else:
        event('knowledge.index.published', job_id=job.get('id'), document_id=job.get('document_id'),
              build_id=job.get('build_id'), status='succeeded')
    finally:
        # 构建或心跳结束后取消另一方，并等待全部清理完成。
        for running in (task, pulse):
            if not running.done():
                running.cancel()
        await asyncio.gather(task, pulse, return_exceptions=True)
    return True


async def worker(store, *, once=False, app_config=None):
    """运行同步扫描、并发索引与同步汇总；once 只检查一轮。"""
    if once:
        if app_config is not None:
            from ai_assistant.knowledge.sync import sync_once
            await sync_once(app_config, store)
        await run_once(store)
        await store.finish_syncs()
        return

    # 分别限制解析和向量化并发；索引循环数量由配置控制。
    store.parse_limiter = asyncio.Semaphore(store.config.parse_concurrency)
    from ai_assistant.knowledge.models import embedding_limiter
    limiter_token = embedding_limiter.set(asyncio.Semaphore(store.config.embedding_concurrency))

    # 三类循环分别处理索引任务、来源同步和同步状态汇总。
    async def indices():
        while True:
            if not await run_once(store):
                await asyncio.sleep(1)
    async def scans():
        from ai_assistant.knowledge.sync import sync_once
        while True:
            if app_config is None or not await sync_once(app_config, store):
                await asyncio.sleep(1)
    async def summaries():
        while True:
            await store.finish_syncs()
            await asyncio.sleep(1)
    tasks = [asyncio.create_task(indices()) for _ in range(store.config.index_concurrency)]
    tasks.extend([asyncio.create_task(scans()), asyncio.create_task(summaries())])
    try:
        await asyncio.gather(*tasks)
    finally:
        # 退出时等待所有子任务清理，再恢复上下文中的向量化限制器。
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        embedding_limiter.reset(limiter_token)


async def vector_store(config, embeddings):
    """按配置的存储后端创建向量存储。"""
    from ai_assistant.storage.backends import backend_factory
    return await backend_factory('vectors', config.backend)(config, embeddings)
