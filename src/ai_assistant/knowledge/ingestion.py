"""Immutable submissions and lease-supervised native indexing."""
import asyncio
import hashlib
import mimetypes
import os
from pathlib import Path
import stat
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError
from ai_assistant.knowledge import IngestionError, KnowledgeError
from ai_assistant.knowledge.loading import parse_snapshot, require_content
from ai_assistant.knowledge.diagnostics import write_diagnostic
from ai_assistant.knowledge.metadata import normalize, terms
from ai_assistant.knowledge.models import embedding


def snapshot(workspace, relative, directory):
    target = workspace._path(relative)
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
    uri, digest, path, name = await asyncio.to_thread(snapshot, workspace, relative, store.config.materials_dir)
    return await store.submit(source_uri=uri, digest=digest, snapshot=path, metadata=dict(
        source_name=name, media_type=mimetypes.guess_type(name)[0] or 'application/octet-stream', tags=list(tags)))


async def vector_store(config, embeddings):
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


async def build(store, job):
    c = store.config
    if job['config_hash'] != c.fingerprint():
        raise KnowledgeError('任务索引配置与 Worker 不一致；请显式重新提交。')
    path = Path(job['snapshot'])
    digest = await asyncio.to_thread(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
    if digest != job['digest']:
        raise KnowledgeError('资料快照摘要不匹配。')
    docs = await parse_snapshot(path, timeout=c.parse_timeout)
    docs = require_content(docs, 'empty_document', '未提取到有效正文。')
    async with embedding(c) as model:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_classic.indexes import SQLRecordManager, aindex
        identity = job['metadata'] | dict(knowledge_base_id=job['kb'], document_id=job['document_id'],
            document_version_id=job['version_id'], source_uri=job['source_uri'],
            parse_config_fingerprint=job['config_hash'], index_version=job['build_id'])
        normalized = [normalize(doc, identity=identity, parser='unstructured') for doc in docs]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=c.chunk_size, chunk_overlap=c.chunk_overlap, length_function=len,
            add_start_index=True, strip_whitespace=False,
            separators=['\n\n', '\n', '。', '！', '？', '；', ' ', ''])
        chunks = await asyncio.to_thread(splitter.split_documents, normalized)
        chunks = require_content(chunks, 'empty_chunks', '未生成有效片段。')
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
        engine, vectors = await vector_store(c, model)
        try:
            # Unique namespace AND build metadata fence all full-cleanup operations.
            manager = SQLRecordManager('rag:' + job['build_id'], engine=store.engine, async_mode=True)
            await aindex(chunks, manager, vectors, batch_size=c.embedding.batch_size,
                         cleanup='full', source_id_key='document_id', key_encoder='sha256')
            # All embedding requests and vector writes finish before keyword preparation/publication.
            async with store.engine.connect() as conn:
                rows = (await conn.execute(text('''SELECT langchain_id,content FROM rag_chunks
                    WHERE langchain_metadata->>'index_version'=:b'''), dict(b=job['build_id']))).all()
            if not any(row.content.strip() for row in rows):
                raise IngestionError('empty_index', '当前构建未写入有效片段，已停止发布。',
                                     f'build_id={job["build_id"]}; stored_rows={len(rows)}')
            values = await asyncio.to_thread(lambda: [dict(id=r.langchain_id, words=terms(r.content)) for r in rows])
            async with store.engine.begin() as conn:
                if values:
                    await conn.execute(text("UPDATE rag_chunks SET content_tsv=to_tsvector('pg_catalog.simple',:words) WHERE langchain_id=:id"), values)
            await store.publish(job)
        finally:
            await engine.close()


async def run_once(store):
    job = await store.claim()
    if job is None:
        return False

    async def heartbeat():
        while True:
            await asyncio.sleep(store.config.heartbeat_seconds)
            await store.heartbeat(job)

    task = asyncio.create_task(build(store, job))
    pulse = asyncio.create_task(heartbeat())
    try:
        # A database lease is authoritative, including after event-loop stalls.
        remaining = max(0, (job['deadline'] - __import__('datetime').datetime.now(__import__('datetime').timezone.utc)).total_seconds())
        async with asyncio.timeout(remaining):
            done, _ = await asyncio.wait((task, pulse), return_when=asyncio.FIRST_COMPLETED)
            for completed in done:
                completed.result()
    except asyncio.CancelledError:
        # Leave the task leased; another worker recovers after expiry.
        raise
    except Exception as exc:
        # Resolve an unknown commit outcome before marking/retrying anything.
        status = await store.status(job['id'])
        if status and status['state'] != 'succeeded':
            transient = isinstance(exc, (OperationalError, ConnectionError)) or (
                isinstance(exc, DBAPIError) and exc.connection_invalidated)
            code = exc.code if isinstance(exc, IngestionError) else ('transient_database' if transient else 'ingestion_failed')
            message = 'transient_database' if transient else (str(exc) if isinstance(exc, KnowledgeError) else type(exc).__name__)
            diagnostic_path = None
            if isinstance(exc, IngestionError):
                try:
                    diagnostic_path = await asyncio.to_thread(write_diagnostic, store.config, job,
                                                             code, message, exc.diagnostic_text)
                except (OSError, ValueError):
                    message += '（诊断文件未能保存。）'
            await store.fail(job, transient=transient, error=message,
                             error_code=code, diagnostic_path=diagnostic_path)
    finally:
        for running in (task, pulse):
            if not running.done():
                running.cancel()
        await asyncio.gather(task, pulse, return_exceptions=True)
    return True


async def worker(store, *, once=False):
    while True:
        found = await run_once(store)
        if once:
            return
        if not found:
            await asyncio.sleep(1)
