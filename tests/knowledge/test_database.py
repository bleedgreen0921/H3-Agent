"""Opt-in integration against a disposable DB; never cleans existing tables."""
import pytest
pytest.importorskip("langchain_postgres")

import asyncio
import os
from uuid import uuid4
import pytest
from pydantic import SecretStr
from sqlalchemy import text
from ai_assistant.knowledge.config import KnowledgeConfig
from ai_assistant.storage.rag import RagStore
from ai_assistant.knowledge import IngestionError, LeaseLost

URL = os.environ.get('RAG_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not URL, reason='Set a disposable RAG_TEST_DATABASE_URL')


async def add_chunk(store, job, content='证据'):
    import json
    metadata = dict(index_version=job['build_id'], document_id=job['document_id'],
                    document_version_id=job['version_id'], knowledge_base_id=store.config.knowledge_base_id)
    async with store.engine.begin() as conn:
        await conn.execute(text("""INSERT INTO rag_chunks(langchain_id,content,embedding,langchain_metadata)
            VALUES(:id,:content,CAST(:vector AS vector),CAST(:metadata AS jsonb))"""),
            dict(id=str(uuid4()), content=content, vector=str([1.] + [0.] * 1023), metadata=json.dumps(metadata)))


def test_fences_recovery_snapshot_and_filters():
    async def scenario():
        store = RagStore(KnowledgeConfig(database_url=SecretStr(URL), knowledge_base_id='test-' + str(uuid4())))
        try:
            await store.init()
            async def submit(digest='a', force=False):
                return await store.submit(source_uri='file:///sample', digest=digest, snapshot='/unused',
                    metadata=dict(source_name='sample', media_type='text/plain', tags=['中文', 'test']), force=force)
            first = await submit()
            assert await submit() == first
            one, other = await asyncio.gather(store.claim(), store.claim())
            old = one or other
            assert old and (one is None or other is None)
            assert await store.active() == []
            await add_chunk(store, old)
            await store.publish(old)
            captured = await store.active()
            assert captured == [old['build_id']]
            assert await store.active({'tags': ['中文'], 'media_type': 'text/plain'}) == captured
            assert await store.active({'tags': ['missing']}) == []
            second = await submit('b')
            expired = await store.claim()
            async with store.engine.begin() as conn:
                await conn.execute(text("UPDATE rag_jobs SET lease_until=now()-interval '1 second' WHERE id=:id"), dict(id=second))
            recovered = await store.claim()
            assert recovered['attempts'] == 2 and recovered['token'] != expired['token']
            with pytest.raises(LeaseLost):
                await store.publish(expired)
            with pytest.raises(LeaseLost):
                await store.heartbeat(expired)
            await store.fail(expired, error='obsolete')
            assert (await store.status(second))['state'] == 'running'
            assert await store.active() == captured
            await add_chunk(store, recovered)
            await store.publish(recovered)
            assert (await store.status(second))['state'] == 'succeeded'
            assert await store.active() == [recovered['build_id']]
            # Historical build persists for already captured queries.
            async with store.engine.connect() as conn:
                assert (await conn.execute(text('SELECT published FROM rag_builds WHERE id=:id'), dict(id=captured[0]))).scalar()
            third = await submit('c')
            obsolete = await store.claim()
            fourth = await submit('d')
            with pytest.raises(LeaseLost):
                await store.publish(obsolete)
            assert (await store.status(third))['state'] == 'superseded'
            current = await store.claim()
            await store.fail(current, transient=False, error='parser_error')
            assert (await store.status(fourth))['state'] == 'failed'
            assert await store.active() == [recovered['build_id']]
            rebuild = await submit('d', force=True)
            rebuilding = await store.claim()
            assert rebuilding['id'] == rebuild
            await store.delete(rebuilding['document_id'])
            with pytest.raises(LeaseLost):
                await store.publish(rebuilding)
            assert await store.active() == []
        finally:
            await store.close()
    asyncio.run(scenario())


def test_native_index_persistence_and_hybrid():
    async def scenario():
        from langchain_core.embeddings import Embeddings
        from langchain_core.documents import Document
        from langchain_classic.indexes import SQLRecordManager, aindex
        from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
        from ai_assistant.knowledge.ingestion import vector_store
        class FakeEmbedding(Embeddings):
            def embed_documents(self, texts):
                return [self.embed_query(t) for t in texts]
            def embed_query(self, text):
                return [1., 0.] + [0.] * 1022
        c = KnowledgeConfig(database_url=SecretStr(URL), knowledge_base_id='test-' + str(uuid4()))
        store = RagStore(c)
        engine = None
        try:
            await store.init()
            engine, vectors = await vector_store(c, FakeEmbedding())
            build = str(uuid4())
            docs = [Document(page_content='中文检索 search_knowledge', metadata=dict(document_id='d', index_version=build, knowledge_base_id=c.knowledge_base_id))]
            manager = SQLRecordManager('test:' + build, engine=store.engine, async_mode=True)
            first = await aindex(docs, manager, vectors, cleanup='full', source_id_key='document_id', key_encoder='sha256')
            assert first['num_added'] == 1
            manager2 = SQLRecordManager('test:' + build, engine=store.engine, async_mode=True)
            second = await aindex(docs, manager2, vectors, cleanup='full', source_id_key='document_id', key_encoder='sha256')
            assert second['num_skipped'] == 1
            async with store.engine.begin() as conn:
                await conn.execute(text("UPDATE rag_chunks SET content_tsv=to_tsvector('simple','中文 检索 search_knowledge') WHERE langchain_metadata->>'index_version'=:b"), dict(b=build))
            hybrid = HybridSearchConfig(tsv_column='content_tsv', tsv_lang='simple', fts_query='检索',
                fusion_function=reciprocal_rank_fusion, primary_top_k=20, secondary_top_k=20,
                fusion_function_parameters={'rrf_k': 60})
            found = await vectors.asimilarity_search('query', k=40, filter={'index_version': build}, hybrid_search_config=hybrid)
            assert len(found) == 1 and found[0].id and found[0].metadata == docs[0].metadata
            assert not await vectors.asimilarity_search('query', filter={'index_version': 'missing'}, hybrid_search_config=hybrid)
            # Full cleanup in a different namespace cannot delete this namespace's rows.
            other = SQLRecordManager('test:' + str(uuid4()), engine=store.engine, async_mode=True)
            await aindex([], other, vectors, cleanup='full', source_id_key='document_id', key_encoder='sha256')
            assert len(await vectors.asimilarity_search('query', filter={'index_version': build})) == 1
        finally:
            if engine:
                await engine.close()
            await store.close()
    asyncio.run(scenario())


def test_retry_deadline_and_rebuild_fences():
    async def scenario():
        store = RagStore(KnowledgeConfig(database_url=SecretStr(URL), knowledge_base_id='test-' + str(uuid4())))
        try:
            await store.init()
            async def submit(force=False):
                return await store.submit(source_uri='file:///retry', digest='hash', snapshot='/unused', metadata={}, force=force)
            job_id = await submit()
            first = await store.claim()
            await store.fail(first, transient=True)
            assert (await store.status(job_id))['state'] == 'queued'
            assert await store.claim() is None  # backoff is enforced
            async with store.engine.begin() as conn:
                await conn.execute(text("UPDATE rag_jobs SET available_at=now()-interval '1 second' WHERE id=:id"), dict(id=job_id))
            second = await store.claim()
            assert second['attempts'] == 2
            replacement_id = await submit(force=True)
            with pytest.raises(LeaseLost):
                await store.publish(second)  # same input version, different desired job
            replacement = await store.claim()
            assert replacement['id'] == replacement_id
            async with store.engine.begin() as conn:
                await conn.execute(text("UPDATE rag_jobs SET deadline=now()-interval '1 second',lease_until=now()-interval '1 second' WHERE id=:id"), dict(id=replacement_id))
            assert await store.claim() is None
            assert (await store.status(replacement_id))['state'] == 'failed'
            assert await store.active() == []
        finally:
            await store.close()
    asyncio.run(scenario())


def test_status_history_empty_publication_and_fenced_diagnostics(tmp_path):
    from ai_assistant.knowledge.diagnostics import write_diagnostic
    async def scenario():
        c = KnowledgeConfig(database_url=SecretStr(URL), knowledge_base_id='test-' + str(uuid4()),
                            diagnostics_dir=tmp_path/'diagnostics', retry_seconds=0)
        store = RagStore(c)
        other = RagStore(c.model_copy(update={'knowledge_base_id': 'another-kb'}))
        try:
            await store.init()
            async def submit(digest, force=False):
                return await store.submit(source_uri='file:///status', digest=digest, snapshot='/unused',
                                          metadata={'source_name': 'status.csv'}, force=force)
            first = await submit('one')
            queued = await store.status(first)
            assert queued['document_id'] and queued['document_version_id'] and queued['source_name'] == 'status.csv'
            assert queued['latest_build_id'] is None and queued['published_build_id'] is None
            assert queued['active_build_id'] is None and not queued['document_deleted']
            assert await other.status(first) is None
            one = await store.claim()
            running = await store.status(first)
            assert running['latest_build_id'] == one['build_id'] and running['error_code'] is None
            with pytest.raises(IngestionError) as caught:
                await store.publish(one)
            assert caught.value.code == 'empty_index'
            await add_chunk(store, one, '\t\n ')
            with pytest.raises(IngestionError):
                await store.publish(one)
            await add_chunk(store, one)
            await store.publish(one)
            published = await store.status(first)
            assert published['published_build_id'] == published['active_build_id'] == one['build_id']
            second = await submit('two')
            two = await store.claim()
            path = write_diagnostic(c, two, 'empty_document', '未提取到有效正文。', 'counts=0')
            await store.fail(two, error='未提取到有效正文。', error_code='empty_document', diagnostic_path=path)
            failed = await store.status(second)
            assert failed['state'] == 'failed' and failed['error_code'] == 'empty_document'
            assert failed['published_build_id'] is None and failed['active_build_id'] == one['build_id']
            assert failed['diagnostic_path'] == path
            third = await submit('three')
            old = await store.claim()
            old_path = write_diagnostic(c, old, 'parser_failed', '失败', 'old attempt detail')
            await store.fail(old, transient=True, error='temporary', error_code='temporary', diagnostic_path=old_path)
            current = await store.claim()
            retried = await store.status(third)
            assert retried['attempts'] == 2 and retried['error'] is None
            assert retried['latest_build_id'] == current['build_id']
            assert retried['error_code'] is None and retried['diagnostic_path'] is None
            await store.fail(old, error='obsolete', error_code='obsolete', diagnostic_path='bad')
            assert (await store.status(third))['error'] is None
            async with store.engine.connect() as conn:
                history = (await conn.execute(text('SELECT error_code,diagnostic_path FROM rag_builds WHERE id=:id'),
                                              dict(id=old['build_id']))).mappings().one()
                assert history['error_code'] == 'temporary' and history['diagnostic_path'] == old_path
            await add_chunk(store, current)
            await store.publish(current)
            assert (await store.status(third))['error'] is None
            historical = await store.status(first)
            assert historical['published_build_id'] == one['build_id']
            assert historical['active_build_id'] == current['build_id']
            await store.delete(one['document_id'])
            deleted = await store.status(first)
            assert deleted['document_deleted'] and deleted['active_build_id'] == current['build_id']
            assert await store.active() == []
        finally:
            await other.close()
            await store.close()
    asyncio.run(scenario())


def test_explicit_idempotent_migration_of_legacy_builds():
    """Create and drop only a new database owned by this test, never existing tables."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.engine import make_url
    from ai_assistant.storage.rag import DDL
    async def scenario():
        name = 'rag_migration_' + uuid4().hex
        admin = create_async_engine(URL, isolation_level='AUTOCOMMIT')
        migrated_url = make_url(URL).set(database=name).render_as_string(hide_password=False)
        store = RagStore(KnowledgeConfig(database_url=SecretStr(migrated_url), knowledge_base_id='legacy'))
        created = False
        try:
            async with admin.connect() as conn:
                await conn.execute(text(f'CREATE DATABASE "{name}"'))
                created = True
            async with store.engine.begin() as conn:
                for statement in DDL[:6]:  # original tables, without new diagnostic columns
                    await conn.execute(text(statement))
                await conn.execute(text("INSERT INTO rag_documents(id,kb,source_uri) VALUES('d','legacy','file:///legacy')"))
                await conn.execute(text("INSERT INTO rag_versions VALUES('v','d','hash','config','/unused','{}')"))
                await conn.execute(text("INSERT INTO rag_jobs(id,version_id,deadline,error,attempts) VALUES('j','v',now()+interval '1 hour','legacy_error',2)"))
                await conn.execute(text("INSERT INTO rag_builds(id,version_id,job_id,token,created_at) VALUES('b1','v','j','t1',now()-interval '1 second'),('b2','v','j','t2',now())"))
            # Database checks report the missing migration without changing schema.
            from ai_assistant.config import AppConfig, ModelConfig
            from ai_assistant.knowledge.cli import execute
            from ai_assistant.knowledge import KnowledgeError
            from types import SimpleNamespace
            app = AppConfig(default_model='test', models={'test': ModelConfig(provider='ollama', model='test')},
                            knowledge=store.config)
            with pytest.raises(KnowledgeError, match='rag init'):
                await execute(app, SimpleNamespace(action='check', database=True, gpu=False))
            # Ordinary status must not migrate the schema.
            from sqlalchemy.exc import ProgrammingError
            with pytest.raises(ProgrammingError):
                await store.status('j')
            await store.init()
            await store.init()
            result = await store.status('j')
            assert result['document_id'] == 'd' and result['latest_build_id'] == 'b2'
            assert result['error'] == 'legacy_error'
            assert result['error_code'] is None and result['diagnostic_path'] is None
            async with store.engine.connect() as conn:
                rows = (await conn.execute(text('SELECT attempt_number FROM rag_builds ORDER BY attempt_number'))).scalars().all()
                assert rows == [1, 2]
        finally:
            await store.close()
            if created:
                async with admin.connect() as conn:
                    await conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
            await admin.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize('phase', ['document', 'chunks', 'index'])
def test_empty_build_pipeline_keeps_previous_evidence(tmp_path, phase):
    from ai_assistant.knowledge.ingestion import run_once
    from langchain_core.documents import Document
    from types import SimpleNamespace
    from unittest.mock import patch, AsyncMock, Mock
    from hashlib import sha256
    async def scenario():
        c = KnowledgeConfig(database_url=SecretStr(URL), knowledge_base_id='test-' + str(uuid4()),
                            diagnostics_dir=tmp_path/'diagnostics')
        store = RagStore(c)
        source = tmp_path/'sample.csv'
        source.write_bytes(b'input')
        try:
            await store.init()
            await store.submit(source_uri=source.as_uri(), digest='original', snapshot=source,
                               metadata={'source_name': 'sample.csv', 'media_type': 'text/csv'})
            original = await store.claim()
            await add_chunk(store, original)
            await store.publish(original)
            job_id = await store.submit(source_uri=source.as_uri(), digest=sha256(b'input').hexdigest(), snapshot=source,
                                        metadata={'source_name': 'sample.csv', 'media_type': 'text/csv'})
            model = Mock(return_value=SimpleNamespace(native=SimpleNamespace(_client=SimpleNamespace(tokenizer=None))))
            splitter = SimpleNamespace(split_documents=lambda docs: [] if phase == 'chunks' else docs)
            index = AsyncMock()
            engine = SimpleNamespace(close=AsyncMock())
            with patch('ai_assistant.knowledge.ingestion.parse_snapshot', AsyncMock(return_value=[
                    Document(page_content=' \t\n') if phase == 'document' else Document(page_content='  正文\n保留 ')])), \
                 patch('ai_assistant.knowledge.ingestion.embedding', model), \
                 patch('langchain_text_splitters.RecursiveCharacterTextSplitter.from_huggingface_tokenizer', return_value=splitter), \
                 patch('ai_assistant.knowledge.ingestion.vector_store', AsyncMock(return_value=(engine, Mock()))), \
                 patch('langchain_classic.indexes.aindex', index):
                assert await run_once(store)
            result = await store.status(job_id)
            assert result['state'] == 'failed' and result['error_code'] == 'empty_' + phase
            assert result['active_build_id'] == original['build_id'] and result['published_build_id'] is None
            assert result['diagnostic_path'] and (await store.active()) == [original['build_id']]
            if phase == 'document':
                model.assert_not_called()
            if phase != 'index':
                index.assert_not_called()
            else:
                index.assert_awaited_once()
                written = index.call_args.args[0]
                assert written[0].page_content == '  正文\n保留 '
                assert written[0].metadata['chunk_index'] == 0
                engine.close.assert_awaited_once()
            async with store.engine.connect() as conn:
                assert (await conn.execute(text("SELECT content FROM rag_chunks WHERE langchain_metadata->>'index_version'=:b"),
                                          dict(b=original['build_id']))).scalar_one() == '证据'
        finally:
            await store.close()
    asyncio.run(scenario())
