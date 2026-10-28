"""管理语义与原子回执，仅使用临时库和本地 fixture。"""
import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import text

from ai_assistant.config import AppConfig, ModelConfig
from ai_assistant.interfaces.chat_commands import parse_command
from ai_assistant.knowledge import KnowledgeError, IngestionError, LeaseLost
from ai_assistant.knowledge.config import KnowledgeConfig, KnowledgeSource
from ai_assistant.knowledge.management import apply, receipt, sources, status
from ai_assistant.storage.rag import RagStore

URL = os.environ.get('RAG_TEST_DATABASE_URL')
database = pytest.mark.skipif(not URL, reason='Set disposable RAG_TEST_DATABASE_URL')


@pytest.mark.parametrize('command,value', [
    ('documents', ('documents', 1)), ('documents 2', ('documents', 2)),
    ('sources', ('sources', 1)), ('sources 3', ('sources', 3)),
    ('sync', ('sync', None)), ('sync foo', ('sync', 'foo')),
    ('status id', ('status', 'id')), ('remove id', ('remove', 'id')),
    ('add https://docs.feishu.cn/docx/abc', ('add', 'https://docs.feishu.cn/docx/abc')),
])
def test_parse_rag(command, value):
    assert parse_command('/rag ' + command) == ('/rag', value)


@pytest.mark.parametrize('command', ['', 'documents 0', 'documents 1 x', 'sources -1',
    'status', 'status a b', 'sync a b', 'remove', 'rebuild x', 'add /tmp/a.txt',
    'add http://docs.feishu.cn/docx/abc', 'add https://evil.test/docx/a',
    'add https://docs.feishu.cn/old/a', 'add https://docs.feishu.cn/docx/a identity=bot'])
def test_parse_rag_rejects_invalid(command):
    assert parse_command('/rag ' + command)[0] == 'error'


@database
def test_atomic_receipts_restore_fences_and_readonly_sources(tmp_path, monkeypatch):
    async def run():
        c = KnowledgeConfig(database_url=URL, knowledge_base_id='management-' + uuid4().hex,
                            sources={'configured': KnowledgeSource(kind='local_directory', path=tmp_path)})
        config = AppConfig(default_model='test', models={'test': ModelConfig(provider='ollama', model='test')}, knowledge=c)
        store = RagStore(c)
        try:
            await store.init()
            assert (await sources(config, store))[0]['checked_at'] is None
            assert await store.source_list() == []
            uri = 'file:///fixture-' + uuid4().hex
            descriptor = {'kind': 'local', 'tags': ['保留']}
            first, duplicate = await asyncio.gather(
                apply(store, 'owner', 'msg-add', 'add', uri, submission=(uri, descriptor)),
                apply(store, 'owner', 'msg-add', 'add', uri, submission=(uri, descriptor)))
            assert first == duplicate
            assert await receipt(store, 'owner', 'msg-add') == first
            assert await receipt(store, 'other-owner', 'msg-add') is None
            assert len(await store.source_list()) == 1
            assert (await status(store, first['sync_id']))['kind'] == 'sync'
            job_id = await store.submit(source_uri=uri, digest='a', snapshot='/fixture',
                metadata={'source_name': 'fixture', 'tags': ['保留']})
            doc = (await store.status(job_id))['document_id']
            leased = await store.claim()
            removed = await apply(store, 'owner', 'msg-remove', 'remove', doc)
            assert removed['removed'] == 1
            assert await store.version_for_rebuild(doc) is None
            assert (await store.status(job_id))['state'] == 'cancelled'
            with pytest.raises(LeaseLost):
                await store.publish(leased)
            with pytest.raises(IngestionError):
                await store.submit(source_uri=uri, digest='b', snapshot='/fixture', metadata={}, restore=False)
            assert await store.add_submission(uri, descriptor, restore=False) is None
            assert (await store.document_list())[0]['deleted']
            await apply(store, 'owner', 'msg-restore', 'add', uri, submission=(uri, {'kind': 'local'}))
            assert not (await store.document_list())[0]['deleted']
            assert (await store.members())[0]['descriptor']['tags'] == ['保留']
            # 重放旧 remove 消息不能再次移除已显式恢复的资料。
            assert await apply(store, 'owner', 'msg-remove', 'remove', doc) == removed
            assert not (await store.document_list())[0]['deleted']
            other = RagStore(c.model_copy(update={'knowledge_base_id': 'other-' + uuid4().hex}))
            try:
                for ident in (first['sync_id'], job_id):
                    with pytest.raises(KnowledgeError):
                        await status(other, ident)
            finally:
                await other.close()
            async with store.engine.begin() as conn:
                await conn.execute(text("UPDATE rag_sync_runs SET state='ready',errors='[{\"reason\":\"access_denied\"}]' WHERE id=:id"), dict(id=first['sync_id']))
            assert (await status(store, first['sync_id']))['coverage'] == 'incomplete'
            # 回执写入前失败，整笔恢复/来源/同步操作回滚。
            with pytest.raises(TypeError):
                await apply(store, 'owner', 'bad', 'add', submission=None)
            assert await receipt(store, 'owner', 'bad') is None
            await store.delete(doc)
            async def broken_sync(*args, **kwargs):
                raise RuntimeError('commit fixture')
            monkeypatch.setattr(store, 'request_sync', broken_sync)
            with pytest.raises(RuntimeError):
                await apply(store, 'owner', 'rollback-restore', 'add', uri, submission=(uri, descriptor))
            assert (await store.document_list())[0]['deleted']
            assert await receipt(store, 'owner', 'rollback-restore') is None
            count = len(await store.source_list())
            with pytest.raises(RuntimeError):
                await apply(store, 'owner', 'rollback-new', 'add', submission=(uri + '-new', descriptor))
            assert len(await store.source_list()) == count
        finally:
            await store.close()
    asyncio.run(run())


@database
def test_local_submit_tags_survive_sync_and_automatic_registration(tmp_path):
    from ai_assistant.config import WorkspaceConfig
    from ai_assistant.knowledge.sync import register, settings, synchronize
    async def run():
        c = AppConfig(default_model='test', models={'test': ModelConfig(provider='ollama', model='test')},
            workspace=WorkspaceConfig(root=str(tmp_path)), knowledge=KnowledgeConfig(database_url=URL,
                knowledge_base_id='tags-' + uuid4().hex, materials_dir=tmp_path / 'materials'))
        (tmp_path / 'document.txt').write_text('中文资料', encoding='utf-8')
        store = RagStore(settings(c))
        try:
            await store.init()
            result = await register(c, 'document.txt', tags=['personal'])
            run = await store.claim_sync()
            assert run['id'] == result['sync_id']
            await synchronize(c, store, run)
            rows = await store.document_list()
            assert rows[0]['metadata']['tags'] == ['personal']
            await store.delete(rows[0]['id'])
            before = len(await store.source_list())
            ignored = await register(c, 'document.txt', restore=False)
            assert ignored['status'] == 'removed'
            assert len(await store.source_list()) == before
            # 同步在移除前开始、提交时才发现移除，也必须拒绝发布新版本。
            automatic_run = await store.request_sync({'source_id': result['source_id']})
            claimed = await store.claim_sync()
            assert claimed['id'] == automatic_run
            await synchronize(c, store, claimed)
            assert (await store.document_list())[0]['deleted']
            restored = await register(c, 'document.txt')
            assert restored['sync_id'] != result['sync_id']
            claimed = await store.claim_sync()
            await synchronize(c, store, claimed)
            row = (await store.document_list())[0]
            assert not row['deleted'] and row['metadata']['tags'] == ['personal']
        finally:
            await store.close()
    asyncio.run(run())


def test_legacy_source_registration_cannot_restore_concurrently_removed_document(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from ai_assistant.config import WorkspaceConfig
    from ai_assistant.knowledge.sync import configure
    config = AppConfig(default_model='test', models={'test': ModelConfig(provider='ollama', model='test')},
        workspace=WorkspaceConfig(root=str(tmp_path)),
        knowledge=KnowledgeConfig(database_url='postgresql+asyncpg://unused/unused'))
    # 列表快照与真正登记之间可能发生移除；迁移登记也必须在行锁内检查。
    store = SimpleNamespace(configure_sources=AsyncMock(), members=AsyncMock(return_value=[]),
        document_list=AsyncMock(return_value=[{'deleted': False, 'source_uri': (tmp_path / 'old.txt').as_uri()}]),
        add_submission=AsyncMock())
    asyncio.run(configure(config, store))
    assert store.add_submission.await_args.kwargs == {'restore': False}


@database
def test_rag_check_default_checks_both_and_flags_narrow_scope(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from ai_assistant.knowledge.cli import execute
    calls = []
    @asynccontextmanager
    async def embedding(config):
        calls.append('embedding')
        async def query(value):
            return [0.] * 1024
        yield SimpleNamespace(aembed_query=query)
    def reranker(config):
        calls.append('reranker')
        return SimpleNamespace(score=lambda pairs: [1.])
    monkeypatch.setattr('ai_assistant.knowledge.models.embedding', embedding)
    monkeypatch.setattr('ai_assistant.knowledge.models.reranker', reranker)
    async def run():
        c = AppConfig(default_model='test', models={'test': ModelConfig(provider='ollama', model='test')},
            knowledge=KnowledgeConfig(database_url=URL, knowledge_base_id='check-' + uuid4().hex))
        store = RagStore(c.knowledge)
        await store.init()
        await store.close()
        assert await execute(c, SimpleNamespace(action='check', database=False, services=False)) == 0
        assert calls == ['embedding', 'reranker']
        calls.clear()
        assert await execute(c, SimpleNamespace(action='check', database=True, services=False)) == 0
        assert calls == []
        monkeypatch.setattr('ai_assistant.storage.rag.RagStore', lambda c: pytest.fail('services-only must not connect database'))
        assert await execute(c, SimpleNamespace(action='check', database=False, services=True)) == 0
        assert calls == ['embedding', 'reranker']
    asyncio.run(run())
