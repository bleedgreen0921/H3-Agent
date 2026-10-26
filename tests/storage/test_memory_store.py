"""Native Store indexing and version guards on a disposable PostgreSQL database."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import os
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from ai_assistant.storage.memory import memory_storage, valid_candidate
from ai_assistant.storage.sessions import session_storage
from ai_assistant.storage.integrations import operation_storage
from tests.integrations.test_mcp import config

URL = os.environ.get('SESSION_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not URL, reason='Set a disposable SESSION_TEST_DATABASE_URL')


class FixtureEmbeddings(Embeddings):
    unavailable = False

    def embed_documents(self, texts):
        if self.unavailable:
            raise RuntimeError('fixture embedding outage')
        return [[1.0] + [0.0] * 1023 for _ in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_save_merge_pause_delete_late_jobs_and_degraded_recall(tmp_path, monkeypatch):
    embed = FixtureEmbeddings()
    @asynccontextmanager
    async def fake_embedding(settings):
        yield embed
    monkeypatch.setattr('ai_assistant.knowledge.models.embedding', fake_embedding)

    async def run():
        c = config(tmp_path, sessions={'database_url': URL}, context={}, memory={'enabled': True},
                   knowledge={'database_url': URL.replace('postgresql://', 'postgresql+asyncpg://')}, integrations={'feishu': {
                       'app_id': 'cli_fixture', 'app_secret': 'fixture', 'owner_open_id': str(uuid4())}})
        async with session_storage(c.sessions, initialize=True):
            pass
        async with operation_storage(c.sessions, initialize=True):
            pass
        async with memory_storage(c, initialize=True) as repo:
            policy = await repo.get_policy()
            now = datetime.now(timezone.utc)
            message = HumanMessage('请记住，我喜欢简洁中文。', id='source', additional_kwargs={
                'user_source': '请记住，我喜欢简洁中文。', 'user_timestamp': now.isoformat()})
            raw = dict(category='preference', topic='回答风格', content='我喜欢简洁中文',
                       quote='我喜欢简洁中文。', source_id='source')
            candidate = valid_candidate(raw, [message], policy)
            assert candidate
            await repo.register([candidate, candidate])
            await repo.drain()
            items = await repo.store.asearch(repo.ns, query='偏好', limit=6)
            assert len(items) == 1
            ident = items[0].key
            # 同主题自动提取的冲突内容不能覆盖旧事实。
            conflicting = {**candidate, 'content': '我喜欢详细英文', 'source_id': 'new'}
            await repo.register([conflicting])
            await repo.drain()
            assert (await repo.store.aget(repo.ns, ident)).value['content'] == raw['content']
            # 暂停及删除后，旧排队任务失效；恢复也不回扫旧消息。
            await repo.manage('pause')
            await repo.register([{**candidate, 'source_id': 'late'}])
            await repo.manage('delete', ident)
            await repo.manage('resume')
            await repo.drain()
            assert not await repo.store.asearch(repo.ns, limit=10)
            # 明确重新记住允许越过水位，且必须确认原生 Store 持久化成功。
            assert '持久化成功' in await repo.remember(raw, [message])
            embed.unavailable = True
            await repo.recall('偏好')
            assert '降级' in repo.reference and raw['content'] in repo.reference
            # 存储失败不得声称已记住。
            with pytest.raises(RuntimeError):
                await repo.remember({**raw, 'topic': '第二主题'}, [message])
            embed.unavailable = False
            await repo.manage('clear')
            assert not await repo.store.asearch(repo.ns, limit=10)
    asyncio.run(run())


@pytest.mark.parametrize('fail_after_summary', [False, True])
def test_summary_candidates_commit_only_after_success_and_no_per_turn_extraction(tmp_path, monkeypatch, fail_after_summary):
    import json
    from langchain_core.messages import AIMessage
    from ai_assistant.runtime import assistant_session
    from ai_assistant.integrations.policy import Caller
    from tests.runtime.test_session import ScriptedModel

    @asynccontextmanager
    async def fake_embedding(settings):
        yield FixtureEmbeddings()
    monkeypatch.setattr('ai_assistant.knowledge.models.embedding', fake_embedding)
    monkeypatch.setattr('ai_assistant.runtime.create_builtin_tools', lambda **kwargs: [])
    # 本测试不检索知识库，只复用其显式配置的本地 Embedding。
    monkeypatch.setattr('ai_assistant.knowledge.tools.knowledge_tools', lambda: [])
    main = ScriptedModel(responses=[AIMessage(content='答' * 400)])
    class SummaryModel(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            response = await super()._agenerate(messages, **kwargs)
            if fail_after_summary:
                main.responses = [AIMessage(content='FAIL')]
                main.i = 0
            return response
    summary = SummaryModel(responses=[AIMessage(content='placeholder')])
    @asynccontextmanager
    async def fake_model(config, name, *args, **kwargs):
        yield summary if name == 'summary' else main
    monkeypatch.setattr('ai_assistant.runtime.model_session', fake_model)

    async def run():
        c = config(tmp_path, sessions={'database_url': URL}, context={},
                   memory={'enabled': True}, knowledge={'database_url': URL.replace('postgresql://', 'postgresql+asyncpg://')},
                   integrations={'feishu': {'app_id': 'cli_fixture', 'app_secret': 'fixture', 'owner_open_id': str(uuid4())}})
        from ai_assistant.config import ModelConfig, ContextConfig
        c = c.model_copy(update={'context': ContextConfig(summary_model='summary'), 'models': {
            'test': ModelConfig(provider='ollama', model='fixture', max_input_tokens=6000),
            'summary': ModelConfig(provider='ollama', model='fixture', max_input_tokens=24000)}})
        async with session_storage(c.sessions, initialize=True):
            pass
        async with operation_storage(c.sessions, initialize=True):
            pass
        async with memory_storage(c, initialize=True):
            pass
        # 显式禁止真实知识检索构建。
        from langchain_core.retrievers import BaseRetriever
        class EmptyRetriever(BaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager):
                return []
        async with assistant_session(c, caller=Caller('feishu_private'), retriever=EmptyRetriever()) as session:
            turn = await session.run_turn('我喜欢简洁中文。' + '请求' * 300)
            assert turn.usage.summary_model_calls == 0
            source = turn.messages[0]
            candidate = dict(category='preference', topic='回答风格', content='我喜欢简洁中文',
                             quote='我喜欢简洁中文。', source_id=source.id)
            summary.responses = [AIMessage(content=json.dumps({'summary': '保留目标和约束', 'memories': [candidate]}, ensure_ascii=False))]
            seen = False
            for _ in range(8):
                try:
                    turn = await session.run_turn('继续' * 300)
                except RuntimeError:
                    assert fail_after_summary
                    jobs = await (await session.context.memory.conn.execute(
                        'SELECT * FROM assistant_memory_jobs WHERE owner=%s', (session.context.memory.owner,))).fetchall()
                    assert not jobs
                    return
                if turn.usage.summary_model_calls:
                    assert turn.usage.summary_model_calls == 1
                    seen = True
                    break
            assert seen
            repo = session.context.memory
            jobs = await (await repo.conn.execute('SELECT * FROM assistant_memory_jobs WHERE owner=%s', (repo.owner,))).fetchall()
            assert len(jobs) == 1 and jobs[0]['status'] == 'pending'
            await repo.drain()
            assert len(await repo.store.asearch(repo.ns, limit=10)) == 1
            await repo.manage('clear')
            # 已压缩摘要没有用户来源元数据，不能当新用户原话再次保存。
            assert valid_candidate(candidate, turn.messages, await repo.get_policy()) is None
        # 群聊没有记忆或本地工具，打开时不读取私人 Store。
        async with assistant_session(c, caller=Caller('feishu_group'), retriever=EmptyRetriever()) as group:
            assert group.context.memory is None and group.tools == ()
    asyncio.run(run())
