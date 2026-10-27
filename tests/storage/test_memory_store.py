"""独立 PostgreSQL、模拟 embedding；不使用账号或付费模型。"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
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


@asynccontextmanager
async def repository(tmp_path, monkeypatch):
    embed = FixtureEmbeddings()
    @asynccontextmanager
    async def fake_embedding(settings):
        yield embed
    monkeypatch.setattr('ai_assistant.knowledge.models.embedding', fake_embedding)
    c = config(tmp_path, sessions={'database_url': URL}, context={}, memory={'enabled': True},
               knowledge={'database_url': URL.replace('postgresql://', 'postgresql+asyncpg://')},
               integrations={'feishu': {'app_id': 'cli_fixture', 'app_secret': 'fixture', 'owner_open_id': str(uuid4())}})
    async with session_storage(c.sessions, initialize=True), operation_storage(c.sessions, initialize=True):
        pass
    async with memory_storage(c, initialize=True) as repo:
        repo.session_id = 'session-one'
        yield repo, embed


def message(content, ident=None):
    return HumanMessage(content, id=ident or str(uuid4()), additional_kwargs={
        'user_source': content, 'user_timestamp': datetime.now(timezone.utc).isoformat()})


async def write(repo, text, **kwargs):
    quote = (kwargs.get('project_name', '') + '：' if kwargs.get('project_name') else '') + text
    source = message(quote)
    await repo.search(source_id=source.id)
    cardinality = kwargs.pop('cardinality', 'single')
    candidate = dict(category='preference', topic='回答语言', values=[text], cardinality=cardinality, quote=quote, source_id=source.id, **kwargs)
    return json.loads(await repo.remember(candidate, [source]))


def test_revisions_identity_history_idempotency_and_delete(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            first = await write(repo, '中文')
            assert first['status'] == 'saved'
            ident = first['memory_id']
            old = (await repo.get(repo.ns, ident)).value
            second = await write(repo, '英文', action='replace', memory_id=ident,
                                 expected_revision=1, attribute_id=old['attribute_id'])
            assert second['memory_id'] == ident and second['revision'] == 2
            repeated = await write(repo, '英文', action='replace', memory_id=ident,
                                   expected_revision=2, attribute_id=old['attribute_id'])
            assert repeated['status'] == 'unchanged' and repeated['revision'] == 2
            assert (await write(repo, '日文', action='replace', memory_id=ident,
                                expected_revision=1, attribute_id=old['attribute_id']))['status'] == 'conflict'
            history = json.loads(await repo.manage('history', ident))
            assert [v['values'] for v in history['history']] == [['中文']]
            assert history['current']['values'] == ['英文']
            # 同义主题复用属性 ID，不得创建第二条事实。
            source = message('法文')
            await repo.search(source_id=source.id)
            raw = dict(category='preference', topic='沟通语言', values=['法文'], cardinality='single', quote='法文', source_id=source.id,
                       attribute_id=old['attribute_id'])
            assert json.loads(await repo.remember(raw, [source]))['status'] == 'needs_clarification'
            # 重入同一个请求返回原回执，不能再次递增。
            source = message('英文')
            raw = dict(category='preference', topic='回答语言', values=['英文'], cardinality='single', quote='英文', source_id=source.id,
                       attribute_id=old['attribute_id'], action='replace', memory_id=ident, expected_revision=2)
            await repo.search(source_id=source.id)
            assert await repo.remember(raw, [source]) == await repo.remember(raw, [source])
            await repo.manage('delete', ident)
            assert not (await repo.get(repo.ns, ident))
            assert json.loads(await repo.manage('history', ident))['history'] == []
            await repo.search(source_id=source.id)
            assert json.loads(await repo.remember({**raw, 'action': 'add', 'memory_id': None,
                                                  'expected_revision': None}, [source]))['status'] == 'rejected'
            assert '英文' not in repo.reference
    asyncio.run(run())


def test_atomic_rollback_and_unrelated_pending_candidate(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            saved = await write(repo, '中文')
            ident = saved['memory_id']
            old = (await repo.get(repo.ns, ident)).value
            # 在历史已经写入后令当前记录写入失败，整个事务必须回滚。
            original = repo.put
            async def fail_current(ns, key, value, **kwargs):
                if ns == repo.ns:
                    raise RuntimeError('fixture write failure')
                await original(ns, key, value, **kwargs)
            monkeypatch.setattr(repo, 'put', fail_current)
            with pytest.raises(RuntimeError):
                await write(repo, '英文', action='replace', memory_id=ident,
                            expected_revision=1, attribute_id=old['attribute_id'])
            monkeypatch.setattr(repo, 'put', original)
            assert (await repo.get(repo.ns, ident)).value == old
            assert not await repo.all(repo.area('history'))
            source = message('我住在月球基地')
            candidate = valid_candidate(dict(category='personal_fact', topic='住所', values=[source.content], cardinality='single',
                                            quote=source.content, source_id=source.id), [source], await repo.get_policy())
            await repo.register([candidate])
            await repo.manage('delete', ident)
            await repo.drain()
            assert len(await repo.all(repo.ns)) == 1
            assert (await repo.all(repo.ns))[0].value['topic'] == '住所'
            # 清空屏障阻止旧候选及旧格式 checkpoint 重新导入。
            await repo.manage('clear')
            await repo.register([candidate, {**candidate, 'format': 1}])
            await repo.drain()
            assert not await repo.all(repo.ns)
    asyncio.run(run())


def test_scopes_multiple_values_and_effective_dates(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            global_value = await write(repo, '中文')
            global_record = (await repo.get(repo.ns, global_value['memory_id'])).value
            project = await write(repo, '英文', scope='project', project_name='虚构灯塔',
                                  attribute_id=global_record['attribute_id'])
            project_record = (await repo.get(repo.ns, project['memory_id'])).value
            repo.project_id = project_record['project_id']
            await repo.recall('语言')
            assert '英文' in repo.reference and '"values": ["中文"]' not in repo.reference
            exception = await write(repo, '日文', scope='session', project_id=repo.project_id,
                                    attribute_id=global_record['attribute_id'],
                                    valid_until=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
            assert exception['status'] == 'saved'
            await repo.recall('语言')
            assert '"values": ["日文"]' in repo.reference
            repo.session_id = 'session-two'
            repo.project_id = None
            await repo.recall('语言')
            assert '"values": ["中文"]' in repo.reference and '"values": ["日文"]' not in repo.reference
            # 未来永久替换生效前保留旧值。
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
            assert (await write(repo, '法文', action='replace', memory_id=global_record['id'],
                                expected_revision=1, attribute_id=global_record['attribute_id'],
                                valid_from=tomorrow))['status'] == 'saved'
            await repo.recall('语言')
            assert '"values": ["中文"]' in repo.reference
            source = message('Java')
            await repo.search(source_id=source.id)
            raw = dict(category='personal_fact', topic='会的语言', values=['Java'], cardinality='multiple', quote='Java', source_id=source.id)
            first = json.loads(await repo.remember(raw, [source]))
            value = (await repo.get(repo.ns, first['memory_id'])).value
            source = message('Python')
            await repo.search(source_id=source.id)
            raw.update(values=['Python'], quote='Python', source_id=source.id, action='append',
                       memory_id=value['id'], expected_revision=1, attribute_id=value['attribute_id'])
            assert json.loads(await repo.remember(raw, [source]))['status'] == 'saved'
            assert (await repo.get(repo.ns, value['id'])).value['values'] == ['Java', 'Python']
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
            candidate = dict(category='preference', topic='回答风格', values=['简洁中文'], cardinality='single',
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


@pytest.mark.parametrize('stage', ['history', 'current', 'receipts'])
@pytest.mark.parametrize('cancel', [False, True])
def test_every_store_write_boundary_rolls_back(tmp_path, monkeypatch, stage, cancel):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            saved = await write(repo, '中文')
            old = (await repo.get(repo.ns, saved['memory_id'])).value
            baseline = {name: [i.value for i in await repo.all(ns)] for name, ns in (
                ('current', repo.ns), ('history', repo.area('history')), ('receipts', repo.area('receipts')))}
            original = repo.put
            async def broken(ns, key, value, **kwargs):
                await original(ns, key, value, **kwargs)
                target = repo.ns if stage == 'current' else repo.area(stage)
                if ns == target:
                    if cancel:
                        raise asyncio.CancelledError()
                    raise RuntimeError('fixture boundary failure')
            monkeypatch.setattr(repo, 'put', broken)
            with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
                await write(repo, '英文', memory_id=old['id'], expected_revision=1,
                            action='replace', attribute_id=old['attribute_id'])
            monkeypatch.setattr(repo, 'put', original)
            for name, ns in (('current', repo.ns), ('history', repo.area('history')), ('receipts', repo.area('receipts'))):
                assert [i.value for i in await repo.all(ns)] == baseline[name]
            # 新连接重新打开，回滚后的状态仍一致。
            async with memory_storage(repo.config) as reopened:
                assert (await reopened.get(reopened.ns, old['id'])).value == old
    asyncio.run(run())


def test_concurrent_revision_lock_across_connections(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            saved = await write(repo, '中文')
            old = (await repo.get(repo.ns, saved['memory_id'])).value
            async with memory_storage(repo.config) as other:
                values = await asyncio.gather(*(write(target, text, action='replace', memory_id=old['id'],
                    expected_revision=1, attribute_id=old['attribute_id']) for target, text in ((repo, '英文'), (other, '日文'))))
                assert sorted(v['status'] for v in values) == ['conflict', 'saved']
                assert (await repo.get(repo.ns, old['id'])).value['revision'] == 2
                assert len(await repo.all(repo.area('history'))) == 1
    asyncio.run(run())


def test_expiration_fallback_and_no_permanent_resurrection(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            global_saved = await write(repo, '中文')
            global_value = (await repo.get(repo.ns, global_saved['memory_id'])).value
            project = await write(repo, '英文', scope='project', project_name='灯塔',
                                  attribute_id=global_value['attribute_id'],
                                  valid_until=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
            assert project['status'] == 'saved'
            repo.project_id = (await repo.get(repo.ns, project['memory_id'])).value['project_id']
            await repo.recall('语言')
            assert '"values": ["中文"]' in repo.reference and '"values": ["英文"]' not in repo.reference
            await write(repo, '法文', action='replace', memory_id=global_value['id'],
                        attribute_id=global_value['attribute_id'], expected_revision=1,
                        valid_until=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
            await repo.recall('语言')
            assert '"values": ["中文"]' not in repo.reference
    asyncio.run(run())


def test_reset_is_owner_only_and_preserves_business_data(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            await write(repo, '中文')
            another = (*repo.ns[:-1], 'another-fixture-owner')
            await repo.put(another, 'preserved', {'content': '其他用户的数据'})
            sessions_before = await (await repo.conn.execute('SELECT count(*) AS n FROM assistant_sessions')).fetchone()
            response = json.loads(await repo.initialize(reset_existing=True))
            assert response['removed']['personal_memory'] == 1
            assert not await repo.all(repo.ns)
            assert (await repo.get(another, 'preserved')).value['content'] == '其他用户的数据'
            assert await (await repo.conn.execute('SELECT count(*) AS n FROM assistant_sessions')).fetchone() == sessions_before
            assert (await repo.get_policy())['format'] == 3
    asyncio.run(run())


def test_project_aliases_and_historical_fact_do_not_replace_present(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            project = await repo.directory('projects', '灯塔', evidence=[{'source_id': 'fixture', 'quote': '灯塔', 'timestamp': datetime.now(timezone.utc).isoformat()}])
            source = message('灯塔也叫 Lighthouse')
            assert (await repo.select_project(project['id'], alias='Lighthouse', expected_revision=1, source=source))['status'] == 'saved'
            assert (await repo.directory('projects', 'Lighthouse'))['id'] == project['id']
            other = await repo.directory('projects', '海鸥', evidence=[{'source_id': 'fixture', 'quote': '海鸥', 'timestamp': datetime.now(timezone.utc).isoformat()}])
            assert (await repo.select_project(other['id'], alias='Lighthouse', expected_revision=1, source=source))['status'] == 'needs_clarification'
            current = await write(repo, '中文')
            value = (await repo.get(repo.ns, current['memory_id'])).value
            historical = await write(repo, '英文', attribute_id=value['attribute_id'],
                                     valid_from='2020-01-01T00:00:00+08:00', valid_until='2021-01-01T00:00:00+08:00')
            assert historical['status'] == 'saved'
            await repo.recall('语言')
            assert '"values": ["中文"]' in repo.reference and '"values": ["英文"]' not in repo.reference
    asyncio.run(run())


def test_old_format_requires_explicit_reset_and_budget_skips_large_record(tmp_path, monkeypatch):
    from ai_assistant.config import ConfigError
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            big = await write(repo, '中文')
            value = (await repo.get(repo.ns, big['memory_id'])).value
            value['content'] = 'oversized-fixture-value'
            value['evidence'] = [{**value['evidence'][0], 'quote': '证据' * 600}] * 5
            await repo.put(repo.ns, value['id'], value)
            source = message('喜欢简洁')
            await repo.search(source_id=source.id)
            raw = dict(category='preference', topic='回复长度', values=[source.content], cardinality='single', quote=source.content, source_id=source.id)
            assert json.loads(await repo.remember(raw, [source]))['status'] == 'saved'
            await repo.recall('偏好')
            assert '喜欢简洁' in repo.reference and 'oversized-fixture-value' not in repo.reference
            await repo.conn.execute('UPDATE assistant_memory_policy SET format=1 WHERE owner=%s', (repo.owner,))
            with pytest.raises(ConfigError, match='初始化'):
                async with memory_storage(repo.config):
                    pass
            assert len(await repo.all(repo.ns)) == 2
            await repo.initialize(reset_existing=True)
            async with memory_storage(repo.config) as reopened:
                assert not await reopened.all(reopened.ns)
    asyncio.run(run())


def test_canonical_values_no_phantom_history_and_attribute_cardinality(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            first = await write(repo, 'Java')
            record = (await repo.get(repo.ns, first['memory_id'])).value
            source = message('主要语言现在是 Python。')
            raw = dict(category='project', topic='主要后端语言', values=['Python'], cardinality='single',
                       quote=source.content, source_id=source.id, action='replace',
                       memory_id=record['id'], expected_revision=1, attribute_id=record['attribute_id'])
            await repo.search(source_id=source.id)
            assert json.loads(await repo.remember(raw, [source]))['revision'] == 2
            current = (await repo.get(repo.ns, record['id'])).value
            source = message('主力服务端语言是Python')
            raw.update(quote=source.content, source_id=source.id, expected_revision=2, topic='服务端主力语言')
            await repo.search(source_id=source.id)
            repeated = json.loads(await repo.remember(raw, [source]))
            assert repeated['status'] == 'unchanged' and repeated['revision'] == 2
            assert (await repo.get(repo.ns, record['id'])).value == current
            assert len(await repo.all(repo.area('history'))) == 1
            raw.update(cardinality='multiple', action='append')
            assert json.loads(await repo.remember(raw, [source]))['reason'] == 'cardinality_mismatch'
    asyncio.run(run())


def test_set_add_remove_replace_and_idempotency(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            initial = await write(repo, 'Java', cardinality='multiple')
            value = (await repo.get(repo.ns, initial['memory_id'])).value
            async def change(text, action, revision, values, **extra):
                source = message(text)
                await repo.search(source_id=source.id)
                raw = dict(category='personal_fact', topic='会的编程语言', cardinality='multiple', values=values,
                           quote=text, source_id=source.id, action=action, expected_revision=revision,
                           memory_id=value['id'], attribute_id=value['attribute_id'], **extra)
                return json.loads(await repo.remember(raw, [source]))
            assert (await change('还会 Python', 'append', 1, ['Python', 'Java']))['revision'] == 2
            assert (await change('也会 Python', 'append', 2, ['Python']))['status'] == 'unchanged'
            assert (await change('不会 Rust', 'remove', 2, ['Rust']))['status'] == 'unchanged'
            assert (await change('不再会 Java', 'remove', 2, ['Java']))['revision'] == 3
            assert (await repo.get(repo.ns, value['id'])).value['values'] == ['Python']
            assert (await change('全部改为 Go', 'replace', 3, ['Go'], replace_all=True))['revision'] == 4
            assert (await change('不再会 Go', 'remove', 4, ['Go']))['revision'] == 5
            current = (await repo.get(repo.ns, value['id'])).value
            assert current['values'] == [] and current['status'] == 'inactive'
    asyncio.run(run())


def test_search_filters_paging_and_deleted_read_contract(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            global_saved = await write(repo, '中文')
            global_record = (await repo.get(repo.ns, global_saved['memory_id'])).value
            saved = await write(repo, 'Python', scope='project', project_name='记忆验收灯塔项目')
            value = (await repo.get(repo.ns, saved['memory_id'])).value
            project_id = value['project_id']
            before_project = repo.project_id
            assert json.loads(await repo.search('无匹配xyz'))['status'] == 'not_found'
            assert json.loads(await repo.search('Python'))['total'] == 1
            assert json.loads(await repo.search(project_id=project_id, include_global=False))['total'] == 1
            assert repo.project_id == before_project
            assert json.loads(await repo.search(project_id=project_id))['total'] == 2
            only_global = json.loads(await repo.search(scope='global', attribute_id=global_record['attribute_id']))
            assert only_global['items'][0]['id'] == global_saved['memory_id']
            detail = json.loads(await repo.manage('detail', value['id']))
            assert detail['status'] == 'found' and detail['history_count'] == 0 and 'history' not in detail
            await repo.manage('delete', value['id'])
            await repo.manage('delete', value['id'])
            deleted = json.loads(await repo.manage('detail', value['id']))
            assert deleted['status'] == 'deleted' and deleted['current'] is None
            found = json.loads(await repo.search('灯塔', project_id=project_id, include_global=False))
            assert found['status'] == 'deleted' and found['deleted'][0]['memory_id'] == value['id']
            assert 'Python' not in json.dumps(found, ensure_ascii=False)
            # 目录和事实都独立分页，恰好20条时不虚报下一页。
            await repo.manage('clear')
            for index in range(23):
                await repo.put(repo.ns, str(index), {**global_record, 'id': str(index),
                    'attribute_id': str(index), 'topic': '分页测试', 'content': str(index)})
                await repo.put(repo.area('attributes'), str(index), {'id': str(index), 'name': '分页测试',
                    'aliases': [], 'revision': 1, 'cardinality': 'single'})
            page = json.loads(await repo.search('分页'))
            assert len(page['items']) == 20 and page['next_offset'] == 20
            assert len(page['attributes']) == 20 and page['attributes_next_offset'] == 20
            last = json.loads(await repo.search('分页', offset=20, attributes_offset=20))
            assert len(last['items']) == 3 and last['next_offset'] is None
            assert len(last['attributes']) == 3 and last['attributes_next_offset'] is None
            assert not set(v['id'] for v in page['items']).intersection(v['id'] for v in last['items'])
    asyncio.run(run())


def test_project_full_name_and_alias_evidence_are_revision_checked(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            source = message('记忆验收灯塔项目的主要后端语言是 Java')
            await repo.search(source_id=source.id)
            raw = dict(category='project', topic='主要后端语言', values=['Java'], cardinality='single',
                       source_id=source.id, quote=source.content, scope='project', project_name='灯塔项目')
            assert json.loads(await repo.remember(raw, [source]))['status'] == 'needs_clarification'
            raw['project_name'] = '记忆验收灯塔项目'
            saved = json.loads(await repo.remember(raw, [source]))
            value = (await repo.get(repo.ns, saved['memory_id'])).value
            projects = await repo.all(repo.area('projects'))
            assert len(projects) == 1 and projects[0].value['name'] == '记忆验收灯塔项目'
            alias_source = message('灯塔项目现在用 Python')
            assert (await repo.select_project(value['project_id'], alias='灯塔项目',
                expected_revision=1, source=alias_source))['status'] == 'saved'
            project = (await repo.get(repo.area('projects'), value['project_id'])).value
            assert project['revision'] == 2 and project['aliases'] == ['灯塔项目']
            assert (await repo.get(repo.ns, value['id'])).value['revision'] == 1
            assert (await repo.select_project(value['project_id'], alias='灯塔',
                expected_revision=1, source=message('灯塔')) )['status'] == 'conflict'
            assert (await repo.select_project(value['project_id'], alias='假名字',
                expected_revision=2, source=alias_source))['status'] == 'rejected'
    asyncio.run(run())


@pytest.mark.parametrize('short_name', ['灯塔项目', '灯塔'])
def test_short_quote_cannot_bypass_full_project_name(tmp_path, monkeypatch, short_name):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            source = message('记忆验收灯塔项目的主要后端语言是Java，请保存。')
            await repo.search(source_id=source.id)
            raw = dict(category='project', topic='主要后端语言', values=['Java'], cardinality='single',
                       quote='灯塔项目的主要后端语言是Java', source_id=source.id, scope='project', project_name=short_name)
            response = json.loads(await repo.remember(raw, [source]))
            assert response['reason'] == 'project_name_evidence'
            assert response['project_name_candidates'] == ['记忆验收灯塔项目']
            assert not await repo.all(repo.area('projects'))
            raw['project_name'] = '记忆验收灯塔项目'
            saved = json.loads(await repo.remember(raw, [source]))
            assert saved['status'] == 'saved'
            entry = (await repo.all(repo.area('projects')))[0].value
            assert entry['name'] == '记忆验收灯塔项目'
            assert entry['evidence'][0]['quote'] == '记忆验收灯塔项目'
            assert 'Java' not in json.dumps(entry)
    asyncio.run(run())


def test_project_name_without_chinese_suffix_is_supported(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            source = message('我的项目叫 Apollo，主要后端语言是 Python。')
            await repo.search(source_id=source.id)
            raw = dict(category='project', topic='主要后端语言', values=['Python'], cardinality='single',
                       quote=source.content, source_id=source.id, scope='project', project_name='Apollo')
            assert json.loads(await repo.remember(raw, [source]))['status'] == 'saved'
    asyncio.run(run())


def test_effective_search_projects_future_history_and_expired_values(tmp_path, monkeypatch):
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            saved = await write(repo, 'Java')
            record = (await repo.get(repo.ns, saved['memory_id'])).value
            future = datetime.now(timezone.utc) + timedelta(days=2)
            await write(repo, 'Python', action='replace', attribute_id=record['attribute_id'],
                        memory_id=record['id'], expected_revision=1, valid_from=future.isoformat())
            now = json.loads(await repo.search(attribute_id=record['attribute_id'], effective_at=datetime.now(timezone.utc)))
            assert now['items'][0]['values'] == ['Java']
            later = json.loads(await repo.search(attribute_id=record['attribute_id'], effective_at=future + timedelta(days=1)))
            assert later['items'][0]['values'] == ['Python']
    asyncio.run(run())


def test_agent_tool_schema_binds_current_source_for_natural_set_changes(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from ai_assistant.agents import create_assistant_agent
    from ai_assistant.config import AgentSettings
    from ai_assistant.runtime import AssistantSession
    from ai_assistant.tools.context import ToolContext
    from ai_assistant.tools.memory import search_memories, remember_memory
    from ai_assistant.workspace.files import ReadOnlyWorkspace
    from tests.runtime.test_session import ScriptedModel
    async def run():
        async with repository(tmp_path, monkeypatch) as (repo, embed):
            source_text = '请记住，我会 Java。'
            add = dict(category='personal_fact', topic='掌握的编程语言', cardinality='multiple',
                       values=['Java'], quote=source_text)
            model = ScriptedModel(responses=[
                AIMessage(content='', tool_calls=[{'name': 'search_memories', 'args': {}, 'id': 'search'}]),
                AIMessage(content='', tool_calls=[{'name': 'remember_memory', 'args': {'change': add}, 'id': 'save'}]),
                AIMessage(content='<memory_result>已保存。</memory_result>')])
            settings = AgentSettings()
            agent = create_assistant_agent(model, [search_memories, remember_memory], checkpointer=InMemorySaver(), settings=settings)
            session = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path), memory=repo), settings)
            first = await session.run_turn(source_text)
            value = (await repo.all(repo.ns))[0].value
            first_id = value['evidence'][0]['source_id']
            assert value['values'] == ['Java'] and value['cardinality'] == 'multiple'
            request = '我还会 Python，Java 也仍然会。'
            change = {**add, 'values': ['Python'], 'quote': request, 'action': 'append',
                      'attribute_id': value['attribute_id'], 'memory_id': value['id'], 'expected_revision': 1}
            model.responses = [
                AIMessage(content='', tool_calls=[{'name': 'search_memories', 'args': {}, 'id': 'search2'}]),
                AIMessage(content='', tool_calls=[{'name': 'remember_memory', 'args': {'change': change}, 'id': 'save2'}]),
                AIMessage(content='<memory_result>已追加。</memory_result>')]
            model.i = 0
            second = await session.run_turn(request)
            current = (await repo.get(repo.ns, value['id'])).value
            assert current['values'] == ['Java', 'Python'] and current['revision'] == 2
            assert current['evidence'][0]['source_id'] != first_id
            assert '已保存' in second.messages[-1].text and '<memory_result>' not in second.messages[-1].text
    asyncio.run(run())
