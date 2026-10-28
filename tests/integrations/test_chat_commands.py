"""飞书指令真实存储回归；模拟身份、模型和投递，无真实账号。"""
import asyncio
import os
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ai_assistant.interfaces.chat_commands import parse_command
from ai_assistant.interfaces.feishu import BotWorker, ProcessResources
from ai_assistant.integrations.mcp import Catalog
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.storage.sessions import session_storage, SessionStorageError
from tests.integrations.test_mcp import config

URL = os.environ.get('SESSION_TEST_DATABASE_URL')
CREATED_ROWS = []
database = pytest.mark.skipif(not URL, reason='Set disposable SESSION_TEST_DATABASE_URL')


@pytest.fixture(autouse=True)
def cleanup_command_rows():
    start = len(CREATED_ROWS)
    yield
    ids = list(set(CREATED_ROWS[start:]))
    if not URL or not ids:
        return

    async def cleanup():
        from ai_assistant.config import SessionsConfig
        settings = SessionsConfig(database_url=URL)
        async with operation_storage(settings) as store:
            rows = await (await store.conn.execute('SELECT DISTINCT binding FROM assistant_inbox WHERE id=ANY(%s)', (ids,))).fetchall()
            for row in rows:
                sessions = await (await store.conn.execute('SELECT id FROM assistant_sessions WHERE binding=%s', (row['binding'],))).fetchall()
                for session in sessions:
                    async with session_storage(settings) as repo:
                        await repo.conn.execute("UPDATE assistant_sessions SET status='interrupted',pending=NULL WHERE id=%s", (session['id'],))
                        await repo.delete(session['id'])
            await store.conn.execute('DELETE FROM assistant_outbox WHERE input_id=ANY(%s)', (ids,))
            await store.conn.execute('DELETE FROM assistant_chat_commands WHERE input_id=ANY(%s)', (ids,))
            await store.conn.execute('DELETE FROM assistant_inbox WHERE id=ANY(%s)', (ids,))
    asyncio.run(cleanup())


@pytest.mark.parametrize('text,expected', [
    ('hello', None), ('/new', ('/new', None)), ('/list', ('/list', 1)),
    ('/list 2', ('/list', 2)), ('/skill skill:abc 总结文章', ('/skill', ('skill:abc', '总结文章'))),
])
def test_parse(text, expected):
    assert parse_command(text) == expected


@pytest.mark.parametrize('text', ['/help', '/status', '/memory', '/plan 请求', '/new x', '/cancel x', '/usage x', '/skills x', '/skill', '/skill id', '/list 0', '/list -1', '/list 1 x', '/delete', '/delete abc'])
def test_invalid_commands(text):
    assert parse_command(text)[0] == 'error'


def configured(tmp_path):
    return config(tmp_path, sessions={'database_url': URL}, integrations={'feishu': {
        'app_id': 'cli_' + uuid4().hex, 'app_secret': 'fixture', 'owner_open_id': 'owner'}})


async def initialize(c):
    async with session_storage(c.sessions, initialize=True):
        pass


async def request(worker, text, binding, *, group=False, row=None):
    from psycopg.types.json import Jsonb
    row = row or {'id': str(uuid4()), 'binding': binding, 'body': {
        'text': text, 'chat_id': 'fixture-chat', 'group': group, 'owner_open_id': 'owner'}}
    CREATED_ROWS.append(row['id'])
    await worker.store.conn.execute('INSERT INTO assistant_inbox(id,binding,body) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING', (row['id'], binding, Jsonb(row['body'])))
    await worker.execute(row)
    replies = await (await worker.store.conn.execute('SELECT body FROM assistant_outbox WHERE input_id=%s ORDER BY part', (row['id'],))).fetchall()
    return row, ''.join(r['body']['text'] for r in replies)


@database
def test_new_list_delete_idempotency_and_isolation(tmp_path, monkeypatch):
    # 管理命令不允许构造模型、MCP 或 runtime。
    monkeypatch.setattr('ai_assistant.interfaces.feishu.assistant_session', lambda *a, **k: pytest.fail('management opened runtime'))

    async def run():
        c = configured(tmp_path)
        await initialize(c)
        async with operation_storage(c.sessions, initialize=True) as store:
            worker = BotWorker(c, store, ProcessResources(Catalog([])), AsyncMock())
            binding, other = str(uuid4()), str(uuid4())
            row, text = await request(worker, '/new', binding)
            first = await worker.binding_session(row)
            assert first in text
            await worker.execute(row, recover=True)
            count = await (await store.conn.execute('SELECT count(*) AS n FROM assistant_sessions WHERE binding=%s', (binding,))).fetchone()
            assert count['n'] == 1
            _, listing = await request(worker, '/list', binding)
            assert first in listing and '●' in listing
            _, listing = await request(worker, '/list', other)
            assert first not in listing
            _, denied = await request(worker, '/delete ' + first, other)
            assert '未找到' in denied
            _, _ = await request(worker, '/new', binding)
            _, text = await request(worker, '/delete ' + first, binding)
            assert '已删除' in text
            current = await worker.binding_session(row)
            deletion, text = await request(worker, '/delete ' + current, binding)
            assert '已删除' in text
            assert await worker.binding_session(row) is None
            await worker.execute(deletion, recover=True)
            assert await worker.binding_session(row) is None
            _, text = await request(worker, '/help', binding)
            assert '未支持' in text
    asyncio.run(run())


@database
def test_pagination_lock_waiting_and_terminal_delete(tmp_path):
    async def run():
        c = configured(tmp_path)
        await initialize(c)
        async with operation_storage(c.sessions, initialize=True) as store:
            worker = BotWorker(c, store, ProcessResources(Catalog([])), AsyncMock())
            binding = str(uuid4())
            for _ in range(11):
                row, _ = await request(worker, '/new', binding)
            _, text = await request(worker, '/list', binding)
            assert text.count('ID：') == 10 and '/list 2' in text
            _, text = await request(worker, '/list 2', binding)
            assert text.count('ID：') == 1
            sid = await worker.binding_session(row)
            async with session_storage(c.sessions) as repo:
                await repo.acquire(sid)
                _, text = await request(worker, '/delete ' + sid, binding)
                assert '正在使用' in text
            await store.conn.execute("UPDATE assistant_sessions SET status='waiting',pending='{}' WHERE id=%s", (sid,))
            _, text = await request(worker, '/delete ' + sid, binding)
            assert '等待续接' in text
            # /new 取消旧等待但不删除会话。
            await request(worker, '/new', binding)
            async with session_storage(c.sessions) as repo:
                await repo.delete(sid)
            assert await worker.binding_session(row) != sid
    asyncio.run(run())


@database
def test_partial_delete_recovery_uses_command_record(tmp_path, monkeypatch):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    original = AsyncPostgresSaver.adelete_thread
    calls = []

    async def fail_once(self, thread_id):
        calls.append(thread_id)
        if len(calls) == 1:
            raise RuntimeError('fixture')
        return await original(self, thread_id)

    async def run():
        c = configured(tmp_path)
        await initialize(c)
        async with operation_storage(c.sessions, initialize=True) as store:
            worker = BotWorker(c, store, ProcessResources(Catalog([])), AsyncMock())
            binding = str(uuid4())
            row, _ = await request(worker, '/new', binding)
            sid = await worker.binding_session(row)
            deletion = {'id': str(uuid4()), 'binding': binding, 'body': {**row['body'], 'text': '/delete ' + sid}}
            monkeypatch.setattr(AsyncPostgresSaver, 'adelete_thread', fail_once)
            with pytest.raises(SessionStorageError):
                await request(worker, '', binding, row=deletion)
            await worker.execute(deletion, recover=True)
            assert await worker.binding_session(row) is None
            assert len(calls) == 2
    asyncio.run(run())


@database
def test_group_skill_command_uses_restricted_runtime(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from langchain_core.messages import AIMessage
    from pydantic import Field
    from ai_assistant.skills.catalog import SkillCatalog
    from ai_assistant.config import SkillsConfig
    from ai_assistant.integrations.config import GroupConfig
    from tests.runtime.test_session import ScriptedModel

    root = tmp_path / '.agents/skills'
    for name in ('summary', 'private'):
        path = root / name
        path.mkdir(parents=True)
        (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: fixture\n---\n{ name.upper() }_RULE')
    catalogue = SkillCatalog([root])
    public, secret = catalogue.resolve('summary').id, catalogue.resolve('private').id

    class RecordingModel(ScriptedModel):
        seen: list = Field(default_factory=list)
        bound_tools: list = Field(default_factory=list)

        def bind_tools(self, tools, **kwargs):
            self.bound_tools = [t.name for t in tools]
            return self

        async def _agenerate(self, messages, **kwargs):
            self.seen.append(messages)
            return await super()._agenerate(messages, **kwargs)

    model = RecordingModel(responses=[AIMessage(content='答复')])

    @asynccontextmanager
    async def models(*args, **kwargs):
        yield model

    monkeypatch.setattr('ai_assistant.runtime.model_session', models)

    async def run():
        c = configured(tmp_path)
        c = c.model_copy(update={'skills': SkillsConfig(),
            'channels': c.channels.model_copy(update={'feishu': c.channels.feishu.model_copy(update={
                'default_group_skill_ids': [public], 'groups': {'fixture-chat': GroupConfig()}})})})
        await initialize(c)
        async with operation_storage(c.sessions, initialize=True) as store:
            worker = BotWorker(c, store, ProcessResources(Catalog([]), catalogue), AsyncMock())
            binding = f'{c.integrations.feishu.app_id}:fixture-chat:owner'
            _, listing = await request(worker, '/skills', binding, group=True)
            assert public in listing and secret not in listing
            _, rejected = await request(worker, '/skill ' + secret + ' 请求', binding, group=True)
            assert '未授权' in rejected and not model.seen
            _, text = await request(worker, '/skill ' + public + ' 请求', binding, group=True)
            assert text == '答复'
            assert 'SUMMARY_RULE' in model.seen[0][0].text
            assert 'PRIVATE_RULE' not in model.seen[0][0].text
            assert 'run_command' not in model.bound_tools
            assert 'read_text' not in model.bound_tools
            assert 'read_skill_resource' in model.bound_tools
            row = await (await store.conn.execute('SELECT id,title,complete,thread_id FROM assistant_sessions WHERE binding=%s', (binding,))).fetchone()
            assert row['title'] == '请求'
            await request(worker, '/delete ' + str(row['id']), binding, group=True)
            remaining = await (await store.conn.execute('SELECT count(*) AS n FROM checkpoints WHERE thread_id=%s', (row['thread_id'],))).fetchone()
            assert remaining['n'] == 0
            await request(worker, '/skill ' + public + ' 新请求', binding, group=True)
            fresh = await (await store.conn.execute('SELECT id,title FROM assistant_sessions WHERE binding=%s', (binding,))).fetchone()
            assert fresh['id'] != row['id'] and fresh['title'] == '新请求'
    try:
        asyncio.run(run())
    finally:
        catalogue.close()


@database
def test_new_reply_failure_rolls_back_creation(tmp_path, monkeypatch):
    from ai_assistant.interfaces import chat_commands
    original = chat_commands.reply

    async def fail(*args):
        raise RuntimeError('fixture reply failure')

    async def run():
        c = configured(tmp_path)
        await initialize(c)
        async with operation_storage(c.sessions, initialize=True) as store:
            worker = BotWorker(c, store, ProcessResources(Catalog([])), AsyncMock())
            binding = str(uuid4())
            row = {'id': str(uuid4()), 'binding': binding, 'body': {
                'text': '/new', 'chat_id': 'fixture-chat', 'group': False, 'owner_open_id': 'owner'}}
            monkeypatch.setattr(chat_commands, 'reply', fail)
            with pytest.raises(RuntimeError):
                await request(worker, '', binding, row=row)
            count = await (await store.conn.execute('SELECT count(*) AS n FROM assistant_sessions WHERE binding=%s', (binding,))).fetchone()
            assert count['n'] == 0
            monkeypatch.setattr(chat_commands, 'reply', original)
            await worker.execute(row, recover=True)
            count = await (await store.conn.execute('SELECT count(*) AS n FROM assistant_sessions WHERE binding=%s', (binding,))).fetchone()
            assert count['n'] == 1
    asyncio.run(run())
