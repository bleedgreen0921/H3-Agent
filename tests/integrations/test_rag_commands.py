"""/rag 独立数据库验收：模拟云解析与模型，验证跨库回执恢复。"""
import asyncio
import os
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ai_assistant.interfaces.feishu import BotWorker, ProcessResources
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.storage.rag import RagStore
from ai_assistant.knowledge.sync import settings
from ai_assistant.knowledge.config import KnowledgeConfig
from tests.integrations.test_chat_commands import configured, initialize, request, cleanup_command_rows

URL = os.environ.get('RAG_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not URL or not os.environ.get('SESSION_TEST_DATABASE_URL'), reason='Set disposable test databases')


def setup_config(tmp_path):
    config = configured(tmp_path)
    return config.model_copy(update={'knowledge': KnowledgeConfig(database_url=URL,
        knowledge_base_id='rag-chat-' + uuid4().hex, sources={'fixture': {'kind': 'local_directory', 'path': tmp_path}})})


def test_management_no_runtime_readonly_and_delivery_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr('ai_assistant.interfaces.feishu.assistant_session', lambda *a, **k: pytest.fail('no model/session'))
    resolve = AsyncMock(return_value=('feishu://fixture/user/docx/token', {'kind': 'docx', 'token': 'token', 'identity': 'user'}))
    monkeypatch.setattr('ai_assistant.knowledge.sync.resolve_submission', resolve)

    async def run():
        c = setup_config(tmp_path)
        await initialize(c)
        knowledge = RagStore(settings(c))
        await knowledge.init()
        try:
            async with operation_storage(c.sessions, initialize=True) as store:
                worker = BotWorker(c, store, ProcessResources(None), AsyncMock())
                binding = 'private-' + uuid4().hex
                _, listing = await request(worker, '/rag sources', binding)
                assert 'fixture' in listing and '未同步' in listing
                assert await knowledge.source_list() == []
                _, listing = await request(worker, '/rag documents 2', binding)
                assert '该页没有记录' in listing
                row, result = await request(worker, '/rag add https://docs.feishu.cn/docx/token', binding)
                assert '同步任务' in result
                await worker.execute(row, recover=True)
                assert resolve.await_count == 1
                replies = await store.commands(row['id'])
                assert replies[0]['status'] == 'completed'
                operation = (await (await store.conn.execute('SELECT result FROM assistant_chat_commands WHERE input_id=%s', (row['id'],))).fetchone())['result']
                assert 'feishu://' not in operation
                # 模拟知识库提交成功但渠道事务丢失：恢复只能读知识库回执。
                await store.conn.execute('DELETE FROM assistant_outbox WHERE input_id=%s', (row['id'],))
                await store.conn.execute('DELETE FROM assistant_chat_commands WHERE input_id=%s', (row['id'],))
                await worker.execute(row, recover=True)
                assert resolve.await_count == 1
                restored = await (await store.conn.execute('SELECT body FROM assistant_outbox WHERE input_id=%s', (row['id'],))).fetchone()
                assert restored['body']['text'] == result
                _, synced = await request(worker, '/rag sync fixture', binding)
                sync_id = synced.split('同步任务：')[1].split('\n')[0]
                _, state = await request(worker, '/rag status ' + sync_id, binding)
                assert sync_id in state and 'queued' in state
                assert str(tmp_path) not in state
                # 无成功业务回执的崩溃恢复不得重新解析和提交。
                lost = dict(row, id=str(uuid4()))
                from tests.integrations.test_chat_commands import CREATED_ROWS
                CREATED_ROWS.append(lost['id'])
                from psycopg.types.json import Jsonb
                await store.conn.execute('INSERT INTO assistant_inbox(id,binding,body) VALUES(%s,%s,%s)', (lost['id'], binding, Jsonb(lost['body'])))
                await worker.execute(lost, recover=True)
                assert resolve.await_count == 1
                response = await (await store.conn.execute('SELECT body FROM assistant_outbox WHERE input_id=%s', (lost['id'],))).fetchone()
                assert '未自动重做' in response['body']['text']
                count = await (await store.conn.execute('SELECT count(*) AS n FROM assistant_sessions WHERE binding=%s', (binding,))).fetchone()
                assert count['n'] == 0
                # 实际投递事务失败后，只补交回执，不能再次调用云解析。
                from ai_assistant.interfaces.chat_commands import reply as original_reply
                async def broken_reply(*args):
                    raise RuntimeError('private delivery failure')
                monkeypatch.setattr('ai_assistant.interfaces.chat_commands.reply', broken_reply)
                failed_row = dict(row, id=str(uuid4()))
                with pytest.raises(RuntimeError):
                    await request(worker, row['body']['text'], binding, row=failed_row)
                assert resolve.await_count == 2
                monkeypatch.setattr('ai_assistant.interfaces.chat_commands.reply', original_reply)
                await worker.recover_rag(failed_row)
                assert resolve.await_count == 2
                command = (await store.commands(failed_row['id']))[0]
                assert command['status'] == 'completed' and '同步任务' in command['receipt']
                from ai_assistant.integrations.cli import execute as diagnostics
                from types import SimpleNamespace
                from contextlib import redirect_stdout
                import io
                import json
                output = io.StringIO()
                with redirect_stdout(output):
                    assert await diagnostics(c, SimpleNamespace(action='operations', operation_action='inspect', operation_id=failed_row['id'])) == 0
                inspected = json.loads(output.getvalue())[0]
                assert inspected['kind'] == 'management_command'
                assert inspected['business_receipt']['result']['sync_id'] in command['receipt']
                assert 'private delivery failure' not in output.getvalue()
        finally:
            await knowledge.close()
    asyncio.run(run())


def test_group_and_disabled_rejected_before_cloud_or_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr('ai_assistant.knowledge.sync.resolve_submission', lambda *a, **k: pytest.fail('cloud forbidden'))
    monkeypatch.setattr('ai_assistant.interfaces.feishu.assistant_session', lambda *a, **k: pytest.fail('runtime forbidden'))
    async def run():
        c = setup_config(tmp_path)
        await initialize(c)
        # 可信入口仍要先验证群授权，然后管理指令明确拒绝。
        from ai_assistant.integrations.config import GroupConfig
        channel = c.channels.feishu.model_copy(update={'groups': {'fixture-chat': GroupConfig()}})
        c = c.model_copy(update={'channels': c.channels.model_copy(update={'feishu': channel})})
        async with operation_storage(c.sessions, initialize=True) as store:
            worker = BotWorker(c, store, ProcessResources(None), AsyncMock())
            _, denied = await request(worker, '/rag add https://docs.feishu.cn/docx/a', 'group-' + uuid4().hex, group=True)
            assert '群聊不支持' in denied
            worker.config = c.model_copy(update={'knowledge': None})
            _, denied = await request(worker, '/rag sync', 'private-' + uuid4().hex)
            assert '未启用' in denied
    asyncio.run(run())


def test_slow_add_does_not_block_usage_or_cancel_and_cleans_up(tmp_path, monkeypatch):
    from psycopg import AsyncConnection, sql
    from psycopg.types.json import Jsonb
    from sqlalchemy.engine import make_url
    from pydantic import SecretStr
    from ai_assistant.config import SessionsConfig

    async def run():
        name = 'rag_cancel_' + uuid4().hex
        admin_url = os.environ['SESSION_TEST_DATABASE_URL']
        admin = await AsyncConnection.connect(admin_url, autocommit=True)
        async with admin:
            await admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
            try:
                c = setup_config(tmp_path)
                # SessionsConfig 要求 URI；从连接参数构建仅用于临时库的 URL。
                session_url = make_url(admin_url).set(database=name).render_as_string(hide_password=False)
                rag_url = make_url(URL).set(database=name).render_as_string(hide_password=False)
                c = c.model_copy(update={'sessions': SessionsConfig(database_url=session_url),
                    'knowledge': c.knowledge.model_copy(update={'database_url': SecretStr(rag_url)})})
                await initialize(c)
                knowledge = RagStore(settings(c))
                await knowledge.init()
                await knowledge.close()
                entered, cleaned = asyncio.Event(), asyncio.Event()
                async def slow(*args, **kwargs):
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cleaned.set()
                monkeypatch.setattr('ai_assistant.knowledge.sync.resolve_submission', slow)
                async with operation_storage(c.sessions, initialize=True) as store:
                    worker = BotWorker(c, store, ProcessResources(None), AsyncMock())
                    worker.send_pending = AsyncMock()
                    async def usage(row):
                        await worker.enqueue_reply(row, 'usage response')
                    worker.show_usage = usage
                    binding = 'cancel-fixture'
                    async def insert(message):
                        ident = str(uuid4())
                        body = {'text': message, 'chat_id': 'fixture-chat', 'group': False, 'owner_open_id': 'owner'}
                        await store.conn.execute('INSERT INTO assistant_inbox(id,binding,body) VALUES(%s,%s,%s)', (ident, binding, Jsonb(body)))
                        return ident
                    task = asyncio.create_task(worker.run())
                    try:
                        await insert('/rag add https://docs.feishu.cn/docx/slow')
                        await asyncio.wait_for(entered.wait(), 5)
                        usage_id = await insert('/usage')
                        cancel_id = await insert('/cancel')
                        await asyncio.wait_for(cleaned.wait(), 5)
                        async with asyncio.timeout(5):
                            while True:
                                rows = await (await store.conn.execute('SELECT body FROM assistant_outbox WHERE input_id=ANY(%s)', ([usage_id, cancel_id],))).fetchall()
                                if len(rows) == 2:
                                    break
                                await asyncio.sleep(.05)
                        assert any('usage response' in r['body']['text'] for r in rows)
                        assert any('已提交的知识库任务继续执行' in r['body']['text'] for r in rows)
                    finally:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            finally:
                await admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))
    asyncio.run(run())
