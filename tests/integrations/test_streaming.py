"""Durable display operations use a disposable database and mocked Feishu."""
import asyncio
import json
import os
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from ai_assistant.config import SessionsConfig
from ai_assistant.integrations.mcp import Catalog
from ai_assistant.integrations.feishu_files import FeishuRejected
from ai_assistant.interfaces.feishu import BotWorker, ProcessResources
from ai_assistant.interfaces.streaming import ReplyStream, card
from ai_assistant.storage.sessions import session_storage
from ai_assistant.storage.integrations import operation_storage
from tests.integrations.test_mcp import config

URL = os.environ.get('SESSION_TEST_DATABASE_URL')


def test_card_limits_serialized_utf8_without_silent_final_loss():
    for text in ('中文' * 30000, '"\\\n' * 20000):
        payload, fits = card(text, 'complete')
        assert not fits
        assert len(json.dumps({'content': json.dumps(payload, ensure_ascii=False)}, ensure_ascii=False).encode()) <= 24576
        assert '预览' in str(payload)
    assert card('完整回答', 'complete')[1]


def test_card_uses_native_markdown_for_sql_answer_and_stream_updates():
    text = ("## 聚合函数\n\n**COUNT** 与 `SUM`\n\n"
            "| 函数 | 说明 |\n| --- | --- |\n| COUNT | 计数 |\n\n"
            "```sql\nSELECT COUNT(*) FROM emp;\n```\n\n"
            "[参考](https://example.com/sql?a=1&b=2)")
    for state in ('delta', 'complete', 'failed'):
        payload, fits = card(text, state)
        assert fits
        assert payload['schema'] == '2.0'
        assert payload['config']['update_multi'] is True
        assert payload['body']['elements'] == [{'tag': 'markdown', 'content': text}]
        assert payload['header']['title']['tag'] == 'plain_text'
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert card('', 'stage')[0]['body']['elements'][0]['content'] == '正在处理…'


@pytest.mark.skipif(not URL, reason='Set a disposable SESSION_TEST_DATABASE_URL')
@pytest.mark.parametrize('outcome', ['success', 'unknown', 'rejected', 'throttle', 'update_rejected', 'missing_id'])
@pytest.mark.parametrize('terminal_state', ['complete', 'partial', 'failed', 'unknown', 'cancelled', 'needs_clarification'])
def test_card_creation_receipts_final_priority_and_resume(tmp_path, outcome, terminal_state):
    async def run():
        from psycopg.types.json import Jsonb
        c = config(tmp_path, sessions={'database_url': URL}, channels={'feishu': {'streaming': True}})
        async with session_storage(c.sessions, initialize=True):
            pass
        async with operation_storage(c.sessions, initialize=True) as store:
            row = dict(id=str(uuid4()), binding=str(uuid4()), body={'chat_id': 'fixture', 'text': 'input'})
            await store.conn.execute('INSERT INTO assistant_inbox(id,binding,body) VALUES(%s,%s,%s)',
                                     (row['id'], row['binding'], Jsonb(row['body'])))
            api = AsyncMock()
            api.send_card.return_value = {'message_id': 'remote'}
            api.send_text.return_value = {'message_id': 'text-remote'}
            api.update_card.return_value = {}
            if outcome == 'missing_id':
                api.send_card.return_value = {}
            if outcome == 'update_rejected':
                api.update_card.side_effect = FeishuRejected(230001)
            if outcome == 'unknown':
                api.send_card.side_effect = OSError('unknown')
            if outcome == 'rejected':
                api.send_card.side_effect = FeishuRejected(230001)
            if outcome == 'throttle':
                api.send_card.side_effect = [FeishuRejected(99991400), {'message_id': 'remote'}]
            worker = BotWorker(c, store, ProcessResources(Catalog()), api)
            stream = ReplyStream(c, row)
            stream.emit({'kind': 'delta', 'text': '预览'})
            await stream.persist(store.conn)
            await worker.send_pending()
            if outcome == 'throttle':
                await store.conn.execute("UPDATE assistant_outbox SET next_attempt_at=now() WHERE input_id=%s", (row['id'],))
                await worker.send_pending()
            # 知识等待恢复会重新创建内存 sink，但同一个 outbox 行和远端卡片复用。
            stream = ReplyStream(c, row)
            stream.emit({'kind': terminal_state, 'text': '最终答案'})
            await stream.persist(store.conn)
            stream.emit({'kind': 'delta', 'text': '迟到增量不能覆盖完成'})
            await stream.persist(store.conn)
            await worker.finish_reply(row, '最终答案')
            await worker.send_pending()
            await worker.send_pending()
            receipt = await (await store.conn.execute("SELECT * FROM assistant_outbox WHERE input_id=%s AND kind='card'", (row['id'],))).fetchone()
            assert receipt['final_state'] == terminal_state
            assert '迟到' not in str(receipt['body'])
            if outcome in {'success', 'throttle'}:
                assert api.send_card.await_count == (2 if outcome == 'throttle' else 1)
                assert receipt['remote_id'] == 'remote'
                assert receipt['status'] == 'delivered'
                api.update_card.assert_awaited_once()
                api.send_text.assert_not_awaited()
            elif outcome in {'unknown', 'missing_id'}:
                api.send_card.assert_awaited_once()
                assert receipt['status'] == 'unknown'
                api.send_text.assert_not_awaited()
            else:
                api.send_card.assert_awaited_once()
                api.send_text.assert_awaited_once()
            counts = (api.send_card.await_count, api.send_text.await_count)
            await worker.recover()
            assert (api.send_card.await_count, api.send_text.await_count) == counts
    asyncio.run(run())


@pytest.mark.skipif(not URL, reason='Set a disposable SESSION_TEST_DATABASE_URL')
@pytest.mark.parametrize('terminal_state', ['complete', 'partial', 'failed', 'unknown', 'cancelled', 'needs_clarification'])
def test_long_final_delivered_in_full_and_failed_display_reuses_card(tmp_path, terminal_state):
    async def run():
        from psycopg.types.json import Jsonb
        c = config(tmp_path, sessions={'database_url': URL}, channels={'feishu': {'streaming': True}})
        async with session_storage(c.sessions, initialize=True):
            pass
        async with operation_storage(c.sessions, initialize=True) as store:
            row = dict(id=str(uuid4()), binding=str(uuid4()), body={'chat_id': 'fixture', 'text': 'input'})
            await store.conn.execute('INSERT INTO assistant_inbox(id,binding,body) VALUES(%s,%s,%s)',
                                     (row['id'], row['binding'], Jsonb(row['body'])))
            api = AsyncMock()
            api.send_card.return_value = {'message_id': 'remote'}
            api.update_card.return_value = {}
            api.send_text.return_value = {'message_id': 'text'}
            worker = BotWorker(c, store, ProcessResources(Catalog()), api)
            stream = ReplyStream(c, row)
            text = '中文正文' * 5000
            stream.emit({'kind': terminal_state, 'text': text})
            await stream.persist(store.conn)
            stored_card = await (await store.conn.execute(
                "SELECT body FROM assistant_outbox WHERE input_id=%s AND kind='card'", (row['id'],))).fetchone()
            assert stored_card['body']['card']['header'] == card('', terminal_state)[0]['header']
            await worker.finish_reply(row, text)
            for _ in range(12):
                await worker.send_pending()
            assert ''.join(call.args[1] for call in api.send_text.call_args_list) == text
            api.send_card.assert_awaited_once()
            # 重复登记最终结果不会产生第二份答案。
            await worker.enqueue_reply(row, text)
            await worker.send_pending()
            assert ''.join(call.args[1] for call in api.send_text.call_args_list) == text
    asyncio.run(run())


def test_outer_failure_preserves_persisted_business_terminal():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from ai_assistant.interfaces.feishu import BotWorker
    async def run():
        existing = {'final_state': 'partial', 'body': {'final_text': '第一项已完成；第二项失败。'}}
        cursor = SimpleNamespace(fetchone=AsyncMock(return_value=existing))
        store = SimpleNamespace(conn=SimpleNamespace(execute=AsyncMock(return_value=cursor)))
        settings = SimpleNamespace(channels=SimpleNamespace(feishu=SimpleNamespace(streaming=True)))
        worker = BotWorker(settings, store, None, None)
        worker.finish_reply = AsyncMock()
        row = {'id': 'fixture-input'}
        await worker.fail_reply(row, '通用失败提示', cancelled=True)
        worker.finish_reply.assert_awaited_once_with(row, '第一项已完成；第二项失败。')
    asyncio.run(run())


def test_markdown_survives_card_creation_and_update_http(tmp_path):
    import httpx
    from ai_assistant.integrations.feishu_files import FeishuAPI

    async def run():
        requests = []
        text = '## SQL\n\n**计数**\n```sql\nSELECT COUNT(*) FROM emp;\n```'
        payload, _ = card(text, 'complete')

        def respond(request):
            if request.url.path.endswith('/tenant_access_token/internal'):
                return httpx.Response(200, json={'tenant_access_token': 'fixture', 'expire': 7200})
            body = json.loads(request.content)
            delivered = json.loads(body['content'])
            assert delivered['schema'] == '2.0'
            assert delivered['body']['elements'] == [{'tag': 'markdown', 'content': text}]
            if request.method == 'POST':
                assert body['msg_type'] == 'interactive'
                assert body['uuid'] == 'fixture-operation'
            requests.append(request.method)
            return httpx.Response(200, json={'code': 0, 'data': {'message_id': 'fixture-message'}})

        c = config(tmp_path, integrations={'feishu': {
            'app_id': 'cli_fixture', 'app_secret': 'not-real', 'owner_open_id': 'owner'}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            api = FeishuAPI(c.integrations.feishu, client)
            await api.send_card('fixture-chat', payload, 'fixture-operation')
            await api.update_card('fixture-message', payload)
        assert requests == ['POST', 'PATCH']
    asyncio.run(run())
