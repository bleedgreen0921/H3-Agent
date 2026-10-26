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


@pytest.mark.skipif(not URL, reason='Set a disposable SESSION_TEST_DATABASE_URL')
@pytest.mark.parametrize('outcome', ['success', 'unknown', 'rejected', 'throttle', 'update_rejected', 'missing_id'])
def test_card_creation_receipts_final_priority_and_resume(tmp_path, outcome):
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
            stream.emit({'kind': 'complete', 'text': '最终答案'})
            await stream.persist(store.conn)
            stream.emit({'kind': 'delta', 'text': '迟到增量不能覆盖完成'})
            await stream.persist(store.conn)
            await worker.finish_reply(row, '最终答案')
            await worker.send_pending()
            await worker.send_pending()
            receipt = await (await store.conn.execute("SELECT * FROM assistant_outbox WHERE input_id=%s AND kind='card'", (row['id'],))).fetchone()
            assert receipt['final_state'] == 'complete'
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
def test_long_final_delivered_in_full_and_failed_display_reuses_card(tmp_path):
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
            stream.emit({'kind': 'complete', 'text': text})
            await stream.persist(store.conn)
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
