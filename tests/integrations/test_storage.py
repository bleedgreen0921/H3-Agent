"""Opt-in disposable PostgreSQL only; no user database or external service."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import os
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from ai_assistant.config import SessionsConfig
from ai_assistant.integrations.config import MCPServer
from ai_assistant.integrations.mcp import Catalog, Entry, stable_id
from ai_assistant.integrations.policy import Caller
from ai_assistant.interfaces.feishu import BotWorker, ProcessResources
from ai_assistant.runtime import assistant_session
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.storage.sessions import session_storage
from tests.integrations.test_mcp import config
from tests.integrations.test_native import message
from tests.runtime.test_session import ScriptedModel

URL = os.environ.get('SESSION_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not URL, reason='Set a disposable SESSION_TEST_DATABASE_URL')


async def setup(c):
    async with session_storage(c.sessions, initialize=True):
        pass
    async with operation_storage(c.sessions, initialize=True):
        pass


def fake_model(monkeypatch, responses):
    @asynccontextmanager
    async def models(*args, **kwargs):
        yield ScriptedModel(responses=responses)
    monkeypatch.setattr('ai_assistant.runtime.model_session', models)


def test_effect_dedup_survives_failed_turn(tmp_path, monkeypatch):
    async def run():
        calls = []
        @tool
        async def write(value: str) -> str:
            """A fixture effect, never an external write."""
            calls.append(value); return 'accepted'
        identity = stable_id('fixture', 'private', 'write')
        settings = MCPServer(transport='http',url='http://fixture',tools=['write'],write_tools=['write'],allow_write=True)
        entry = Entry(identity,'fixture','write','schema',write.model_copy(update={'name':identity}),settings)
        directory = Catalog([entry])
        c = config(tmp_path, sessions={'database_url':URL},mcp={})
        await setup(c)
        fake_model(monkeypatch,[message('load_mcp_tools',{'tool_ids':[identity]}),
            message(identity,{'value':'one'}),AIMessage(content='FAIL'),
            message('load_mcp_tools',{'tool_ids':[identity]}),message(identity,{'value':'one'}),AIMessage(content='recovered')])
        input_id = str(uuid4())
        # Inject the fixed local catalog through the same borrowed-resource path as the bot.
        c = c.model_copy(update={'mcp':c.mcp.model_copy(update={'servers':{'fixture':settings}})})
        async with assistant_session(c,resources=directory) as session:
            with pytest.raises(RuntimeError):
                await session.run_turn('first', input_id=input_id)
            result = await session.run_turn('retry input', input_id=input_id)
            assert calls == ['one']
            assert len(result.operations) == 1 and result.operations[0]['status'] == 'succeeded'
            row = await (await session.repository.execute('SELECT last_input_id FROM assistant_sessions WHERE id=%s',
                (session.session_id,))).fetchone()
            assert row['last_input_id'] == input_id
    asyncio.run(run())


def test_operation_claim_concurrency_and_unknown(tmp_path):
    async def run():
        c = config(tmp_path, sessions={'database_url':URL})
        await setup(c)
        input_id = str(uuid4())
        async with operation_storage(c.sessions) as a, operation_storage(c.sessions) as b:
            results = await asyncio.gather(a.begin(input_id,'owner','write',{'value':1}),
                                           b.begin(input_id,'owner','write',{'value':1}))
            assert sum(fresh for fresh,row in results) == 1
            assert results[0][1]['id'] == results[1][1]['id']
            await a.finish(results[0][1]['id'],'unknown')
            fresh, receipt = await b.begin(input_id,'owner','write',{'value':1})
            assert not fresh and receipt['status'] == 'unknown'
    asyncio.run(run())


def test_bot_success_without_reply_recovers_without_model(tmp_path, monkeypatch):
    async def run():
        from psycopg.types.json import Jsonb
        c = config(tmp_path, sessions={'database_url':URL},mcp={},integrations={'feishu':{
            'app_id':'cli_fixture','app_secret':'fake','owner_open_id':'owner'}},channels={'feishu':{'enabled':True}})
        await setup(c)
        input_id = str(uuid4()); binding = str(uuid4())
        row = {'id':input_id,'binding':binding,'body':{'text':'input','chat_id':'chat','group':False,'owner_open_id':'owner'}}
        fake_model(monkeypatch,[AIMessage(content='completed answer')])
        async with assistant_session(c,caller=Caller('feishu_private',True,binding)) as session:
            await session.run_turn('input', input_id=input_id)
            sid = session.session_id
        async with operation_storage(c.sessions) as store:
            await store.conn.execute('INSERT INTO assistant_channel_sessions VALUES(%s,%s)',(binding,sid))
            await store.conn.execute("INSERT INTO assistant_inbox(id,binding,body,status) VALUES(%s,%s,%s,'running')",
                                     (input_id,binding,Jsonb(row['body'])))
            api = AsyncMock(); api.send_text.return_value={'message_id':'remote'}
            worker = BotWorker(c,store,ProcessResources(Catalog()),api)
            # Any accidental Agent rerun would fail this scripted model immediately.
            fake_model(monkeypatch,[AIMessage(content='FAIL')])
            await worker.recover()
            api.send_text.assert_awaited_once()
            assert api.send_text.call_args.args[1] == 'completed answer'
            received = await (await store.conn.execute('SELECT status,body FROM assistant_inbox WHERE id=%s',(input_id,))).fetchone()
            assert received == {'status':'delivered','body':None}
            await worker.recover()
            api.send_text.assert_awaited_once()
    asyncio.run(run())


def test_unknown_outbox_is_not_resent_and_parts_keep_order(tmp_path):
    async def run():
        from psycopg.types.json import Jsonb
        c = config(tmp_path,sessions={'database_url':URL})
        await setup(c)
        input_id,binding = str(uuid4()),str(uuid4())
        row = {'id':input_id,'binding':binding,'body':{'text':'input','chat_id':'chat'}}
        async with operation_storage(c.sessions) as store:
            await store.conn.execute('INSERT INTO assistant_inbox(id,binding,body) VALUES(%s,%s,%s)',
                (input_id,binding,Jsonb(row['body'])))
            api = AsyncMock(); api.send_text.side_effect = [{'message_id':'first'}, OSError('uncertain')]
            worker = BotWorker(c,store,ProcessResources(Catalog()),api)
            await worker.enqueue_reply(row,'A'*3000+'B'*100)
            await worker.send_pending()
            await worker.send_pending()
            assert [call.args[1] for call in api.send_text.call_args_list] == ['A'*3000,'B'*100]
            await worker.send_pending()
            assert api.send_text.await_count == 2
            parts = await (await store.conn.execute('SELECT status,body FROM assistant_outbox WHERE input_id=%s ORDER BY part',
                (input_id,))).fetchall()
            assert [p['status'] for p in parts] == ['delivered','unknown']
    asyncio.run(run())


def test_effect_survives_checkpoint_pointer_failure(tmp_path,monkeypatch):
    async def run():
        from unittest.mock import patch
        from ai_assistant.storage.sessions import SessionStorageError
        calls = []
        @tool
        async def write() -> str:
            """Fixture effect for checkpoint failure recovery."""
            calls.append(1); return 'accepted'
        settings = MCPServer(transport='http',url='http://fixture',tools=['write'],write_tools=['write'],allow_write=True)
        identity = stable_id('fixture','private','write')
        entry = Entry(identity,'fixture','write','schema',write.model_copy(update={'name':identity}),settings)
        c = config(tmp_path,sessions={'database_url':URL},mcp={'servers':{'fixture':settings}})
        await setup(c)
        fake_model(monkeypatch,[message('load_mcp_tools',{'tool_ids':[identity]}),message(identity),AIMessage(content='done'),
            message('load_mcp_tools',{'tool_ids':[identity]}),message(identity),AIMessage(content='recovered')])
        input_id = str(uuid4())
        async with assistant_session(c,resources=Catalog([entry])) as session:
            with patch.object(session.repository,'complete',side_effect=SessionStorageError('pointer failed')):
                with pytest.raises(SessionStorageError):
                    await session.run_turn('effect then checkpoint failure',input_id=input_id)
            result = await session.run_turn('same input',input_id=input_id)
            assert calls == [1] and result.operations[0]['status'] == 'succeeded'
    asyncio.run(run())
