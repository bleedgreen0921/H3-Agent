"""独立数据库验证会话用量，不接触真实模型和账号。"""
import asyncio
import os
from uuid import uuid4
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from ai_assistant.runtime.observation import TurnObserver
from ai_assistant.storage.usage import usage_storage
from ai_assistant.storage.sessions import session_storage
from tests.storage.test_sessions import config, fake_models
from ai_assistant.runtime import assistant_session

URL = os.environ.get('SESSION_TEST_DATABASE_URL')
pytestmark = pytest.mark.skipif(not URL, reason='Set a disposable SESSION_TEST_DATABASE_URL')


def test_usage_roles_partial_dedup_failure_and_reopen(tmp_path):
    async def run():
        c = config(tmp_path)
        async with session_storage(c.sessions, initialize=True) as repo:
            row = await repo.open(tmp_path)
            sid = row['id']
            async with usage_storage(c.sessions) as store:
                observer = TurnObserver(5, 'same-profile', store, sid)
                call = uuid4()
                await observer.on_chat_model_start({}, [[]], run_id=call)
                response = LLMResult(generations=[[ChatGeneration(message=AIMessage('ok',
                    usage_metadata={'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15}))]])
                await observer.on_llm_end(response, run_id=call)
                await observer.on_llm_end(response, run_id=call)
                call = uuid4()
                await observer.on_chat_model_start({}, [[]], run_id=call,
                    metadata={'lc_source': 'summarization', 'assistant_model_profile': 'same-profile'})
                await store.finish(call, 'failed', {'input_tokens': 7})
                missing = uuid4()
                await store.start(missing, sid, 'same-profile', 'main')
                report = await store.report(sid)
                assert '主模型（same-profile）：输入 12，输出 3，合计 15' in report
                assert '摘要模型（same-profile）：输入 7，输出 未知，合计 未知' in report
                assert '进行中 1' in report and '部分用量 1' in report
                await store.abandon(sid)
                report = await store.report(sid)
                assert '进行中 1' not in report
            async with usage_storage(c.sessions) as reopened:
                assert '合计 15' in await reopened.report(sid)
            await repo.delete(sid)
            async with usage_storage(c.sessions) as reopened:
                count = await (await reopened.conn.execute('SELECT count(*) AS n FROM assistant_model_usage WHERE session_id=%s', (sid,))).fetchone()
                assert count['n'] == 0
    asyncio.run(run())


def test_session_usage_survives_failed_branch_new_boundary_and_active_lock(tmp_path, monkeypatch):
    fake_models(monkeypatch, [AIMessage('ok', usage_metadata={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}),
                              AIMessage('FAIL'), AIMessage('ok')])
    async def run():
        c = config(tmp_path)
        async with session_storage(c.sessions, initialize=True):
            pass
        ids = []
        try:
            async with assistant_session(c) as session:
                ids.append(session.session_id)
                await session.run_turn('first')
                with pytest.raises(RuntimeError):
                    await session.run_turn('failure')
                # report 使用独立连接，正在持有会话锁时也能读取。
                async with usage_storage(c.sessions) as store:
                    report = await store.report(session.session_id)
                    assert '调用 2' in report and '合计 12' in report
                await session.new_conversation()
                ids.append(session.session_id)
                async with usage_storage(c.sessions) as store:
                    assert '尚无模型调用' in await store.report(session.session_id)
                    assert '调用 2' in await store.report(ids[0])
        finally:
            async with session_storage(c.sessions) as repo:
                for sid in ids:
                    await repo.delete(sid)
    asyncio.run(run())
