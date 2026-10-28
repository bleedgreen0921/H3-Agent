"""业务验收报告应能序列化 PostgreSQL 汇总值，不触发真实模型。"""
import asyncio
from contextlib import asynccontextmanager
from decimal import Decimal
import json
import runpy
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage

from ai_assistant.config import AppConfig, ModelConfig, ContextConfig
from ai_assistant.runtime.observation import TurnUsage


def test_summary_report_serializes_postgres_numeric(monkeypatch):
    module = runpy.run_path('scripts/accept_responses_business.py')
    operation = module['summary_reopen']
    config = AppConfig(default_model='offline', models={
        'offline': ModelConfig(provider='ollama', model='offline')}, context=ContextConfig())
    content = ''
    calls = 0

    async def seed(config, update):
        nonlocal content
        content = update['messages'][0].text
        return {'configurable': {'thread_id': 'test'}}

    async def state(config):
        return SimpleNamespace(values={'messages': [HumanMessage(
            content, additional_kwargs={'lc_source': 'summarization'})]})

    async def turn(prompt):
        nonlocal calls
        calls += 1
        return SimpleNamespace(messages=[AIMessage(content)], usage=TurnUsage(
            main_model_calls=1, summary_model_calls=1 if calls == 1 else 0))

    rows = [{'role': 'main', 'calls': 2, 'total_tokens': Decimal('3929')},
            {'role': 'summary', 'calls': 1, 'total_tokens': Decimal('1353')}]
    session = SimpleNamespace(session_id='test', _thread='test', _complete=None,
        agent=SimpleNamespace(aupdate_state=seed, aget_state=state),
        repository=SimpleNamespace(complete=AsyncMock()), run_turn=turn,
        context=SimpleNamespace(usage_store=SimpleNamespace(conn=SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(fetchall=AsyncMock(return_value=rows)))))))

    @asynccontextmanager
    async def opened(*args, **kwargs):
        yield session

    monkeypatch.setitem(operation.__globals__, 'assistant_session', opened)
    report = {}
    asyncio.run(operation(config, report))
    restored = json.loads(json.dumps(report))
    assert restored['passed']
    assert restored['cumulative_usage'][0]['total_tokens'] == 3929
    assert calls == 2
