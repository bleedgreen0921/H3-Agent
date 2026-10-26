"""脱敏真实失败轨迹：多动作发现、固定官方 schema、旧时间与调用耗尽。"""
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from ai_assistant.agents.tasks import task_view
from ai_assistant.integrations.mcp import Catalog, Entry, schema_tokens, stable_id
from ai_assistant.integrations.config import MCPServer
from ai_assistant.tools.mcp import load_mcp_tools
from ai_assistant.storage.sessions import SessionError
from tests.runtime.test_task_reliability import task_fixture, invoke

FIXTURES = Path(__file__).parents[1] / 'fixtures'


def official_catalog():
    fixture = json.loads((FIXTURES / 'feishu_task_schemas.json').read_text())
    settings = MCPServer(transport='feishu', identity='user', tools=list(fixture['tools']))
    async def noop(**kwargs):
        return '{}'
    return Catalog([Entry(stable_id('fixture', 'user', name), 'fixture', name, 'hash',
        StructuredTool.from_function(name=stable_id('fixture', 'user', name),
            description=fixture['descriptions'][name], args_schema=schema, coroutine=noop), settings)
        for name, schema in fixture['tools'].items()])


def test_multi_action_search_aliases_and_missing_delete():
    c = official_catalog()
    for query in ['task get members add', '任务 读取 负责人 添加', 'task get member addMembers']:
        page = c.search(query)
        assert {'task_v2_task_get', 'task_v2_task_addMembers'} <= {i['name'] for i in page['items']}
        assert not page['unmatched_terms']
        assert all(i['name'].startswith('task_') for i in page['items'])
        assert len(json.dumps(page, ensure_ascii=False)) <= 1500
    page = c.search('task delete')
    assert page['unmatched_terms'] == ['delete']
    assert page['items'] == []
    for entry in c.entries.values():
        assert c.search(entry.id)['items'][0]['id'] == entry.id


def test_official_schema_budget_atomic_failure_and_suggestion():
    async def run():
        c = official_catalog()
        ids = [e.id for e in c.entries.values() if e.original_name in {
            'task_v2_task_get', 'task_v2_task_patch', 'task_v2_task_addMembers'}]
        runtime = SimpleNamespace(context=SimpleNamespace(mcp=c, mcp_schema_tokens=6000),
            state={'active_mcp_tools': {'old': 'hash'}}, tool_call_id='load')
        assert schema_tokens([c.entries[i].tool for i in ids]) < 6000
        await load_mcp_tools.coroutine(ids, runtime)
        create = next(e.id for e in c.entries.values() if e.original_name == 'task_v2_task_create')
        with pytest.raises(ToolException) as exc:
            await load_mcp_tools.coroutine(ids + [create], runtime)
        assert len(str(exc.value)) <= 1500
        receipt = json.loads(str(exc.value))
        assert receipt['selected_tokens'] > 6000
        assert set(receipt['suggested_tool_ids']) == set(ids)
        assert runtime.state['active_mcp_tools'] == {'old': 'hash'}
        await load_mcp_tools.coroutine(receipt['suggested_tool_ids'], runtime)
    asyncio.run(run())


def test_old_time_view_keeps_raw_timestamp_and_roles():
    failure = json.loads((FIXTURES / 'task_discovery_failure.json').read_text())
    result = ToolMessage(json.dumps({'code': 0, 'data': {'task': {'guid': 'fixture-guid',
        'due': {'timestamp': failure['actual_due']}, 'members': [], 'reminders': [30]}}}), tool_call_id='get')
    view = task_view(result, 'Asia/Shanghai')['task']
    assert view['due']['timestamp'] == '1793002800000'
    assert view['due']['local_datetime'] == '2026-10-26T16:20:00+08:00'
    assert view['timezone'] == 'Asia/Shanghai'
    assert view['members'] == [] and view['reminders'] == [30]


def test_call_exhaustion_does_not_claim_success(tmp_path):
    async def run():
        session, calls = task_fixture(tmp_path, [invoke('search_mcp_tools', {'query': 'task'}, 'search')])
        session.settings = settings = session.settings.model_copy(update={'model_call_limit': 1})
        from ai_assistant.agents import create_assistant_agent
        from langgraph.checkpoint.memory import InMemorySaver
        from tests.runtime.test_session import ScriptedModel
        from ai_assistant.tools.mcp import discovery_tools
        session.agent = create_assistant_agent(ScriptedModel(responses=[invoke('search_mcp_tools',
            {'query': 'task'}, 'search')]), discovery_tools(), checkpointer=InMemorySaver(),
            settings=settings, mcp_enabled=True)
        with pytest.raises(SessionError, match='本轮未记录远端写入'):
            await session.run_turn('修改已有任务')
        assert not calls
    asyncio.run(run())


def test_task_get_view_retains_official_artifact_and_does_not_register_time(tmp_path):
    from ai_assistant.agents.tasks import task_model_result
    from ai_assistant.agents.results import ToolResultMiddleware, read_tool_result
    from ai_assistant.integrations.policy import Caller
    from ai_assistant.tools.context import ToolContext
    from ai_assistant.workspace.files import ReadOnlyWorkspace
    async def run():
        c = official_catalog()
        entry = next(e for e in c.entries.values() if e.original_name == 'task_v2_task_get')
        context = ToolContext(ReadOnlyWorkspace(tmp_path), mcp=c, caller=Caller('feishu_private'))
        raw = json.dumps({'code': 0, 'data': {'task': {'guid': 'fixture-guid',
            'due': {'timestamp': '1793002800000'}, 'members': [], 'reminders': []}}})
        result = ToolMessage(raw, tool_call_id='get', artifact={'structured_content': json.loads(raw)})
        async def handler(request):
            return result
        request = SimpleNamespace(runtime=SimpleNamespace(context=context), tool_call={'name': entry.id})
        bounded = await ToolResultMiddleware().awrap_tool_call(request, handler)
        assert '16:20:00+08:00' in bounded.content
        assert not context.resolved_datetimes
        assert task_model_result(bounded, context) is bounded
        page = await read_tool_result.coroutine('get', SimpleNamespace(context=context, state={'messages': [bounded]}))
        assert raw in page and 'local_datetime' not in page
        group = task_model_result(result, replace(context, caller=Caller('feishu_group')))
        assert json.loads(group.content)['task']['timezone'] == 'UTC'
    asyncio.run(run())


def test_previous_turn_time_registration_is_not_reused(tmp_path):
    async def run():
        session, calls = task_fixture(tmp_path, [
            invoke('load_mcp_tools', {'tool_ids': ['task_v2_task_create']}, 'load'),
            invoke('task_v2_task_create', {'data': {'summary': 'fixture',
                'due': {'timestamp': '1793012400000', 'is_all_day': False}}}, 'write'), AIMessage('未执行')])
        session.context.resolved_datetimes.add(('1793012400000', False))
        await session.run_turn('创建任务')
        assert not calls
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['error', 'timeout', 'cancel'])
def test_transport_failure_has_unknown_receipt_and_never_claims_success(tmp_path, failure):
    async def run():
        from ai_assistant.tools.mcp import discovery_tools
        from ai_assistant.tools.datetime import resolve_datetime
        from ai_assistant.agents import create_assistant_agent
        from langgraph.checkpoint.memory import InMemorySaver
        from tests.runtime.test_session import ScriptedModel
        responses = [invoke('load_mcp_tools', {'tool_ids': ['task_v2_task_create']}, 'load'),
            invoke('resolve_datetime', {'value': '2026-10-26T19:00:00'}, 'time'),
            invoke('task_v2_task_create', {'data': {'summary': 'fixture',
                'due': {'timestamp': '1793012400000', 'is_all_day': False}}}, 'write')]
        session, _ = task_fixture(tmp_path, responses)
        calls = []
        started = asyncio.Event()
        async def broken(**kwargs):
            calls.append(True)
            if failure == 'cancel':
                started.set()
                await asyncio.Event().wait()
            if failure == 'timeout':
                raise TimeoutError('fixture timeout')
            raise RuntimeError('fixture transport failure')
        original = session.context.mcp.entries['task_v2_task_create']
        entry = replace(original, tool=original.tool.model_copy(update={'coroutine': broken}))
        c = Catalog([entry if e.id == entry.id else e for e in session.context.mcp.entries.values()])
        session.context = replace(session.context, mcp=c)
        session.agent = create_assistant_agent(ScriptedModel(responses=responses),
            discovery_tools() + [resolve_datetime] + [e.tool for e in c.entries.values()],
            checkpointer=InMemorySaver(), settings=session.settings, mcp_enabled=True)
        events = []
        expected = asyncio.CancelledError if failure == 'cancel' else TimeoutError if failure == 'timeout' else RuntimeError
        execution = asyncio.create_task(session.run_turn('创建任务', on_output=events.append))
        if failure == 'cancel':
            await asyncio.wait_for(started.wait(), 2)
            execution.cancel()
        with pytest.raises(expected):
            await execution
        assert calls == [True]
        assert (await session.context.operations.list(''))[0]['status'] == 'unknown'
        assert not any(e['kind'] == 'complete' for e in events)
    asyncio.run(run())


def test_exhaustion_after_write_reports_partial_and_feishu_preserves_it(tmp_path):
    from ai_assistant.agents import create_assistant_agent
    from ai_assistant.runtime import TurnIncompleteError
    from ai_assistant.interfaces.feishu import failure_message
    from ai_assistant.tools.datetime import resolve_datetime
    from ai_assistant.tools.mcp import discovery_tools
    from langgraph.checkpoint.memory import InMemorySaver
    from tests.runtime.test_session import ScriptedModel
    async def run():
        responses = [invoke('load_mcp_tools', {'tool_ids': ['task_v2_task_create']}, 'load'),
            invoke('resolve_datetime', {'value': '2026-10-26T19:00:00'}, 'time'),
            invoke('task_v2_task_create', {'data': {'summary': '模拟面试',
                'due': {'timestamp': '1793012400000', 'is_all_day': False}}}, 'write')]
        session, calls = task_fixture(tmp_path, responses)
        session.settings = session.settings.model_copy(update={'model_call_limit': 3})
        session.agent = create_assistant_agent(ScriptedModel(responses=responses),
            discovery_tools() + [resolve_datetime] + [e.tool for e in session.context.mcp.entries.values()],
            checkpointer=InMemorySaver(), settings=session.settings, mcp_enabled=True)
        events = []
        with pytest.raises(TurnIncompleteError) as exc:
            await session.run_turn('创建任务', on_output=events.append)
        assert '部分完成' in str(exc.value) and 'succeeded' in str(exc.value)
        assert failure_message(exc.value) == str(exc.value)
        assert events[-1]['text'] == str(exc.value)
        assert [name for name, _ in calls] == ['task_v2_task_create']
    asyncio.run(run())
