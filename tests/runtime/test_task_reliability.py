"""离线复现任务时间、默认成员与错误回执；不调用真实服务。"""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from copy import deepcopy
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent
from ai_assistant.agents.tasks import prepare_task, discrepancies
from ai_assistant.config import AgentSettings
from ai_assistant.integrations.config import MCPServer
from ai_assistant.integrations.mcp import Catalog, Entry
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import AssistantSession, final_text
from ai_assistant.tools.context import ToolContext
from ai_assistant.tools.datetime import convert_datetime, resolve_datetime, get_current_identity
from ai_assistant.tools.mcp import discovery_tools
from ai_assistant.workspace.files import ReadOnlyWorkspace
from tests.runtime.test_session import ScriptedModel


def test_time_conversion_and_dst():
    assert convert_datetime('2026-10-26T19:00:00')['timestamp'] == '1793012400000'
    assert convert_datetime('2026-10-27T00:00:00')['timestamp'] == '1793030400000'
    assert convert_datetime('2026-10-26T11:00:00Z')['timestamp'] == '1793012400000'
    assert convert_datetime('2026-10-26')['is_all_day']
    for value in ['2026-03-08T02:30:00', '2026-11-01T01:30:00']:
        with pytest.raises(ToolException):
            convert_datetime(value, 'America/New_York')
    assert convert_datetime('2026-11-01T01:30:00-04:00', 'America/New_York')['datetime'].endswith('-04:00')
    for value in ['tonight', '2026-02-30T19:00:00']:
        with pytest.raises(ToolException):
            convert_datetime(value)


def test_owner_explicit_members_and_group_isolation(tmp_path):
    context = ToolContext(ReadOnlyWorkspace(tmp_path), caller=Caller('feishu_private'),
                          owner_open_id='owner-fixture', default_task_assignee='owner')
    entry = SimpleNamespace(original_name='task_v2_task_create')
    args = {'data': {'summary': 'fixture'}}
    assert prepare_task(context, entry, args)['members'][0]['id'] == 'owner-fixture'
    for members in [[], [{'id': 'other', 'role': 'assignee'}]]:
        assert prepare_task(context, entry, {'data': {'members': members}})['members'] == members
    group = replace(context, caller=Caller('feishu_group'))
    assert 'members' not in prepare_task(group, entry, {'data': {}})
    assert not discrepancies({'members': []}, {'guid': 'fixture-guid', 'members': [
        {'id': 'creator', 'role': 'follower'}]}, 'task_v2_task_create')
    with pytest.raises(ToolException, match='resolve_datetime'):
        prepare_task(context, entry, {'data': {'due': {'timestamp': '1793002800000'}}})
    with pytest.raises(ToolException):
        asyncio.run(get_current_identity.coroutine(SimpleNamespace(context=group)))
    with pytest.raises(ToolException, match='群聊不继承'):
        asyncio.run(resolve_datetime.coroutine('2026-10-26T19:00:00', SimpleNamespace(context=group)))


class Operations:
    def __init__(self):
        self.rows = {}
    async def begin(self, input_id, binding, tool, args):
        key = (input_id, tool, json.dumps(args, sort_keys=True))
        if key in self.rows:
            return False, self.rows[key]
        row = {'id': str(len(self.rows)), 'status': 'pending', 'result': None}
        self.rows[key] = row
        return True, row
    async def finish(self, ident, status, result=None):
        row = next(row for row in self.rows.values() if row['id'] == ident)
        row.update(status=status, result=result or row['result'])
    async def list(self, input_id):
        return list(self.rows.values())


def task_fixture(tmp_path, responses, *, wrong=False, fail_read=False):
    calls = []
    actual = {'guid': 'fixture-guid', 'task_id': 'fixture-id', 'summary': '模拟面试',
              'due': {'timestamp': '1793002800000' if wrong else '1793012400000', 'is_all_day': False},
              'members': [{'id': 'owner-fixture', 'role': 'assignee', 'type': 'user'}],
              'reminders': [{'relative_fire_minute': 30}]}
    names = ['task_v2_task_create', 'task_v2_task_get', 'task_v2_task_patch', 'task_v2_task_addMembers']
    settings = MCPServer(transport='feishu', identity='user', tools=names,
                         write_tools=[n for n in names if not n.endswith('_get')], allow_write=True)
    entries = []
    for name in names:
        def bind(name):
            async def invoke(**kwargs):
                calls.append((name, deepcopy(kwargs)))
                if name.endswith('_get') and fail_read:
                    raise RuntimeError('read failed')
                return json.dumps({'code': 0, 'data': {'task': actual}})
            return invoke
        schemas = json.loads((Path(__file__).parents[1] / 'fixtures/feishu_task_schemas.json').read_text())['tools']
        native = StructuredTool.from_function(name=name, description='Fixture task',
            coroutine=bind(name), args_schema=schemas[name])
        entries.append(Entry(name, 'fixture', name, 'hash', native, settings))
    directory = Catalog(entries)
    context = ToolContext(ReadOnlyWorkspace(tmp_path), mcp=directory, caller=Caller('feishu_private'),
        operations=Operations(), owner_open_id='owner-fixture', default_task_assignee='owner')
    tools = discovery_tools() + [resolve_datetime] + [e.tool for e in entries]
    model = ScriptedModel(responses=responses)
    agent = create_assistant_agent(model, tools, checkpointer=InMemorySaver(),
                                  settings=AgentSettings(), mcp_enabled=True)
    return AssistantSession(agent, context, AgentSettings()), calls


def invoke(name, args, ident):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': ident}])


@pytest.mark.parametrize('fail_read', [False, True])
def test_wrong_receipt_replaces_model_success_and_keeps_remote_receipt(tmp_path, fail_read):
    async def run():
        responses = [invoke('load_mcp_tools', {'tool_ids': ['task_v2_task_create']}, 'load'),
            invoke('resolve_datetime', {'value': '2026-10-26T19:00:00'}, 'time'),
            invoke('task_v2_task_create', {'data': {'summary': '模拟面试',
                'due': {'timestamp': '1793012400000', 'is_all_day': False},
                'reminders': [{'relative_fire_minute': 30}]}}, 'write'),
            invoke('task_v2_task_create', {'data': {'summary': '自动补写尝试'}}, 'forbidden-rewrite'),
            AIMessage(content='已成功创建，晚上七点。'),
            AIMessage(content='你好')]
        session, calls = task_fixture(tmp_path, responses, wrong=True, fail_read=fail_read)
        result = await session.run_turn('为我创建面试任务，今天晚上七点，提前半小时提醒')
        assert final_text(result.messages).startswith('未完成')
        assert '16:20:00+08:00' in final_text(result.messages)
        assert len([name for name, _ in calls if name.endswith('_create')]) == 1
        assert (await session.context.operations.list(''))[0]['status'] == 'succeeded'
        assert (await session.agent.aget_state(session._complete)).values['turn_outcome']
        followup = await session.run_turn('你好')
        assert final_text(followup.messages) == '你好'
        assert not (await session.agent.aget_state(session._complete)).values['active_mcp_tools'].get('task_v2_task_create')
    asyncio.run(run())


def test_unconverted_time_never_reaches_remote(tmp_path):
    async def run():
        responses = [invoke('load_mcp_tools', {'tool_ids': ['task_v2_task_create']}, 'load'),
            invoke('task_v2_task_create', {'data': {'summary': 'fixture',
                   'due': {'timestamp': '1793002800000'}}}, 'write'), AIMessage(content='请先确认日期')]
        session, calls = task_fixture(tmp_path, responses)
        result = await session.run_turn('创建任务')
        assert not calls
        assert any(isinstance(m, ToolMessage) and 'resolve_datetime' in str(m.content) for m in result.messages)
    asyncio.run(run())


def test_patch_requires_get_and_member_add_avoids_duplicates(tmp_path):
    async def run():
        responses = [invoke('load_mcp_tools', {'tool_ids': ['task_v2_task_addMembers']}, 'load'),
            invoke('task_v2_task_addMembers', {'path': {'task_guid': 'fixture-guid'}, 'data': {
                'members': [{'id': 'owner-fixture', 'role': 'assignee'}]}}, 'add'), AIMessage(content='已有负责人')]
        session, calls = task_fixture(tmp_path, responses)
        await session.run_turn('为已有任务添加我为负责人')
        assert [name for name, _ in calls] == ['task_v2_task_get']
    asyncio.run(run())


def test_greeting_naming_create_then_modify_official_schemas(tmp_path):
    async def run():
        responses = [AIMessage('你好'), AIMessage('以后叫我小助理'),
            invoke('load_mcp_tools', {'tool_ids': ['task_v2_task_create']}, 'load-create'),
            invoke('resolve_datetime', {'value': '2026-10-26T19:00:00'}, 'time-create'),
            invoke('task_v2_task_create', {'data': {'summary': '模拟面试',
                'due': {'timestamp': '1793012400000', 'is_all_day': False},
                'reminders': [{'relative_fire_minute': 30}]}}, 'create'), AIMessage('已核实创建'),
            invoke('load_mcp_tools', {'tool_ids': ['task_v2_task_patch', 'task_v2_task_addMembers']}, 'load-patch'),
            invoke('resolve_datetime', {'value': '2026-10-26T19:00:00'}, 'time-patch'),
            invoke('task_v2_task_patch', {'path': {'task_guid': 'fixture-guid'}, 'data': {
                'task': {'due': {'timestamp': '1793012400000', 'is_all_day': False}}, 'update_fields': ['due']}}, 'patch'),
            invoke('task_v2_task_addMembers', {'path': {'task_guid': 'fixture-guid'}, 'data': {
                'members': [{'id': 'owner-fixture', 'role': 'assignee'}]}}, 'member'), AIMessage('已核实时间和负责人')]
        session, calls = task_fixture(tmp_path, responses)
        await session.run_turn('你好')
        await session.run_turn('就叫你小助理')
        await session.run_turn('给自己创建模拟面试任务，今天七点，提前半小时提醒')
        turn = await session.run_turn('把时间设为今天晚上七点，负责人是我')
        assert final_text(turn.messages) == '已核实时间和负责人'
        assert [name for name, _ in calls] == ['task_v2_task_create', 'task_v2_task_get',
                                               'task_v2_task_patch', 'task_v2_task_get']
        assert calls[0][1]['data']['members'][0]['id'] == 'owner-fixture'
        assert not (await session.agent.aget_state(session._complete)).values['turn_outcome']
    asyncio.run(run())
