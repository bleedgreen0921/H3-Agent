"""多次空搜索和多页目录不会挤占用户请求及工具配对。"""
import asyncio
import json
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from ai_assistant.agents.context import project_messages, count_tokens, validate_tool_pairs, InputBudgetMiddleware
from ai_assistant.agents.results import read_tool_result
from ai_assistant.integrations.mcp import Entry, Catalog
from ai_assistant.integrations.config import MCPServer
from tests.runtime.test_tool_bounds import directory
from tests.runtime.test_context import managed
from tests.runtime.test_task_reliability import invoke
from ai_assistant.tools.mcp import search_mcp_tools
from dataclasses import replace


def test_chinese_alias_ranking_filter_and_page_budget():
    base = directory(1).entries['0']
    entries = [replace(base, id='patch', original_name='task_v2_task_patch'),
               replace(base, id='get', original_name='task_v2_task_get'),
               replace(base, id='contact', original_name='contact_v3_user_get')]
    catalog = Catalog(entries)
    assert catalog.search('任务 修改')['items'][0]['id'] == 'patch'
    assert catalog.search('get', domain='task')['items'][0]['id'] == 'get'
    assert catalog.search('get', server='absent')['total'] == 0
    assert 'addMembers' in catalog.search('patch')['items'][0]['description']
    page = directory(50).search(limit=100)
    assert page['returned'] <= 5
    assert len(json.dumps(page, ensure_ascii=False)) <= 1500
    assert page['next_offset'] == page['returned']


def test_projection_preserves_original_pairs_and_latest_body():
    messages = [HumanMessage('当前请求')]
    for index in range(7):
        messages += [invoke('search_mcp_tools', {'query': 'task', 'offset': index * 5}, str(index)),
                     ToolMessage(json.dumps({'items': ['x' * 1000], 'next_offset': index * 5 + 5}),
                                 tool_call_id=str(index), name='search_mcp_tools')]
    projected = project_messages(messages)
    validate_tool_pairs(projected)
    assert projected[0] is messages[0]
    assert 'x' * 1000 in projected[-1].content
    assert '旧工具正文' in projected[2].content
    assert 'x' * 1000 in messages[2].content


def test_two_empty_searches_and_five_pages_fit_24000(tmp_path):
    async def run():
        responses = [invoke('search_mcp_tools', {'query': 'no-such-word'}, 'empty1'),
                     invoke('search_mcp_tools', {'query': 'another-missing-word'}, 'empty2')]
        for index in range(5):
            responses.append(invoke('search_mcp_tools', {'query': '', 'offset': index * 5}, f'page{index}'))
        responses.append(AIMessage('目录读取完成，缺少的能力未执行。'))
        session = managed(tmp_path, responses, tools=[search_mcp_tools, read_tool_result], budget=24000)
        session.context = replace(session.context, mcp=directory(50))
        turn = await session.run_turn('查找任务工具；不要猜测用户身份')
        assert turn.usage.main_model_calls == 8
        assert turn.usage.max_input_tokens < 21600
        validate_tool_pairs(turn.messages)
    asyncio.run(run())


def test_projection_total_limits_and_write_receipt_protection():
    messages = [HumanMessage('保留请求')]
    for index in range(5):
        messages.extend([invoke('read', {}, f'r{index}'), ToolMessage('正文' * 1500,
            tool_call_id=f'r{index}', name='read', artifact={'full_tool_content': '原文'})])
    receipt = ToolMessage('写回执' * 2000, tool_call_id='write', name='write',
                          artifact={'protected_receipt': True})
    messages.extend([invoke('write', {}, 'write'), receipt])
    projected = project_messages(messages)
    assert count_tokens([m for m in projected if isinstance(m, ToolMessage) and m.name == 'read']) <= 4000
    assert projected[-1] is receipt
    validate_tool_pairs(projected)
