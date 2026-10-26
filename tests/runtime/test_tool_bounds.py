"""Large catalogs, selected schemas and paged result artifacts."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools import ToolException
from ai_assistant.integrations.mcp import Catalog, Entry
from ai_assistant.integrations.config import MCPServer
from ai_assistant.tools.mcp import load_mcp_tools
from ai_assistant.agents.results import ToolResultMiddleware


def directory(count=20):
    @tool
    def sample():
        """A fixture read."""
        return ''
    settings = MCPServer(transport='http', url='http://fixture', tools=['read'])
    return Catalog([Entry(str(i), 'fixture', 'read', 'digest', sample.model_copy(
        update={'description': '描述' * 120}), settings) for i in range(count)])


def test_catalog_bound_and_cursor_use_actual_return_count():
    catalog = directory()
    first = catalog.search(limit=100)
    assert len(json.dumps(first, ensure_ascii=False)) <= 3000
    assert first['next_offset'] == len(first['items']) < 20
    second = catalog.search(offset=first['next_offset'], limit=100)
    assert not set(i['id'] for i in first['items']) & set(i['id'] for i in second['items'])
    assert len(first['items']) + len(second['items']) <= 20


def test_selection_releases_previous_tools_and_refuses_oversized_schema():
    async def run():
        runtime = SimpleNamespace(context=SimpleNamespace(mcp=directory(), mcp_schema_tokens=6000),
                                  state={'active_mcp_tools': {'old': 'digest'}}, tool_call_id='load')
        command = await load_mcp_tools.coroutine(['1'], runtime)
        assert command.update['active_mcp_tools'] == {'old': None, '1': 'digest'}
        runtime.context.mcp_schema_tokens = 100
        with pytest.raises(ToolException, match='分批'):
            await load_mcp_tools.coroutine([str(i) for i in range(20)], runtime)
    asyncio.run(run())


def test_result_artifact_and_error_status_are_preserved():
    async def run():
        result = ToolMessage(content='正文' * 10000, tool_call_id='result', status='error', artifact={'id': 'receipt'})
        async def handler(request):
            return result
        request = SimpleNamespace(runtime=SimpleNamespace(context=SimpleNamespace(mcp=None)))
        bounded = await ToolResultMiddleware(4000).awrap_tool_call(request, handler)
        assert len(bounded.content) < 4000
        assert bounded.status == 'error'
        assert bounded.artifact['full_tool_content'] == result.content
        assert bounded.artifact['original_artifact'] == {'id': 'receipt'}
        assert 'read_tool_result' in bounded.content
        assert 'receipt' in bounded.content and 'error' in bounded.content
    asyncio.run(run())


def test_full_result_remains_readable_after_summary_from_same_thread_history():
    from ai_assistant.agents.results import read_tool_result
    async def run():
        seen = []
        async def history(config, **kwargs):
            seen.append(config)
            yield SimpleNamespace(values={'messages': [ToolMessage('preview', tool_call_id='old',
                artifact={'full_tool_content': '完整正文'})]}, config={'configurable': {'checkpoint_id': 'cursor'}})
        runtime = SimpleNamespace(state={'messages': []}, context=SimpleNamespace(checkpoint_history=history),
                                  config={'configurable': {'thread_id': 'trusted'}})
        assert '完整正文' in await read_tool_result.coroutine('old', runtime)
        assert seen == [{'configurable': {'thread_id': 'trusted'}}]
    asyncio.run(run())


def test_native_checkpoint_artifact_read_after_compression(tmp_path):
    from dataclasses import replace
    from ai_assistant.agents.results import read_tool_result
    from tests.runtime.test_context import managed
    from tests.runtime.test_session import call
    from ai_assistant.agents.context import is_summary

    async def run():
        @tool
        async def large():
            """Read a large immutable fixture."""
            return '原始完整正文' * 3000
        read_call = AIMessage(content='', tool_calls=[dict(name='read_tool_result', id='read',
            args={'result_id': 'original', 'offset': 5000})])
        session = managed(tmp_path, [call('large', 'original'), AIMessage(content='答' * 500),
                                   read_call, AIMessage(content='完成')],
                          tools=[large, read_tool_result], budget=10000)
        session.context = replace(session.context, checkpoint_history=session.agent.aget_state_history)
        await session.run_turn('读取大结果')
        turn = await session.run_turn('继续读取，保留约束。' + '问' * 3000)
        assert any(is_summary(m) for m in turn.messages)
        assert not any(isinstance(m, ToolMessage) and m.tool_call_id == 'original' for m in turn.messages)
        result = next(m for m in turn.messages if isinstance(m, ToolMessage) and m.tool_call_id == 'read')
        assert '原始完整正文' in result.content and 'next_offset=' in result.content
    asyncio.run(run())
