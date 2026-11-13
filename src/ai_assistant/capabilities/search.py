"""模型可调用的能力发现入口；只扩展计划，不直接执行。"""
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import Field

from ai_assistant.tools.context import ToolContext


@tool
async def search_tools(query: str, runtime: ToolRuntime[ToolContext],
                       limit: Annotated[int, Field(ge=1, le=30)] = 8) -> dict:
    """Search additional capabilities not currently exposed; results are names and short descriptions."""
    state = getattr(runtime.context, 'capabilities', None)
    if state is None or state.registry is None or state.plan is None:
        raise ToolException('能力搜索当前不可用。')
    maximum = getattr(state.registry.settings, 'max_search_calls', 3)
    if state.search_calls >= maximum:
        raise ToolException('本轮能力搜索次数已达上限。')
    results = state.registry.search_tools(query, limit)
    ids = [item['id'] for item in results]
    known = set(state.plan.tool_ids)
    for item in ids:
        if item not in known:
            state.plan.tool_ids.append(item)
            known.add(item)
    state.plan.tool_schema_tokens = state.registry.schema_tokens(state.plan.tool_ids)
    state.search_calls += 1
    state.registry.record_plan(
        state.plan, input_id=getattr(runtime.context, 'input_id', None),
        session_id=getattr(runtime.context, 'session_id', None),
        change='expand_tools',
        execution_thread_id=getattr(runtime.context, 'execution_thread_id', None),
        search_calls=state.search_calls,
    )
    return {'tools': results, 'exposed_capabilities': len(state.plan.tool_ids)}


search_tools = search_tools.model_copy(update={'handle_tool_error': True})
