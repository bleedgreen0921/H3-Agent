"""模型可调用的能力发现入口；只扩展计划，不直接执行。"""
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import Field

from ai_assistant.capabilities.plan import extend_tools
from ai_assistant.capabilities.selector import selection_text
from ai_assistant.runtime.execution import raise_if_fatal
from ai_assistant.tools.context import ToolContext


def _messages(runtime):
    state = getattr(runtime, 'state', None)
    return state.get('messages', ()) if isinstance(state, dict) else ()


@tool
async def search_tools(
    query: Annotated[str, Field(min_length=1, max_length=1000)],
    runtime: ToolRuntime[ToolContext],
) -> dict:
    """Search additional capabilities not currently exposed; only relevant tools are added."""
    # 必须已有本轮能力状态、注册表和初始计划，才能扩展工具。
    state = getattr(runtime.context, 'capabilities', None)
    if state is None or state.registry is None or state.plan is None:
        raise ToolException('能力搜索当前不可用。')

    # 限制本轮搜索次数；未设置时使用默认上限。
    maximum = getattr(state.registry.settings, 'max_search_calls', 3)
    if state.search_calls >= maximum:
        raise ToolException('本轮能力搜索次数已达上限。')

    selector = (getattr(runtime.context, 'services', {}) or {}).get('capability_selector')
    if selector is None:
        raise ToolException('能力选择器当前不可用。')

    # 只读状态优先于计划中的效果；候选继续遵守身份、创建和 schema 预算。
    effect = 'read' if getattr(state, 'read_only', False) else state.plan.effect
    if effect not in {'read', 'write'}:
        effect = 'write' if state.plan.requires_write else 'read'
    candidates = state.registry.eligible_tools(
        effect=effect, server=state.plan.server, identity=state.plan.identity,
        forbid_create=state.plan.forbid_create, exclude=state.plan.tool_ids)
    state.search_calls += 1
    if not candidates:
        return {'tools': [], 'exposed_capabilities': len(state.plan.tool_ids),
                'message': '没有新的相关能力。'}

    settings = state.registry.settings
    try:
        selection = await selector.select(
            selection_text(_messages(runtime), fallback=query),
            tools=candidates, skills=[], max_tools=getattr(settings, 'top_k', 8),
            schema_budget=getattr(settings, 'max_schema_tokens', 0),
            directory_budget=0)
    except Exception as error:
        raise_if_fatal(error)
        raise ToolException('能力搜索失败，请稍后重试。') from None

    added, skipped, exhausted = extend_tools(
        state.registry, state.plan, selection.tool_ids,
        eligible_tools=candidates)
    state.registry.record_plan(
        state.plan, input_id=getattr(runtime.context, 'input_id', None),
        session_id=getattr(runtime.context, 'session_id', None),
        change='expand_tools',
        execution_thread_id=getattr(runtime.context, 'execution_thread_id', None),
        search_calls=state.search_calls)

    result = {
        'tools': [item.summary() for item in added],
        'exposed_capabilities': len(state.plan.tool_ids),
    }
    if not added:
        result['message'] = '没有新的相关能力。'
    if skipped:
        result['budget_skipped'] = skipped
    if exhausted:
        result['budget_exhausted'] = True
    return result


# 将工具抛出的 ToolException 转为工具错误结果，供模型处理。
search_tools = search_tools.model_copy(update={'handle_tool_error': True})
