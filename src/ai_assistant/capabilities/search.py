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
    # 必须已有本轮能力状态、注册表和初始计划，才能扩展工具。
    state = getattr(runtime.context, 'capabilities', None)
    if state is None or state.registry is None or state.plan is None:
        raise ToolException('能力搜索当前不可用。')

    # 限制本轮搜索次数；未设置时使用默认上限。
    maximum = getattr(state.registry.settings, 'max_search_calls', 3)
    if state.search_calls >= maximum:
        raise ToolException('本轮能力搜索次数已达上限。')

    # 只读状态优先于计划中的效果；注册表继续检查身份和 schema 预算。
    effect = 'read' if getattr(state, 'read_only', False) else state.plan.effect
    added, skipped, exhausted = state.registry.extend_plan(
        state.plan, query, limit=limit, effect=effect)
    state.search_calls += 1

    # 记录扩展后的计划快照，便于与本轮请求和执行线程关联。
    state.registry.record_plan(
        state.plan, input_id=getattr(runtime.context, 'input_id', None),
        session_id=getattr(runtime.context, 'session_id', None),
        change='expand_tools',
        execution_thread_id=getattr(runtime.context, 'execution_thread_id', None),
        search_calls=state.search_calls,
    )

    summaries = [item.summary() for item in added]
    if not summaries:
        # 没有新增工具时返回检索摘要；这些命中不一定已加入当前计划。
        summaries = state.registry.search_tools(query, limit)

        # 对摘要补充只读和禁止创建过滤；摘要本身不表示工具可执行。
        if getattr(state, 'read_only', False) or getattr(state.plan, 'effect', None) == 'read':
            summaries = [item for item in summaries
                         if not state.registry.descriptors[item['id']].write]
        if getattr(state.plan, 'forbid_create', False):
            summaries = [
                item for item in summaries
                if not any(term in state.registry.descriptors[item['id']].original_name.casefold()
                           for term in ('document_create', 'doc_create', 'spacenode_create'))
            ]

    # 出现预算跳过时返回数量；有跳过且没有新增时标记预算耗尽。
    result = {
        'tools': summaries,
        'exposed_capabilities': len(state.plan.tool_ids),
    }
    if skipped:
        result['budget_skipped'] = skipped
    if exhausted:
        result['budget_exhausted'] = True
    return result


# 将工具抛出的 ToolException 转为工具错误结果，供模型处理。
search_tools = search_tools.model_copy(update={'handle_tool_error': True})
