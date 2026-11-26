"""Native skill tools and checkpoint-safe activation reducer."""
from typing import Annotated
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langgraph.types import Command
from pydantic import Field
from ai_assistant.capabilities.plan import extend_skills
from ai_assistant.capabilities.selector import selection_text
from ai_assistant.runtime.execution import raise_if_fatal
from ai_assistant.tools.context import ToolContext
from ai_assistant.skills.catalog import SkillError



def catalog(runtime):
    """取得运行时 Skill 目录，未配置时返回工具错误。"""
    if runtime.context.services.get('skills') is None:
        raise ToolException('Skills are not configured.')
    return runtime.context.services.get('skills')


def selected(runtime, identity):
    """要求 Skill 已激活且会话记录摘要与当前包快照一致。"""
    try:
        skill = catalog(runtime).prepare(identity)
        if runtime.state.get('active_skills', {}).get(skill.id) != skill.digest:
            raise SkillError('Load this skill before accessing resources or dependencies.')
        return skill
    except SkillError as exc:
        raise ToolException(str(exc)) from None


@tool
async def search_skills(
    query: Annotated[str, Field(min_length=1, max_length=1000)],
    runtime: ToolRuntime[ToolContext],
) -> dict:
    """Search for additional Skills not currently in the directory; this never activates a Skill."""
    state = getattr(runtime.context, 'capabilities', None)
    plan = getattr(state, 'plan', None)
    if state is None or state.registry is None or plan is None:
        raise ToolException('Skill 搜索当前不可用。')

    maximum = getattr(state.registry.settings, 'max_search_calls', 3)
    if state.search_calls >= maximum:
        raise ToolException('本轮能力搜索次数已达上限。')

    selector = (getattr(runtime.context, 'services', {}) or {}).get('capability_selector')
    if selector is None:
        raise ToolException('能力选择器当前不可用。')

    candidates = state.registry.eligible_skills(
        exclude=plan.skill_directory_ids + plan.active_skill_ids)
    state.search_calls += 1
    if not candidates:
        return {'skills': [], 'directory_count': len(plan.skill_directory_ids),
                'message': '没有新的相关 Skill。'}

    messages = getattr(runtime, 'state', None)
    messages = messages.get('messages', ()) if isinstance(messages, dict) else ()
    try:
        selection = await selector.select(
            selection_text(messages), query=query,
            tools=[], skills=candidates, max_tools=0, schema_budget=0,
            directory_budget=getattr(runtime.context, 'skills_directory_tokens', 6000))
    except Exception as error:
        raise_if_fatal(error)
        raise ToolException('Skill 搜索失败，请稍后重试。') from None

    added, skipped, exhausted = extend_skills(
        state.registry, plan, selection.skill_ids, eligible_skills=candidates,
        directory_budget=getattr(runtime.context, 'skills_directory_tokens', 6000))
    state.registry.record_plan(
        plan, input_id=getattr(runtime.context, 'input_id', None),
        session_id=getattr(runtime.context, 'session_id', None),
        change='expand_skills',
        execution_thread_id=getattr(runtime.context, 'execution_thread_id', None),
        search_calls=state.search_calls)

    result = {
        'skills': [item.summary() for item in added],
        'directory_count': len(plan.skill_directory_ids),
    }
    if not added:
        result['message'] = '没有新的相关 Skill。'
    if skipped:
        result['budget_skipped'] = skipped
    if exhausted:
        result['budget_exhausted'] = True
    return result


@tool
async def load_skill(skill_id: str, runtime: ToolRuntime[ToolContext]) -> Command:
    """Activate a skill for subsequent model requests; its instructions do not expand permissions."""
    try:
        skill = catalog(runtime).prepare(skill_id)
    except SkillError as exc:
        raise ToolException(str(exc)) from None

    # 用 ID 和快照摘要更新会话状态；实际指令由后续模型请求加载。
    info = skill.info()
    info.update(loaded=True, workspace=str(runtime.context.workspace.root),
                output='Write outputs to the configured workspace using the available file tools.')
    from ai_assistant.observability import event
    event('capability.skill', skill_id=skill.id, change='load', status='loaded')
    return Command(update={'active_skills': {skill.id: skill.digest}, 'messages': [
        ToolMessage(content=str(info), tool_call_id=runtime.tool_call_id, name='load_skill')]})


@tool
async def unload_skill(skill_id: str, runtime: ToolRuntime[ToolContext]) -> Command:
    """Stop providing this skill's instructions on subsequent model calls."""
    try:
        skill = catalog(runtime).resolve(skill_id)
    except SkillError as exc:
        raise ToolException(str(exc)) from None

    # 以 None 标记卸载，停止在后续请求中提供该 Skill 指导。
    from ai_assistant.observability import event
    event('capability.skill', skill_id=skill.id, change='unload', status='unloaded')
    return Command(update={'active_skills': {skill.id: None}, 'messages': [
        ToolMessage(content='Unloaded ' + skill.id, tool_call_id=runtime.tool_call_id, name='unload_skill')]})


@tool
async def read_skill_resource(skill_id: str, resource: str, runtime: ToolRuntime[ToolContext],
                              offset: Annotated[int, Field(ge=0)] = 0,
                              limit: Annotated[int, Field(ge=1, le=20000)] = 20000) -> dict:
    """Read a resource from an active skill; binary files return their snapshot location."""
    # 先校验激活状态和摘要，再由目录限制资源路径和分页范围。
    skill = selected(runtime, skill_id)
    try:
        return catalog(runtime).read(skill.id, resource, offset, limit)
    except SkillError as exc:
        raise ToolException(str(exc)) from None


def create_skill_tools():
    """创建支持预期工具错误处理的 Skill 工具实例。"""
    items = [search_skills, load_skill, unload_skill, read_skill_resource]
    return [item.model_copy(update={'handle_tool_error': True}) for item in items]
