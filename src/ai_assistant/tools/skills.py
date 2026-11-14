"""Native skill tools and checkpoint-safe activation reducer."""
from typing import Annotated
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langgraph.types import Command
from pydantic import Field
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
async def search_skills(runtime: ToolRuntime[ToolContext], query: str = '',
                        offset: Annotated[int, Field(ge=0)] = 0,
                        limit: Annotated[int, Field(ge=1, le=100)] = 20) -> dict:
    """Search names and descriptions in the fixed local skill snapshot; pagination is explicit."""
    result = catalog(runtime).search(query, offset, limit)

    # 搜索结果扩展当前计划中的目录项，但不会激活 Skill 正文。
    state = getattr(runtime.context, 'capabilities', None)
    plan = getattr(state, 'plan', None)
    if plan is not None:
        for item in result['skills']:
            if item['id'] not in plan.skill_directory_ids:
                plan.skill_directory_ids.append(item['id'])
        plan.skill_directory_tokens = state.registry.skill_directory_tokens(plan.skill_directory_ids)
        state.registry.record_plan(
            plan, input_id=getattr(runtime.context, 'input_id', None),
            session_id=getattr(runtime.context, 'session_id', None),
            change='expand_skills',
            execution_thread_id=getattr(runtime.context, 'execution_thread_id', None),
            search_calls=state.search_calls,
        )
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
