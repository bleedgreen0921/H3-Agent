"""Native skill tools and checkpoint-safe activation reducer."""
from typing import Annotated
from langchain.agents import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langgraph.types import Command
from pydantic import Field
from ai_assistant.tools.context import ToolContext
from ai_assistant.skills.catalog import SkillError


def merge_skills(current, updates):
    return {**(current or {}), **(updates or {})}


class SkillState(AgentState):
    active_skills: Annotated[dict[str, str | None], merge_skills]


def catalog(runtime):
    if runtime.context.skills is None:
        raise ToolException('Skills are not configured.')
    return runtime.context.skills


def selected(runtime, identity):
    try:
        skill = catalog(runtime).resolve(identity)
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
    return catalog(runtime).search(query, offset, limit)


@tool
async def load_skill(skill_id: str, runtime: ToolRuntime[ToolContext]) -> Command:
    """Activate a skill for subsequent model requests; its instructions do not expand permissions."""
    try:
        skill = catalog(runtime).resolve(skill_id)
    except SkillError as exc:
        raise ToolException(str(exc)) from None
    info = skill.info()
    info.update(loaded=True, workspace=str(runtime.context.workspace.root),
                output='Write outputs to the workspace (scratch profile: /tmp).',
                execution='run_command' if runtime.context.terminal else 'Terminal disabled; scripts cannot execute.')
    return Command(update={'active_skills': {skill.id: skill.digest}, 'messages': [
        ToolMessage(content=str(info), tool_call_id=runtime.tool_call_id, name='load_skill')]})


@tool
async def unload_skill(skill_id: str, runtime: ToolRuntime[ToolContext]) -> Command:
    """Stop providing this skill's instructions on subsequent model calls."""
    try:
        skill = catalog(runtime).resolve(skill_id)
    except SkillError as exc:
        raise ToolException(str(exc)) from None
    return Command(update={'active_skills': {skill.id: None}, 'messages': [
        ToolMessage(content='Unloaded ' + skill.id, tool_call_id=runtime.tool_call_id, name='unload_skill')]})


@tool
async def read_skill_resource(skill_id: str, resource: str, runtime: ToolRuntime[ToolContext],
                              offset: Annotated[int, Field(ge=0)] = 0,
                              limit: Annotated[int, Field(ge=1, le=20000)] = 20000) -> dict:
    """Read a resource from an active skill; binary files return their sandbox location."""
    skill = selected(runtime, skill_id)
    try:
        return catalog(runtime).read(skill.id, resource, offset, limit)
    except SkillError as exc:
        raise ToolException(str(exc)) from None


@tool
async def install_skill_dependencies(skill_id: str, manager: str, requirements: list[str],
                                     runtime: ToolRuntime[ToolContext]) -> dict:
    """Install a structured pip/npm requirement transaction, allowing only new dependencies."""
    selected(runtime, skill_id)
    if runtime.context.skill_dependencies is None:
        raise ToolException('Controlled dependency installation is disabled.')
    try:
        return await runtime.context.skill_dependencies.install(manager, requirements)
    except SkillError as exc:
        raise ToolException(str(exc)) from None


def create_skill_tools(dependencies=True):
    items = [search_skills, load_skill, unload_skill, read_skill_resource]
    if dependencies:
        items.append(install_skill_dependencies)
    return [item.model_copy(update={'handle_tool_error': True}) for item in items]
