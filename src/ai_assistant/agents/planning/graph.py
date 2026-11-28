"""规划父图组装：普通 Agent 与规划节点共享原生 checkpoint。"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ai_assistant.agents.planning.executor import _execute_task
from ai_assistant.agents.planning.finalizer import _finalizer
from ai_assistant.agents.planning.intake import _intake_with_intent, _route_intake
from ai_assistant.agents.planning.models import PlanState
from ai_assistant.agents.planning.planner import _planner, _replanner
from ai_assistant.agents.planning.scheduler import _route_scheduler, _scheduler
from ai_assistant.tools.context import ToolContext


def create_planning_agent(*, normal_agent, task_agent, model, checkpointer, registry,
                          settings, max_input_tokens=None):
    """组装规划父图；normal_agent 必须使用 PlanState 作为 state_schema。"""

    async def intake(state, config: RunnableConfig):
        return await _intake_with_intent(state, config, model=model, settings=settings)

    async def planner(state, config: RunnableConfig):
        return await _planner(state, config, model=model, registry=registry, settings=settings)

    async def replanner(state, config: RunnableConfig):
        return await _replanner(state, config, model=model, registry=registry, settings=settings)

    def scheduler(state):
        return _scheduler(state, registry=registry, settings=settings)

    def router(state):
        return _route_scheduler(state, registry=registry, settings=settings)

    async def executor_read(payload, config: RunnableConfig, runtime: Runtime[ToolContext]):
        return await _execute_task(payload, config, runtime, write=False,
                                   task_agent=task_agent, registry=registry, settings=settings)

    async def executor_write(payload, config: RunnableConfig, runtime: Runtime[ToolContext]):
        return await _execute_task(payload, config, runtime, write=True,
                                   task_agent=task_agent, registry=registry, settings=settings)

    builder = StateGraph(PlanState, context_schema=ToolContext)
    builder.add_node('intake', intake)
    builder.add_node('normal', normal_agent)
    builder.add_node('planner', planner)
    builder.add_node('scheduler', scheduler)
    builder.add_node('executor_read', executor_read)
    builder.add_node('executor_write', executor_write)
    builder.add_node('replanner', replanner)
    builder.add_node('finalizer', _finalizer)

    builder.add_edge(START, 'intake')
    builder.add_conditional_edges('intake', _route_intake, {
        'normal': 'normal', 'planner': 'planner', 'scheduler': 'scheduler',
        'finalizer': 'finalizer'})
    builder.add_edge('normal', END)
    builder.add_edge('planner', 'finalizer')
    builder.add_conditional_edges('scheduler', router, [
        'executor_read', 'executor_write', 'replanner', 'finalizer'])
    builder.add_edge('executor_read', 'scheduler')
    builder.add_edge('executor_write', 'scheduler')
    builder.add_edge('replanner', 'scheduler')
    builder.add_edge('finalizer', END)
    return builder.compile(checkpointer=checkpointer)
