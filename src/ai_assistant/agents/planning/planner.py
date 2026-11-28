"""Planner/Replanner 节点：生成计划、调整计划和补全资源解析。"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.runnables import RunnableConfig

from ai_assistant.agents.planning.models import Plan, _now
from ai_assistant.agents.planning.prompts import _planner_prompt, _replanner_prompt
from ai_assistant.agents.planning.resources import _bind_resources, _resolve_plan
from ai_assistant.agents.planning.validation import (
    _safe_plan_error, _validate_revision, validate_plan,
)
from ai_assistant.llm.structured import invoke_structured
from ai_assistant.observability import event
from ai_assistant.runtime.execution import raise_if_fatal

logger = logging.getLogger('ai_assistant.agents.planning')


async def _structured_plan(model, messages, config, *, phase, timeout):
    base = dict(config or {})
    base['metadata'] = {**(base.get('metadata') or {}), 'assistant_phase': phase}
    async with asyncio.timeout(timeout):
        return await invoke_structured(model, Plan, messages, config=base)


async def _planner(state, config: RunnableConfig, *, model, registry, settings):
    planning = state.get('planning') or {}
    try:
        resources = list(planning.get('resources') or [])
        plan = await _structured_plan(
            model, _planner_prompt(state, planning, settings.max_tasks), config,
            phase='plan', timeout=settings.planner_timeout_seconds)
        _bind_resources(plan, resources)
        validate_plan(plan, settings.max_tasks, resources)
        resolved = _resolve_plan({**planning, 'resources': resources}, plan, registry, settings)
        updated = {**resolved, 'status': 'draft',
                   'plan': plan.model_dump(mode='json'), 'goal': plan.goal,
                   'questions': list(plan.questions), 'updated_at': _now()}
        event('plan.created', task_count=len(plan.tasks), plan_id=planning.get('plan_id'),
              question_count=len(plan.questions))
        return {'planning': updated, 'planning_note': None,
                'planning_action': None, 'planning_goal': None}
    except Exception as error:
        raise_if_fatal(error)
        logger.warning('Planning failed: category=%s detail=%s',
                       type(error).__name__, _safe_plan_error(error))
        event('plan.failed', status='failed', category=type(error).__name__)
        return {'planning': planning, 'planning_note': _safe_plan_error(error),
                'planning_action': None, 'planning_goal': None}


async def _replanner(state, config: RunnableConfig, *, model, registry, settings):
    planning = state.get('planning') or {}
    results = state.get('task_results') or {}
    replans = int(planning.get('replans') or 0)
    try:
        resources = list(planning.get('resources') or [])
        plan = await _structured_plan(
            model, _replanner_prompt(state, planning, settings.max_tasks), config,
            phase='replan', timeout=settings.planner_timeout_seconds)
        _bind_resources(plan, resources)
        validate_plan(plan, settings.max_tasks, resources)
        _validate_revision(planning.get('plan'), plan, results)
        resolved = _resolve_plan({**planning, 'resources': resources}, plan, registry, settings)
        updated = {**resolved, 'plan': plan.model_dump(mode='json'), 'status': 'running',
                   'replans': replans + 1, 'updated_at': _now()}
        event('plan.replanned', task_count=len(plan.tasks), plan_id=planning.get('plan_id'),
              replans=replans + 1)
        return {'planning': updated, 'planning_note': None}
    except Exception as error:
        raise_if_fatal(error)
        logger.warning('Replanning failed: category=%s detail=%s',
                       type(error).__name__, _safe_plan_error(error))
        event('plan.replanned', status='failed', category=type(error).__name__,
              replans=replans + 1)
        # 失败原因必须可读；沿用脱敏摘要，不透传异常正文。
        return {'planning': {**planning, 'status': 'failed', 'replans': replans + 1},
                'planning_note': f'计划调整未完成（{_safe_plan_error(error)}）；保留已有结果。'}


def _ensure_resolved(planning, registry, settings):
    plan = planning.get('plan') or {}
    tasks = plan.get('tasks') or []
    resolved = dict(planning.get('resolved') or {})
    if tasks and all(task.get('id') in resolved for task in tasks):
        return planning
    model = Plan.model_validate(plan)
    return _resolve_plan(planning, model, registry, settings)
