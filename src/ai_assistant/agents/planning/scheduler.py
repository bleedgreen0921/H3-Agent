"""按 DAG 调度任务：只读并行、写任务串行、失败阻塞传播。"""
from __future__ import annotations

from langgraph.types import Send

from ai_assistant.agents.planning.models import BLOCKING, PlanValidationError
from ai_assistant.agents.planning.planner import _ensure_resolved
from ai_assistant.agents.planning.results import (
    _effective_status, _replacement_map, _task_result,
)
from ai_assistant.runtime.execution import raise_if_fatal


def _scheduler(state, *, registry, settings):
    planning = state.get('planning') or {}
    plan = planning.get('plan') or {}
    tasks = plan.get('tasks') or []
    plan_id = planning.get('plan_id')
    results = state.get('task_results') or {}
    items = dict(results.get('items') or {})
    by_id = {task.get('id'): task for task in tasks}
    if planning.get('status') == 'failed':
        return {'task_results': {'plan_id': plan_id, 'items': items}, 'dispatch': []}
    try:
        planning = _ensure_resolved(planning, registry, settings)
    except Exception as error:
        raise_if_fatal(error)
        note = str(error) if isinstance(error, PlanValidationError) else '任务资源解析失败，请补充目标或调整计划。'
        return {'planning': {**planning, 'status': 'draft', 'planning_note': note},
                'task_results': {'plan_id': plan_id, 'items': items}, 'dispatch': []}
    resolved = planning.get('resolved') or {}
    replacements = _replacement_map(planning)
    if any(_effective_status(task_id, items, replacements) == 'needs_input'
           for task_id in items):
        return {'planning': planning,
                'task_results': {'plan_id': plan_id, 'items': items},
                'dispatch': []}

    # 依赖失败时向下游传播 blocked；固定点处理传递依赖。
    for _ in range(len(tasks) + 1):
        changed = False
        for task in tasks:
            task_id = task.get('id')
            if task_id in items or task_id not in by_id:
                continue
            deps = task.get('deps') or []
            if any(_effective_status(dep, items, replacements) in BLOCKING for dep in deps):
                items[task_id] = _task_result(
                    task_id, 'blocked', '依赖任务未完成或被阻塞。',
                    error={'category': 'dependency_blocked', 'message': '前置任务未完成。'})
                changed = True
        if not changed:
            break

    ready = [
        task for task in tasks
        if task.get('id') not in items
        and all(_effective_status(dep, items, replacements) in {'done', 'superseded'}
                for dep in (task.get('deps') or []))
    ]
    writes = [task for task in ready
              if (resolved.get(task.get('id')) or {}).get('effect') == 'write']
    if writes:
        selected = writes[:1]
    else:
        selected = ready[:settings.max_parallel]
    dispatch = [task.get('id') for task in selected if task.get('id')]
    return {'planning': planning, 'task_results': {'plan_id': plan_id, 'items': items},
            'dispatch': dispatch}


def _route_scheduler(state, *, registry, settings):
    planning = state.get('planning') or {}
    plan = planning.get('plan') or {}
    tasks = plan.get('tasks') or []
    plan_id = planning.get('plan_id')
    dispatch = state.get('dispatch') or []
    results = state.get('task_results') or {}
    items = results.get('items') or {}
    by_id = {task.get('id'): task for task in tasks}
    resolved = planning.get('resolved') or {}

    if dispatch:
        sends = []
        for task_id in dispatch:
            task = by_id.get(task_id)
            if task is None:
                continue
            execution = resolved.get(task_id) or {}
            dependencies = {
                dep: {'status': (items.get(dep) or {}).get('status'),
                      'summary': (items.get(dep) or {}).get('summary', ''),
                      'output': (items.get(dep) or {}).get('output', '')}
                for dep in (task.get('deps') or [])
            }
            write = execution.get('effect') == 'write'
            node = 'executor_write' if write else 'executor_read'
            sends.append(Send(node, {'task': task, 'dependencies': dependencies,
                                     'plan_id': plan_id, 'goal': plan.get('goal'),
                                     'request': planning.get('request'),
                                     'execution': execution,
                                     'side_effect': 'write' if write else 'read'}))
        if sends:
            return sends

    if planning.get('planning_note') or planning.get('status') == 'failed':
        return 'finalizer'
    replacements = _replacement_map(planning)
    if any(_effective_status(task_id, items, replacements) == 'needs_input'
           for task_id in items):
        return 'finalizer'
    failures = any(_effective_status(task_id, items, replacements) in BLOCKING
                   for task_id in items)
    if failures and int(planning.get('replans') or 0) < settings.max_replans:
        return 'replanner'
    return 'finalizer'
