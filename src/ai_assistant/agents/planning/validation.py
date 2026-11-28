"""计划 DAG 校验、修订校验和安全错误摘要。"""
from __future__ import annotations

import asyncio

from pydantic import ValidationError

from ai_assistant.agents.planning.models import Plan, PlanValidationError
from ai_assistant.llm.completion import IncompleteResponseError
from ai_assistant.llm.structured import StructuredOutputError


def validate_plan(plan: Plan, max_tasks: int, resources=()):
    """校验 DAG、依赖、任务数量和写任务资源；失败时拒绝整个计划。"""
    if not plan.tasks and not plan.questions:
        raise PlanValidationError('计划必须包含任务，或明确列出需要用户补充的信息。')
    if len(plan.tasks) > max_tasks:
        raise PlanValidationError(f'计划任务超过上限（最多 {max_tasks} 个）。')
    ids = [task.id for task in plan.tasks]
    if len(ids) != len(set(ids)):
        raise PlanValidationError('计划包含重复的任务 ID。')
    known = set(ids)
    resource_ids = {item['id'] if isinstance(item, dict) else item.id
                    for item in resources or []}
    for task in plan.tasks:
        if len(set(task.deps)) != len(task.deps):
            raise PlanValidationError('任务依赖包含重复 ID。')
        if task.id in task.deps:
            raise PlanValidationError('任务不能依赖自身。')
        if any(dep not in known for dep in task.deps):
            raise PlanValidationError('任务依赖引用了不存在的任务。')
        if any(item not in resource_ids for item in task.resources):
            raise PlanValidationError('任务引用了不存在的资源 ID。')
        if task.effect == 'write' and not task.resources and not task.creates:
            raise PlanValidationError('写任务缺少目标资源；请补充文档链接或明确创建新文档。')
        if task.append and task.effect != 'write':
            raise PlanValidationError('append 只能用于写入任务。')
        if task.append and task.creates:
            raise PlanValidationError('append 不能与创建新资源同时使用。')

    # Kahn 拓扑校验：存在环时拒绝执行。
    indegree = {task.id: len(set(task.deps)) for task in plan.tasks}
    children = {task.id: [] for task in plan.tasks}
    for task in plan.tasks:
        for dep in set(task.deps):
            children[dep].append(task.id)
    ready = sorted(task_id for task_id, value in indegree.items() if value == 0)
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(plan.tasks):
        raise PlanValidationError('计划依赖存在环路。')
    return plan


def _safe_plan_error(error):
    if isinstance(error, StructuredOutputError):
        return str(error)
    if isinstance(error, IncompleteResponseError):
        return str(error)
    if isinstance(error, PlanValidationError):
        message = str(error).strip()
        if message and len(message) <= 240:
            return message
        return '计划结构不符合要求，请缩小目标或补充约束。'
    if isinstance(error, ValidationError):
        # 只报告字段路径和类型，不记录模型原始内容。
        fields = []
        for item in error.errors(include_input=False):
            location = '.'.join(str(part) for part in item.get('loc', ())) or '计划'
            fields.append(f'{location}: {item.get("type", "invalid")}')
        detail = '；'.join(fields[:6])
        return f'计划输出字段不符合要求（{detail}）；请补充目标或调整约束。'
    if isinstance(error, ValueError):
        return '计划结构不符合要求，请缩小目标或补充约束。'
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return '计划生成超时，请稍后重试。'
    return '计划生成失败，请稍后重试或补充目标。'


def _validate_revision(old_plan, new_plan, results):
    old = {task['id']: task for task in (old_plan or {}).get('tasks') or []}
    new = {task.id: task for task in new_plan.tasks}
    items = (results or {}).get('items') or {}
    replaced = {}
    for task in new_plan.tasks:
        for task_id in task.replaces:
            if task_id not in old:
                raise PlanValidationError(f'replaces 引用了不存在的旧任务 {task_id}。')
            if task_id in replaced:
                raise PlanValidationError(f'同一旧任务 {task_id} 被多个新任务替代。')
            if (items.get(task_id) or {}).get('status') == 'done':
                raise PlanValidationError(f'已完成任务 {task_id} 不能被替代。')
            replaced[task_id] = task.id
    for task_id, result in items.items():
        if task_id in new:
            continue
        if result.get('status') == 'done':
            raise PlanValidationError(f'修订计划删除了已完成任务 {task_id}。')
        if task_id not in replaced:
            raise PlanValidationError(
                f'删除失败或阻塞任务 {task_id} 时必须通过 replaces 指定替代任务。')
    for task in new_plan.tasks:
        if task.id not in old and task.id in items:
            raise PlanValidationError(f'新任务 {task.id} 复用了已有结果的任务 ID。')
        if task.id in old and (items.get(task.id) or {}).get('status') == 'done':
            before = old[task.id]
            if task.description != before.get('description') or \
                    task.deps != list(before.get('deps') or []) or \
                    task.resources != list(before.get('resources') or []):
                raise PlanValidationError(f'修订计划修改了已完成任务 {task.id}。')
