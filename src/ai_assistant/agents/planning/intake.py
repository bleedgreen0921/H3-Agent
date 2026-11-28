"""规划入口状态机、自然语言意图判断和路由。"""
from __future__ import annotations

import asyncio
import logging

from ai_assistant.agents.planning.models import PlanIntent, _now
from ai_assistant.agents.planning.prompts import _intent_prompt, _last_user_text
from ai_assistant.agents.planning.rendering import _planning_status
from ai_assistant.agents.planning.resources import _extract_resources, _merge_resources
from ai_assistant.agents.planning.results import _empty_results
from ai_assistant.llm.structured import invoke_structured
from ai_assistant.observability import event
from ai_assistant.runtime.execution import raise_if_fatal

logger = logging.getLogger('ai_assistant.agents.planning')


def _new_plan(goal, input_id):
    plan_id = input_id or f'plan-{_now()}'
    return {'active': True, 'status': 'draft', 'request': goal, 'goal': goal,
            'resources': [item.model_dump(mode='json') for item in _extract_resources(goal)],
            'supplements': [], 'resolved': {}, 'plan_id': plan_id,
            'plan': None, 'questions': [], 'replans': 0, 'updated_at': _now()}


def _normalize_planning(planning, text=None):
    """兼容旧 draft，并合并补充消息中的新资源引用。"""
    if not planning:
        return {}
    normalized = dict(planning)
    request = normalized.get('request') or normalized.get('goal') or ''
    resources = list(normalized.get('resources') or _extract_resources(request))
    if text:
        resources = _merge_resources(resources, text)
    normalized['request'] = request
    normalized['resources'] = resources
    normalized.setdefault('supplements', [])
    normalized.setdefault('resolved', {})
    normalized.setdefault('questions', [])
    return normalized


def _intake(state, action=None):
    if action is None:
        action = state.get('planning_action')
    goal = (state.get('planning_goal') or '').strip()
    planning = _normalize_planning(state.get('planning'))
    input_id = state.get('input_id') or ''
    common = {'planning_action': None, 'planning_goal': None, 'dispatch': []}

    if action == 'create':
        if planning.get('status') == 'running':
            return {**common, 'planning': planning,
                    'planning_note': '当前计划正在执行；请等待完成，或使用 /cancel 后重新规划。'}
        if planning.get('status') in {'draft'}:
            return {**common, 'planning': planning,
                    'planning_note': '当前已有进行中的计划；请直接补充内容，/plan run 执行，或 /plan exit 退出。'}
        if not goal:
            return {**common, 'planning': planning or None,
                    'planning_note': '用法：/plan <目标>。例如：/plan 整理本周会议并创建待办。'}
        created = _new_plan(goal, input_id)
        return {**common, 'planning': created, 'planning_note': None,
                'task_results': _empty_results(created['plan_id'])}

    if action == 'run':
        plan = planning.get('plan') or {}
        questions = plan.get('questions') or []
        if questions:
            return {**common, 'planning': planning,
                    'planning_note': '计划还需要补充信息；请先回复以下问题，补充后会重新规划。'}
        if planning.get('status') == 'running':
            return {**common, 'planning': planning,
                    'planning_note': '计划正在执行；请等待完成，或使用 /cancel 后重新规划。'}
        if planning.get('status') in {'completed', 'failed', 'cancelled'}:
            return {**common, 'planning': planning,
                    'planning_note': '当前计划已结束；发送新的规划目标可重新规划，或 /plan exit 退出规划模式。'}
        if planning.get('status') != 'draft' or not plan or not plan.get('tasks'):
            return {**common, 'planning': planning,
                    'planning_note': '当前没有可执行的计划；先使用 /plan <目标> 创建计划。'}
        updated = {**planning, 'status': 'running', 'replans': 0,
                   'resolved': {}, 'updated_at': _now()}
        return {**common, 'planning': updated, 'planning_note': None,
                'task_results': _empty_results(planning.get('plan_id'))}

    if action == 'exit':
        if planning.get('status') == 'running':
            return {**common, 'planning': planning,
                    'planning_note': '计划正在执行；请先 /cancel，完成清理后再退出规划模式。'}
        return {**common, 'planning': None,
                'planning_note': '已退出规划模式。' if planning else '当前不在规划模式。',
                'task_results': _empty_results(input_id)}

    if action == 'status':
        return {**common, 'planning': planning or None,
                'planning_note': _planning_status(planning)}

    # 普通消息：规划模式下始终保持在规划父图。
    if not planning:
        return {**common, 'planning': None, 'planning_note': None,
                'task_results': _empty_results(input_id)}
    status = planning.get('status')
    if status == 'draft':
        supplement = _last_user_text(state)
        supplements = list(planning.get('supplements') or [])
        if supplement and supplement not in supplements:
            supplements.append(supplement)
        planning = {**planning, 'supplements': supplements,
                    'resources': _merge_resources(planning.get('resources'), supplement),
                    'resolved': {}}
        return {**common, 'planning': planning}
    if status == 'running':
        return {**common, 'planning': planning,
                'planning_note': '计划正在执行；请等待完成，或使用 /cancel 后重新规划。'}
    if status in {'completed', 'failed', 'cancelled'}:
        new_goal = _last_user_text(state).strip()
        if not new_goal:
            return {**common, 'planning': planning,
                    'planning_note': '当前计划已结束；发送新的规划目标可重新规划，或 /plan exit 退出规划模式。'}
        created = _new_plan(new_goal, input_id)
        return {**common, 'planning': created, 'planning_note': None,
                'task_results': _empty_results(created['plan_id'])}
    # 未知状态：保留规划工作区并提示，不静默回到普通 Agent。
    return {**common, 'planning': {**planning, 'active': False},
            'planning_note': '当前计划状态无法执行；请发送新的规划目标，或 /plan exit 退出。'}


async def _classify_plan_intent(state, planning, *, model, config, settings):
    """一次短模型调用；失败时安全回退到 continue，不执行计划。"""
    timeout = float(getattr(settings, 'intent_timeout_seconds', 15))
    base = dict(config or {})
    base['metadata'] = {**(base.get('metadata') or {}), 'assistant_phase': 'plan_intent'}
    try:
        async with asyncio.timeout(timeout):
            result = await invoke_structured(
                model, PlanIntent, _intent_prompt(state, planning), config=base)
        event('plan.intent', intent=result.intent, confidence=result.confidence)
        return result
    except Exception as error:
        raise_if_fatal(error)
        logger.warning('Plan intent classification failed: category=%s',
                       type(error).__name__)
        event('plan.intent_failed', category=type(error).__name__)
        return PlanIntent(intent='continue', confidence=0.0)


_INTENT_ACTIONS = {'execute': 'run', 'exit': 'exit', 'status': 'status'}


async def _intake_with_intent(state, config, *, model, settings):
    """显式动作优先；只有规划模式内的自然语言才调用一次意图模型。"""
    action = state.get('planning_action')
    planning = _normalize_planning(state.get('planning'))
    if action is None and planning:
        intent = await _classify_plan_intent(
            state, planning, model=model, config=config, settings=settings)
        threshold = float(getattr(settings, 'intent_confidence', 0.8))
        if intent.intent != 'continue' and intent.confidence >= threshold:
            action = _INTENT_ACTIONS[intent.intent]
    return _intake(state, action=action)


def _route_intake(state):
    if state.get('planning_note'):
        return 'finalizer'
    planning = state.get('planning')
    if not planning:
        return 'normal'
    status = planning.get('status')
    if status == 'draft':
        return 'planner'
    if status == 'running':
        return 'scheduler'
    if status in {'completed', 'failed', 'cancelled'}:
        return 'finalizer'
    return 'normal'
