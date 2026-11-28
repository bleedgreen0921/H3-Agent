"""规划父图最终输出：状态归类、用户正文和效果汇总。"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ai_assistant.agents.planning.models import BLOCKING
from ai_assistant.agents.planning.rendering import _render_plan, _render_results
from ai_assistant.agents.planning.results import (
    _dedupe_effects, _effective_status, _replacement_map,
)
from ai_assistant.observability import event
from ai_assistant.tools.context import ToolContext


def _finalizer(state, runtime: Runtime[ToolContext]):
    planning = dict(state.get('planning') or {})
    note = state.get('planning_note')
    results = state.get('task_results') or {}
    items = results.get('items') or {}
    effects = list(runtime.context.effect_facts)
    for value in items.values():
        effects.extend(value.get('effects') or [])
    effects = _dedupe_effects(effects)

    replacements = _replacement_map(planning)
    plan = dict(planning.get('plan') or {})
    questions = [str(item).strip() for item in (plan.get('questions') or [])
                 if str(item).strip()]
    for value in items.values():
        if _effective_status(value.get('id'), items, replacements) == 'needs_input':
            questions.extend(str(item).strip() for item in (value.get('questions') or ())
                             if str(item).strip())
    questions = list(dict.fromkeys(questions))
    if questions:
        # 缺少必要输入：保留计划，停止执行下游，等待用户补充后重新规划。
        plan['questions'] = questions
        planning = {**planning, 'status': 'draft', 'plan': plan,
                    'questions': questions}
        text = _render_plan(planning)
        execution = {'status': 'needs_clarification', 'phase': 'plan'}
        keep = planning
        diagnostics = [
            {'status': value.get('status'),
             'summary': (value.get('error') or {}).get('message')}
            for value in items.values()
            if _effective_status(value.get('id'), items, replacements) in BLOCKING
        ]
    elif note and planning.get('status') != 'failed':
        text = note
        if planning.get('status') == 'draft' and planning.get('plan') and '计划：' not in note:
            text += '\n\n' + _render_plan(planning)
        status = planning.get('status')
        execution_status = ('complete' if status in {'draft', 'running', 'completed'}
                            else 'cancelled' if status == 'cancelled' else 'failed')
        execution = {'status': execution_status, 'phase': 'plan'}
        keep = planning or None
        diagnostics = []
    elif planning.get('status') == 'draft':
        text = _render_plan(planning)
        execution = {'status': 'complete', 'phase': 'plan'}
        keep = planning
        diagnostics = []
    else:
        statuses = {_effective_status(task_id, items, replacements)
                    for task_id in items}
        if planning.get('status') == 'failed':
            status = 'failed'
        elif 'unknown' in statuses:
            status = 'unknown'
        elif 'cancelled' in statuses:
            status = 'cancelled'
        elif statuses & {'failed', 'blocked'}:
            status = 'failed'
        else:
            status = 'complete'
        text = _render_results(planning, results, note, effects)
        execution = {'status': status, 'phase': 'plan'}
        plan_status = {'complete': 'completed', 'failed': 'failed',
                       'cancelled': 'cancelled', 'unknown': 'failed'}[status]
        keep = {**planning, 'status': plan_status, 'active': False}
        diagnostics = [
            {'status': value.get('status'), 'summary': (value.get('error') or {}).get('message')}
            for value in items.values()
            if _effective_status(value.get('id'), items, replacements) in BLOCKING
        ]
    event('plan.finished', status=(execution or {}).get('status'), plan_id=planning.get('plan_id'),
          task_count=len(items))
    return {
        'messages': [AIMessage(content=text)],
        'execution': execution,
        'diagnostics': diagnostics,
        'planning': keep,
        'planning_note': None,
        'dispatch': [],
        'effect_facts': effects,
    }
