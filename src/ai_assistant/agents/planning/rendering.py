"""计划状态和任务结果的可读文本渲染。"""
from __future__ import annotations

from ai_assistant.agents.planning.results import (
    _effective_status, _replacement_map, _task_summary,
)


def _render_plan(planning):
    plan = (planning or {}).get('plan') or {}
    goal = plan.get('goal') or planning.get('goal') or '未命名计划'
    status = planning.get('status') or 'draft'
    labels = {'draft': '草稿', 'running': '执行中', 'completed': '已完成',
              'failed': '未完成', 'cancelled': '已取消'}
    lines = [f'计划：{goal}', f'状态：{labels.get(status, status)}']
    resolved = (planning or {}).get('resolved') or {}
    resources = {item['id']: item for item in (planning or {}).get('resources') or []}
    if resources:
        lines.append('目标资源：')
        for item in resources.values():
            target = item.get('url') or item.get('token') or item.get('label')
            identity = f"（{item.get('identity')}）" if item.get('identity') else ''
            lines.append(f"- {item['id']} {target}{identity}")
    questions = [str(item).strip() for item in (plan.get('questions') or []) if str(item).strip()]
    if questions:
        lines.append('需要补充：')
        lines.extend(f'{index}. {question}' for index, question in enumerate(questions, 1))
    for index, task in enumerate(plan.get('tasks') or [], 1):
        deps = '、'.join(task.get('deps') or []) or '无'
        resolution = resolved.get(task.get('id')) or {}
        effect_value = resolution.get('effect') or task.get('effect') or task.get('side_effect')
        effect = '写' if effect_value == 'write' else '读'
        refs = '、'.join(task.get('resources') or []) or '无'
        lines.append(f'{index}. [{task.get("id")}] {task.get("description")}')
        lines.append(f'   依赖：{deps}；类型：{effect}；资源：{refs}')
        if task.get('expected_output'):
            lines.append(f'   预期：{task["expected_output"]}')
    if status == 'draft':
        if questions:
            lines.append('请直接回复补充内容；补充后会重新规划，/plan run 暂不可用。')
        else:
            lines.append('回复补充可调整计划；/plan run 开始执行，/plan exit 退出。')
    elif status in {'completed', 'failed', 'cancelled'}:
        lines.append('当前计划已结束；发送新的规划目标可重新规划，或 /plan exit 退出规划模式。')
    return '\n'.join(lines)


def _planning_status(planning):
    if not planning:
        return '当前不在规划模式。'
    return _render_plan(planning)


def _render_results(planning, results, note=None, effects=()):
    plan = (planning or {}).get('plan') or {}
    items = (results or {}).get('items') or {}
    replacements = _replacement_map(planning)
    tasks = plan.get('tasks') or []
    task_ids = {task.get('id') for task in tasks if task.get('id')}
    # 叶子任务的完整 output 是最终交付物；中间任务 output 只用于依赖。
    leaf_ids = [task.get('id') for task in tasks if task.get('id') and not any(
        task.get('id') in (other.get('deps') or []) for other in tasks)]
    if not leaf_ids:
        leaf_ids = list(task_ids)
    ordered = [items[task_id] for task_id in leaf_ids if task_id in items]
    ordered.extend(value for task_id, value in items.items() if task_id not in task_ids)
    done = [value for value in ordered
            if _effective_status(value.get('id'), items, replacements) == 'done']
    other = [value for value in ordered
             if _effective_status(value.get('id'), items, replacements) not in {'done', 'superseded'}]
    lines = []
    for value in done:
        output = str(value.get('output') or '').strip()
        if output:
            # 叶子任务的完整产物就是最终交付物；summary 只作为无产物时的回退。
            lines.append((f"## {value.get('id')}\n{output}" if len(done) > 1 else output))
            continue
        summary = _task_summary(value)
        if summary:
            lines.append(summary)
    if other:
        lines.append('未完成或待核实：')
        for value in other:
            summary = _task_summary(value)
            reason = (value.get('error') or {}).get('message') or ''
            detailed = summary or reason or '任务未完成。'
            if reason and reason not in detailed:
                detailed += f'（{reason}）'
            lines.append(f'- {detailed}')
    if note:
        lines.append('计划调整说明：' + note)
    if not items:
        lines.append('没有可执行的任务。')
    text = '\n'.join(lines).strip()
    links = []
    for resource in (planning or {}).get('resources') or []:
        value = resource.get('url') if isinstance(resource, dict) else None
        if isinstance(value, str) and value and value not in text and value not in links:
            links.append(value)
    for effect in effects or ():
        if not isinstance(effect, dict):
            continue
        for key in ('url', 'link'):
            value = effect.get(key)
            if isinstance(value, str) and value and value not in text and value not in links:
                links.append(value)
    for value in ordered:
        for ref in value.get('references') or ():
            if isinstance(ref, str) and ref.startswith(('http://', 'https://')):
                if ref not in text and ref not in links:
                    links.append(ref)
    if links:
        text += '\n链接：' + '、'.join(links)
    return text or ('已完成。' if done else '未完成。')
