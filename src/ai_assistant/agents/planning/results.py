"""任务结果字典、替换关系和效果去重。"""
from __future__ import annotations

import json
import re

from ai_assistant.agents.planning.models import BLOCKING


def _empty_results(plan_id: str | None):
    return {'plan_id': plan_id or '', 'items': {}}


def _task_result(task_id, status, summary, *, output=None, references=None,
                 error=None, effects=None, attempts=1, questions=None):
    return {
        'id': task_id,
        'status': status,
        # summary 只用于短状态；output 供依赖任务和最终交付使用。
        'summary': (summary or '')[:600],
        'output': (summary or '') if output is None else output,
        'references': list(references or ()),
        'questions': [str(item) for item in (questions or ())],
        'error': error,
        'effects': effects or [],
        'attempts': attempts,
        'report_version': 2,
    }


def _replacement_map(planning):
    result = {}
    for task in ((planning or {}).get('plan') or {}).get('tasks') or []:
        for old_id in task.get('replaces') or []:
            result[old_id] = task.get('id')
    return result


def _effective_status(task_id, items, replacements):
    value = items.get(task_id) or {}
    status = value.get('status')
    replacement = replacements.get(task_id)
    if status in BLOCKING and replacement and (items.get(replacement) or {}).get('status') == 'done':
        return 'superseded'
    return status


def _dedupe_effects(facts):
    seen = set()
    result = []
    for item in facts or []:
        if not isinstance(item, dict):
            continue
        key = item.get('effect_id') or json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def _brief_text(value, limit=600):
    """把旧格式或自由文本压成用户可读摘要；绝不回放完整产物。"""
    text = str(value or '').strip()
    if not text:
        return ''
    text = re.sub(r'```.*?```', '[完整产物已保存在任务 output]', text, flags=re.S)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('|') or set(line) <= {'-', ':', '|', ' '}:
            continue
        line = re.sub(r'^#{1,6}\s*', '', line)
        lines.append(line)
    text = ' '.join(lines)
    parts = re.split(r'(?<=[。！？.!?])\s*', text)
    result = ''
    for part in parts:
        if not part:
            continue
        if len(result) + len(part) > limit:
            break
        result += part
    return (result or text)[:limit]


def _task_summary(value):
    """旧结果的安全短摘要；新结果直接使用 summary。"""
    summary = str((value or {}).get('summary') or '').strip()
    if (value or {}).get('report_version') == 2:
        return summary[:600]
    return _brief_text(summary, 600)
