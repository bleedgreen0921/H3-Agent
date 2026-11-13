"""飞书日历事件规则：默认日历、可信时间、开始前提醒与回执核验。"""
from __future__ import annotations

import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from ai_assistant.integrations.feishu_tasks import payload
from ai_assistant.integrations.feishu_temporal import calendar_timestamp_seconds, time_view
from ai_assistant.integrations.tool_policy import PreflightIssue, ToolPreflightError

EVENT_WRITES = {
    'calendar_v4_calendarEvent_create',
    'calendar_v4_calendarEvent_patch',
    'calendar_v4_calendarEvent_delete',
}
EVENT_TIME_FIELDS = ('start_time', 'end_time')


def guidance(entries):
    names = {entry.original_name for entry in entries if entry.settings.transport == 'feishu'}
    return ('会议、考试、面试和日程优先使用日历事件；任务只表示截止事项。'
            '日历事件的 start_time/end_time 必须来自本轮 resolve_datetime，适配器会转换为接口需要的秒级 timestamp。'
            'start_time 必须早于 end_time；reminders.minutes 相对开始时间，60 表示开始前 60 分钟。'
            '创建事件优先使用配置的默认 calendar_id；没有配置时先查询日历并选择，不要猜测。'
            '最终回答不要复述 organizer、attendee 的 open_id/union_id 或邮箱。'
            if any(name.startswith('calendar_v4_') for name in names) else '')


def is_calendar(entry):
    return getattr(entry, 'original_name', '').startswith('calendar_v4_')


def event_data(arguments):
    return arguments.get('data', {}) if isinstance(arguments, dict) else {}


def event_payload(result):
    value = payload(result)
    if not isinstance(value, dict):
        return {}
    data = value.get('data', value)
    event = data.get('event', data) if isinstance(data, dict) else {}
    return event if isinstance(event, dict) else {}


def _calendar_id(arguments, context):
    path = arguments.get('path', {}) if isinstance(arguments, dict) else {}
    if isinstance(path, dict) and path.get('calendar_id'):
        return path['calendar_id']
    service = (getattr(context, 'services', None) or {}).get('feishu')
    return getattr(service, 'default_calendar_id', None)


def _calendar_time(value):
    if not isinstance(value, dict):
        return None
    return value.get('timestamp') or value.get('date')


def preflight(entry, arguments, context):
    name = entry.original_name
    if name not in EVENT_WRITES:
        return
    if not _calendar_id(arguments, context):
        raise ToolPreflightError(PreflightIssue(
            code='calendar.id_required',
            message='创建日历事件需要 calendar_id；请先查询日历并指定，或配置默认日历。',
            field='path.calendar_id'))
    if name == 'calendar_v4_calendarEvent_delete':
        return
    data = event_data(arguments)
    if not isinstance(data, dict):
        return
    start, end = data.get('start_time'), data.get('end_time')
    if name == 'calendar_v4_calendarEvent_create':
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise ToolPreflightError(PreflightIssue(
                code='calendar.time_required',
                message='日历事件必须同时提供 start_time 和 end_time。',
                field='data.start_time'))
    for field, value in (('start_time', start), ('end_time', end)):
        if not isinstance(value, dict):
            continue
        if value.get('date') and not value.get('timestamp'):
            continue
        if value.get('timestamp') is None:
            raise ToolPreflightError(PreflightIssue(
                code='calendar.time_required',
                message=f'{field} 必须提供 timestamp 或 date。',
                field=f'data.{field}'))
        if calendar_timestamp_seconds(context, value) is None:
            raise ToolPreflightError(PreflightIssue(
                code='calendar.time_not_resolved',
                message=f'{field} 未匹配本轮可信时间转换；请先调用 resolve_datetime。',
                field=f'data.{field}'))
    if isinstance(start, dict) and isinstance(end, dict):
        if bool(start.get('date')) != bool(end.get('date')):
            raise ToolPreflightError(PreflightIssue(
                code='calendar.time_flags_mismatch',
                message='start_time 和 end_time 必须同为 date 或同为 timestamp。',
                field='data.start_time'))
        if start.get('timestamp') and end.get('timestamp'):
            if int(calendar_timestamp_seconds(context, start)) >= int(calendar_timestamp_seconds(context, end)):
                raise ToolPreflightError(PreflightIssue(
                    code='calendar.start_after_end',
                    message='日历开始时间必须早于结束时间。',
                    field='data.start_time'))
    reminders = data.get('reminders')
    if reminders is not None:
        if not isinstance(reminders, list) or any(
                not isinstance(item, dict) or not isinstance(item.get('minutes'), (int, float))
                for item in reminders):
            raise ToolPreflightError(PreflightIssue(
                code='calendar.reminder_invalid',
                message='reminders 必须是包含 minutes 的列表；minutes 相对事件开始时间。',
                field='data.reminders'))


def prepare(entry, arguments, context):
    if not is_calendar(entry):
        return {}
    name = entry.original_name
    if name in EVENT_WRITES:
        calendar_id = _calendar_id(arguments, context)
        if calendar_id:
            arguments.setdefault('path', {})['calendar_id'] = calendar_id
    data = event_data(arguments)
    if name in EVENT_WRITES and isinstance(data, dict):
        for field in EVENT_TIME_FIELDS:
            value = data.get(field)
            if isinstance(value, dict) and value.get('timestamp') is not None and not value.get('date'):
                seconds = calendar_timestamp_seconds(context, value)
                if seconds is None:
                    raise ToolException(f'{field} 未匹配本轮可信时间转换；请先调用 resolve_datetime。')
                value['timestamp'] = seconds
    return {}


def _verify_time(expected, actual):
    if not isinstance(expected, dict):
        return None
    if expected.get('date'):
        return expected.get('date') == (actual or {}).get('date') if isinstance(actual, dict) else False
    return isinstance(actual, dict) and str(expected.get('timestamp')) == str(actual.get('timestamp'))


def verify(entry, operation, arguments, expected, result, context, config=None):
    if entry.original_name not in EVENT_WRITES or entry.original_name == 'calendar_v4_calendarEvent_delete':
        return result, None
    event = event_payload(result)
    expected_data = event_data(arguments)
    checked, unchecked, issues = [], [], []
    for key in ('summary', 'start_time', 'end_time', 'reminders'):
        if key not in expected_data:
            continue
        if key not in event:
            unchecked.append(key)
            continue
        if key in EVENT_TIME_FIELDS:
            same = _verify_time(expected_data[key], event[key])
        elif key == 'reminders':
            wanted = expected_data[key] or []
            got = event[key] or []
            same = sorted(str(item.get('minutes')) for item in wanted) == sorted(
                str(item.get('minutes')) for item in got)
        else:
            same = expected_data[key] == event[key]
        if same:
            checked.append(key)
        else:
            issues.append(f'{key} 与发送值不符')
    detail = {'verification': 'mismatch' if issues else 'unverified' if unchecked else 'passed',
              'verification_source': '本次接口返回', 'checked_fields': checked,
              'unchecked_fields': unchecked,
              'issues': issues + [f'{key} 未核对或未返回' for key in unchecked]}
    if operation['status'] != 'succeeded':
        detail['verification'] = 'unverified'
    if isinstance(result, ToolMessage):
        result = result.model_copy(update={
            'status': 'error' if detail['verification'] == 'mismatch' else result.status,
            'artifact': {**(result.artifact if isinstance(result.artifact, dict) else {}),
                         'protected_receipt': True, 'verification': detail}})
    return result, detail


def after_read(entry, arguments, result, context):
    if entry.original_name not in EVENT_WRITES or not isinstance(result, ToolMessage):
        return result
    event = event_payload(result)
    view = time_view(event, context.timezone, EVENT_TIME_FIELDS, unit='s')
    if view is None or (result.artifact or {}).get('calendar_model_view'):
        return result
    blocks = list(result.content) if isinstance(result.content, list) else [{'type': 'text', 'text': result.content}]
    blocks.append({'type': 'text', 'text': json.dumps({'calendar_time_annotation': view}, ensure_ascii=False)})
    return result.model_copy(update={'content': blocks, 'artifact': {
        **(result.artifact if isinstance(result.artifact, dict) else {}),
        'calendar_model_view': True}})


__all__ = ['EVENT_WRITES', 'after_read', 'event_data', 'event_payload', 'guidance',
           'is_calendar', 'preflight', 'prepare', 'verify']
