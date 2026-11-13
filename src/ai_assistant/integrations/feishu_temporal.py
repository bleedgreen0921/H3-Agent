"""飞书任务/日历共享的可信时间解析与本地展示。"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def _timestamp(value):
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def resolved_timestamp_ms(context, value, *, unit='ms'):
    """验证时间来自本轮 resolve_datetime；日历使用秒级时间戳。"""
    timestamp = _timestamp((value or {}).get('timestamp') if isinstance(value, dict) else value)
    if timestamp is None:
        return None
    candidates = {timestamp} if unit == 'ms' else {timestamp * 1000, timestamp}
    resolved = {int(key[0]) for key in context.resolved_datetimes}
    match = next((item for item in candidates if item in resolved), None)
    return match


def calendar_timestamp_seconds(context, value):
    """把本轮可信毫秒时间转换为日历 API 的秒级 timestamp。"""
    timestamp = _timestamp((value or {}).get('timestamp') if isinstance(value, dict) else value)
    if timestamp is None:
        return None
    resolved = {int(key[0]) for key in context.resolved_datetimes}
    if timestamp in resolved:
        return str(timestamp // 1000)
    if timestamp * 1000 in resolved:
        return str(timestamp)
    return None


def local_iso(timestamp_ms, timezone):
    try:
        return datetime.fromtimestamp(int(timestamp_ms) / 1000, ZoneInfo(timezone)).isoformat()
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None


def time_view(value, timezone, fields, *, unit='ms', report_invalid=False):
    """给任务/日历共用的本地时间说明；不修改原对象。"""
    if not isinstance(value, dict):
        return None
    view = {'timezone': timezone}
    for field in fields:
        item = value.get(field)
        if not isinstance(item, dict):
            continue
        current = dict(item)
        timestamp = _timestamp(current.get('timestamp'))
        if timestamp is None:
            if report_invalid:
                current['display_error'] = '时间戳无法转换，保留原值'
        else:
            local = local_iso(timestamp * 1000 if unit == 's' else timestamp, timezone)
            if local:
                current['local_datetime'] = local
            elif report_invalid:
                current['display_error'] = '时间戳无法转换，保留原值'
        view[field] = current
    return view or None


__all__ = ['calendar_timestamp_seconds', 'local_iso', 'resolved_timestamp_ms', 'time_view']
