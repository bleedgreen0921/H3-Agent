"""可信身份与确定性的时区转换，不依赖外部联系人服务。"""
from datetime import datetime, timezone as utc_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from ai_assistant.tools.context import ToolContext


def convert_datetime(value: str, timezone: str = 'Asia/Shanghai') -> dict:
    try:
        zone = ZoneInfo(timezone)
        parsed = datetime.fromisoformat(value)
        all_day = len(value) == 10
        if parsed.tzinfo is None:
            # UTC 往返校验排除夏令时跳过或重复的本地时间。
            candidates = {candidate.utcoffset(): candidate for fold in (0, 1)
                          if (candidate := parsed.replace(tzinfo=zone, fold=fold))
                          .astimezone(utc_timezone.utc).astimezone(zone).replace(tzinfo=None) == parsed}
            if len(candidates) != 1:
                raise ValueError('ambiguous or nonexistent time')
            parsed = next(iter(candidates.values()))
        else:
            parsed = parsed.astimezone(zone)
        return {'datetime': parsed.isoformat(), 'timezone': timezone,
                'timestamp': str(int(parsed.timestamp() * 1000)), 'is_all_day': all_day}
    except (ValueError, OverflowError, ZoneInfoNotFoundError):
        raise ToolException('日期时间或 IANA 时区无效；夏令时重复/不存在的时间必须提供明确偏移。') from None


@tool
async def resolve_datetime(value: str, runtime: ToolRuntime[ToolContext], timezone: str | None = None) -> dict:
    """Convert an explicit ISO datetime (or all-day YYYY-MM-DD) to milliseconds. Clarify unclear dates first."""
    if runtime.context.caller and runtime.context.caller.channel == 'feishu_group' and timezone is None:
        raise ToolException('群聊不继承主人默认时区；请先明确 IANA 时区。')
    result = convert_datetime(value, timezone or runtime.context.timezone)
    runtime.context.resolved_datetimes.add((result['timestamp'], result['is_all_day']))
    return result


@tool
async def get_current_identity(runtime: ToolRuntime[ToolContext]) -> dict:
    """Read the configured owner's identity, timezone and current time in a trusted owner private chat."""
    context = runtime.context
    if not context.caller or context.caller.channel != 'feishu_private' or not context.caller.owner or not context.owner_open_id:
        raise ToolException('当前入口没有主人私聊身份权限。')
    return {'open_id': context.owner_open_id, 'timezone': context.timezone,
            'current_time': datetime.now(ZoneInfo(context.timezone)).isoformat()}
