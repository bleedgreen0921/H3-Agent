"""飞书可信主人身份；不从模型参数或渠道名称推导私人权限。"""
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from ai_assistant.tools.context import ToolContext


@tool
async def get_current_identity(runtime: ToolRuntime[ToolContext]) -> dict:
    """Read the configured owner's identity, timezone and current time in a trusted owner private chat."""
    context = runtime.context
    caller = context.caller
    if (caller is None or caller.channel != 'feishu_private' or caller.scope != 'private'
            or not caller.owner or not context.owner_open_id):
        raise ToolException('当前入口没有主人私聊身份权限。')
    return {'open_id': context.owner_open_id, 'timezone': context.timezone,
            'current_time': datetime.now(ZoneInfo(context.timezone)).isoformat()}


def identity_tools(caller):
    if caller.channel == 'feishu_private' and caller.scope == 'private' and caller.owner:
        return [get_current_identity.model_copy(update={'handle_tool_error': True})]
    return []
