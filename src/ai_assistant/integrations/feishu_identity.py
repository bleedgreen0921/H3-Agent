"""飞书可信主人身份；不从模型参数或渠道名称推导私人权限。"""
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from ai_assistant.tools.context import ToolContext


@tool
async def get_current_identity(runtime: ToolRuntime[ToolContext], include_open_id: bool = False) -> dict:
    """Read the configured owner's identity, timezone and current time in a trusted owner private chat.

    默认不返回账号 ID；只有后续工具调用确实需要时才把 include_open_id 设为 true，
    最终回答仍不应复述该 ID。
    """
    context = runtime.context
    caller = context.caller
    if (caller is None or caller.channel != 'feishu_private' or caller.scope != 'private'
            or not caller.owner or not context.services['feishu'].owner_open_id):
        raise ToolException('当前入口没有主人私聊身份权限。')
    result = {'identity': 'owner', 'timezone': context.timezone,
              'current_time': datetime.now(ZoneInfo(context.timezone)).isoformat()}
    if include_open_id:
        result['open_id'] = context.services['feishu'].owner_open_id
    return result


def identity_tools(caller):
    if caller.channel == 'feishu_private' and caller.scope == 'private' and caller.owner:
        return [get_current_identity.model_copy(update={'handle_tool_error': True})]
    return []
