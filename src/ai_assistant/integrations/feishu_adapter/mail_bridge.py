"""邮件来源任务的飞书身份桥接。"""
from langchain_core.tools import ToolException


def email_source_key(entry, arguments, context):
    extra = arguments.get('data', {}).get('extra')
    if not isinstance(extra, str) or not extra.startswith('email-task:'):
        return None
    from ai_assistant.tools.mail import allowed
    if (entry.original_name != 'task_v2_task_create' or entry.settings.transport != 'feishu'
            or entry.settings.identity != 'user' or not allowed(context.caller)
            or 'mail' not in context.services):
        raise ToolException('邮件来源任务仅允许主人私聊通过 user 身份创建。')
    return context.services['mail'].validate(arguments)
