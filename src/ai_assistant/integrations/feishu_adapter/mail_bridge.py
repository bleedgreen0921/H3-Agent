"""邮件来源任务的飞书身份桥接。"""
from langchain_core.tools import ToolException


def email_source_key(entry, arguments, context):
    """验证邮件来源任务的调用身份，并返回经过校验的来源键。"""
    extra = arguments.get('data', {}).get('extra')
    if not isinstance(extra, str) or not extra.startswith('email-task:'):
        return None

    # 邮件来源只允许有邮件服务的主人私聊以飞书 user 身份创建任务。
    from ai_assistant.tools.mail import allowed
    if (entry.original_name != 'task_v2_task_create' or entry.settings.transport != 'feishu'
            or entry.settings.identity != 'user' or not allowed(context.caller)
            or 'mail' not in context.services):
        raise ToolException('邮件来源任务仅允许主人私聊通过 user 身份创建。')
    return context.services['mail'].validate(arguments)
