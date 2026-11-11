"""飞书任务业务规则：参数保护、回执核验和时间说明。"""
from ai_assistant.integrations.feishu_tasks import (
    TASK_WRITES, prepare_task, task_data, task_model_result, verification,
)


def guidance(entries):
    names = {entry.original_name for entry in entries if entry.settings.transport == 'feishu'}
    return ('任务时间先结合可信当前日期明确为 ISO，再调用 resolve_datetime；写入时间须来自本轮转换，日期不明确先澄清。'
            '身份使用可信身份工具，不从姓名推导 ID。主人飞书私聊创建时未传 members 遵循默认负责人配置，'
            '显式空列表不覆盖；群聊须明确时区与身份。patch 只更新 update_fields，负责人使用 addMembers。'
            '回执核验仅比较本次发送字段与本次接口返回；未返回字段表示未确认，接口接受不证明提醒实际送达。'
            if any(name.startswith('task_v2_task_') for name in names) else '')


def task_write(entry):
    return entry.settings.transport == 'feishu' and entry.original_name in TASK_WRITES


def prepare(context, entry, arguments):
    return prepare_task(context, entry, arguments) if task_write(entry) else {}


async def verify(entry, operation, arguments, expected, result, context, config=None):
    from langchain_core.messages import ToolMessage
    if not task_write(entry):
        return result, None
    detail = verification(expected, task_data(result), arguments.get('path', {}).get('task_guid'))
    if operation['status'] != 'succeeded':
        detail['verification'] = 'unverified'
    if isinstance(result, ToolMessage):
        result = result.model_copy(update={
            'status': 'error' if detail['verification'] == 'mismatch' else result.status,
            'artifact': {**(result.artifact if isinstance(result.artifact, dict) else {}),
                         'protected_receipt': True, 'verification': detail}})
    return result, detail


async def after_read(entry, arguments, result, context):
    from langchain_core.messages import ToolMessage
    if (entry.settings.transport == 'feishu' and entry.original_name.startswith('task_v2_task_')
            and isinstance(result, ToolMessage)):
        result = task_model_result(result, context)
    return result
