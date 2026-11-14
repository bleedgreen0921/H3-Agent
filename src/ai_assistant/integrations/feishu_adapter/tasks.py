"""飞书任务业务规则：参数保护、回执核验和时间说明。"""
from ai_assistant.integrations.feishu_tasks import (
    TASK_WRITES, prepare_task, preflight_task, task_data, task_model_result, verification,
)


def guidance(entries):
    """存在任务工具时提供时间、负责人和回执处理提示。"""
    names = {entry.original_name for entry in entries if entry.settings.transport == 'feishu'}
    return ('任务时间先结合可信当前日期明确为 ISO，再调用 resolve_datetime；写入时间须来自本轮转换，日期不明确先澄清。'
            '飞书任务提醒 relative_fire_minute 只能按截止时间 due 解释；创建任务时设置 reminders 必须同时提供 due。'
            '如果用户要在会议或考试开始前提醒，不要改造任务 due，应优先使用日历日程，或先向用户确认。'
            '负责人由系统按部署策略绑定；owner-only 下不要传 members，也不要调用 addMembers。'
            '身份使用可信身份工具，不从姓名推导 ID；身份工具默认不返回 open_id，只有后续写入确实需要时才请求，且不在最终回答中复述。'
            '主人飞书私聊创建时未传 members 遵循默认负责人配置，'
            '显式空列表不覆盖；群聊须明确时区与身份。patch 只更新 update_fields，负责人使用 addMembers。'
            '最终回答不要复述 open_id、union_id、邮箱或手机号；用“你/负责人/成员”表述。'
            '回执核验仅比较本次发送字段与本次接口返回；未返回字段表示未确认，接口接受不证明提醒实际送达。'
            if any(name.startswith('task_v2_task_') for name in names) else '')


def task_write(entry):
    """判断当前条目是否为已识别的飞书任务写工具。"""
    return entry.settings.transport == 'feishu' and entry.original_name in TASK_WRITES


def preflight(entry, arguments, context):
    """任务写入前的跨字段校验；只读，不触碰远端。"""
    if getattr(entry.settings, 'transport', None) == 'feishu' and entry.original_name.startswith('task_v2_'):
        preflight_task(context, entry, arguments)


def prepare(context, entry, arguments):
    """任务写入委托业务规则准备参数和预期字段。"""
    return prepare_task(context, entry, arguments) if task_write(entry) else {}


async def verify(entry, operation, arguments, expected, result, context, config=None):
    """根据本次接口返回核验任务字段，并保护核验回执。"""
    from langchain_core.messages import ToolMessage
    if not task_write(entry):
        return result, None
    detail = verification(expected, task_data(result), arguments.get('path', {}).get('task_guid'))

    # 调用未成功时不能将字段比较视为成功验证。
    if operation['status'] != 'succeeded':
        detail['verification'] = 'unverified'
    if isinstance(result, ToolMessage):
        result = result.model_copy(update={
            'status': 'error' if detail['verification'] == 'mismatch' else result.status,
            'artifact': {**(result.artifact if isinstance(result.artifact, dict) else {}),
                         'protected_receipt': True, 'verification': detail}})
    return result, detail


async def after_read(entry, arguments, result, context):
    """对任务工具消息附加业务所需的模型展示信息。"""
    from langchain_core.messages import ToolMessage
    if (entry.settings.transport == 'feishu' and entry.original_name.startswith('task_v2_task_')
            and isinstance(result, ToolMessage)):
        result = task_model_result(result, context)
    return result
