"""飞书 MCP 适配器统一入口，组合文档、任务、日历和传输处理。"""

# 引入各业务模块的钩子，由下方统一接口按工具类型分派。
from ai_assistant.integrations.feishu_adapter.calendar import (
    after_read as calendar_after_read,
    guidance as calendar_guidance,
    preflight as calendar_preflight,
    prepare as calendar_prepare,
    verify as calendar_verify,
)
from ai_assistant.integrations.feishu_adapter.docs import (
    display_name as doc_display_name,
    guidance as docs_guidance,
    preflight as docs_preflight,
    references as docs_references,
    unfinished_document_notice,
)
from ai_assistant.integrations.feishu_adapter.mail_bridge import email_source_key
from ai_assistant.integrations.feishu_adapter.projection import project, project_output
from ai_assistant.integrations.feishu_adapter.tasks import (
    after_read as task_after_read, guidance as task_guidance,
    preflight as task_preflight, prepare as prepare_task, task_write,
    verify as task_verify,
)
from ai_assistant.integrations.feishu_adapter.transport import IDEMPOTENCY, idempotency_path, interpret


def guidance(entries, context):
    """合并文档、任务和日历模块提供的工具使用提示。"""
    return docs_guidance(entries) + task_guidance(entries) + calendar_guidance(entries)


def display_name(entry):
    """使用工具原始名称生成面向用户的显示名称。"""
    return doc_display_name(getattr(entry, 'original_name', ''))


def references(entry, result):
    """从结果中提取文档相关引用，供调用流程记录和绑定。"""
    return docs_references(entry, result)


def is_calendar(entry):
    """根据飞书工具原始名称识别日历接口。"""
    return getattr(entry, 'original_name', '').startswith('calendar_v4_')


def preflight(entry, arguments, context):
    """远端调用前按日历或其他工具执行对应的业务预检。"""
    if is_calendar(entry):
        calendar_preflight(entry, arguments, context)
    else:
        docs_preflight(entry, arguments, context)
        task_preflight(entry, arguments, context)


async def prepare(entry, arguments, context, state, config=None):
    """准备调用参数与预期结果，统一返回预期值和可选提前结果。"""
    # 邮件来源关联先处理，再按日历或任务适配器准备本次调用。
    email_source_key(entry, arguments, context)
    if is_calendar(entry):
        return calendar_prepare(entry, arguments, context), None
    return prepare_task(context, entry, arguments), None


async def verify(entry, operation, arguments, expected, result, context, config=None):
    """将写入结果交给对应业务模块核对并补充回执信息。"""
    if is_calendar(entry):
        return calendar_verify(entry, operation, arguments, expected, result, context, config)
    return await task_verify(entry, operation, arguments, expected, result, context, config)


async def after_read(entry, arguments, result, context):
    """调用完成后交给对应模块处理读取结果。"""
    if is_calendar(entry):
        return calendar_after_read(entry, arguments, result, context)
    return await task_after_read(entry, arguments, result, context)


# 声明统一适配器入口，同时导出直接复用的传输、投影及文档辅助函数。
__all__ = ['IDEMPOTENCY', 'idempotency_path', 'guidance', 'task_write', 'preflight', 'prepare',
           'interpret', 'verify', 'after_read', 'email_source_key', 'project', 'project_output',
           'display_name', 'references', 'unfinished_document_notice']
