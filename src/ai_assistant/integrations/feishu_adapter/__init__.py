"""飞书 MCP adapter facade：transport/docs/tasks/calendar/mail_bridge 的组合。"""
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
    return docs_guidance(entries) + task_guidance(entries) + calendar_guidance(entries)


def display_name(entry):
    return doc_display_name(getattr(entry, 'original_name', ''))


def references(entry, result):
    return docs_references(entry, result)


def is_calendar(entry):
    return getattr(entry, 'original_name', '').startswith('calendar_v4_')


def preflight(entry, arguments, context):
    if is_calendar(entry):
        calendar_preflight(entry, arguments, context)
    else:
        docs_preflight(entry, arguments, context)
        task_preflight(entry, arguments, context)


async def prepare(entry, arguments, context, state, config=None):
    email_source_key(entry, arguments, context)
    if is_calendar(entry):
        return calendar_prepare(entry, arguments, context), None
    return prepare_task(context, entry, arguments), None


async def verify(entry, operation, arguments, expected, result, context, config=None):
    if is_calendar(entry):
        return calendar_verify(entry, operation, arguments, expected, result, context, config)
    return await task_verify(entry, operation, arguments, expected, result, context, config)


async def after_read(entry, arguments, result, context):
    if is_calendar(entry):
        return calendar_after_read(entry, arguments, result, context)
    return await task_after_read(entry, arguments, result, context)


__all__ = ['IDEMPOTENCY', 'idempotency_path', 'guidance', 'task_write', 'preflight', 'prepare',
           'interpret', 'verify', 'after_read', 'email_source_key', 'project', 'project_output',
           'display_name', 'references', 'unfinished_document_notice']
