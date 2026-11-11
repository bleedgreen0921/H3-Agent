"""飞书 MCP adapter facade：transport/docs/tasks/mail_bridge 的组合。"""
from ai_assistant.integrations.feishu_adapter.docs import guidance as docs_guidance
from ai_assistant.integrations.feishu_adapter.docs import unfinished_document_notice
from ai_assistant.integrations.feishu_adapter.mail_bridge import email_source_key
from ai_assistant.integrations.feishu_adapter.tasks import (
    after_read, guidance as task_guidance, prepare as prepare_task, task_write, verify,
)
from ai_assistant.integrations.feishu_adapter.transport import IDEMPOTENCY, idempotency_path, interpret


def guidance(entries, context):
    return docs_guidance(entries) + task_guidance(entries)


async def prepare(entry, arguments, context, state, config=None):
    email_source_key(entry, arguments, context)
    return prepare_task(context, entry, arguments), None


__all__ = ['IDEMPOTENCY', 'idempotency_path', 'guidance', 'task_write', 'prepare',
           'interpret', 'verify', 'after_read', 'email_source_key', 'unfinished_document_notice']
