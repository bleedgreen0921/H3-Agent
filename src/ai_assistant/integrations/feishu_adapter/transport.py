"""飞书协议事实：幂等参数和通用结果解释，不包含文档/任务业务规则。"""
from langchain_core.messages import ToolMessage

from ai_assistant.integrations.feishu_tasks import payload

IDEMPOTENCY = {
    'im_v1_message_create': ('data', 'uuid'), 'im_v1_message_reply': ('data', 'uuid'),
    'calendar_v4_calendarEvent_create': ('params', 'idempotency_key'),
    'task_v2_task_create': ('data', 'client_token'), 'task_v2_task_addMembers': ('data', 'client_token'),
    'bitable_v1_appTableRecord_create': ('params', 'client_token'),
}


def idempotency_path(entry):
    return IDEMPOTENCY.get(entry.original_name) if entry.settings.transport == 'feishu' else None


def interpret(entry, result):
    status = 'unknown' if getattr(result, 'status', '') == 'error' else 'succeeded'
    artifact = getattr(result, 'artifact', None)
    structured = artifact.get('structured_content') if isinstance(artifact, dict) else None
    state = structured.get('status') if isinstance(structured, dict) else None
    if state in {'unknown', 'rejected', 'partially_accepted', 'replacement_saved_old_retained'}:
        status = 'unknown' if state == 'unknown' else 'failed' if state == 'rejected' else 'succeeded'
    remote = payload(result)
    if isinstance(remote, dict) and remote.get('code', 0) != 0:
        status = 'failed'
    if isinstance(result, ToolMessage) and (status != 'succeeded' or state in {'partially_accepted', 'replacement_saved_old_retained'}):
        result = result.model_copy(update={'status': 'error'})
    return result, status
