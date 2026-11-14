"""飞书协议事实：幂等参数和通用结果解释，不包含文档/任务业务规则。"""
from langchain_core.messages import ToolMessage

from ai_assistant.integrations.feishu_tasks import payload

# 原生工具名映射到接口支持的幂等参数位置。
IDEMPOTENCY = {
    'im_v1_message_create': ('data', 'uuid'), 'im_v1_message_reply': ('data', 'uuid'),
    'calendar_v4_calendarEvent_create': ('params', 'idempotency_key'),
    'task_v2_task_create': ('data', 'client_token'), 'task_v2_task_addMembers': ('data', 'client_token'),
    'bitable_v1_appTableRecord_create': ('params', 'client_token'),
}


def idempotency_path(entry):
    """仅对飞书传输返回已登记的幂等参数路径。"""
    return IDEMPOTENCY.get(entry.original_name) if entry.settings.transport == 'feishu' else None


def interpret(entry, result):
    """综合工具状态、结构化状态和远端错误码解释调用结果。"""
    status = 'unknown' if getattr(result, 'status', '') == 'error' else 'succeeded'
    artifact = getattr(result, 'artifact', None)
    structured = artifact.get('structured_content') if isinstance(artifact, dict) else None
    state = structured.get('status') if isinstance(structured, dict) else None

    # 特殊结构化状态可能区分远端拒绝、未知及部分接受。
    if state in {'unknown', 'rejected', 'partially_accepted', 'replacement_saved_old_retained'}:
        status = 'unknown' if state == 'unknown' else 'failed' if state == 'rejected' else 'succeeded'
    remote = payload(result)
    if isinstance(remote, dict) and remote.get('code', 0) != 0:
        status = 'failed'

    # 失败或部分完成时附加执行事实，写操作停止，读操作允许继续处理。
    if isinstance(result, ToolMessage) and (status != 'succeeded' or state in {'partially_accepted', 'replacement_saved_old_retained'}):
        writing = entry.original_name in getattr(entry.settings, 'write_tools', ())
        code = remote.get('code') if isinstance(remote, dict) else None
        category = ('permission' if code in {1770032, 99991679}
                    else 'remote_rejected' if status == 'failed' else 'unknown')
        result = result.model_copy(update={
            'status': 'error',
            'artifact': {
                **(result.artifact if isinstance(result.artifact, dict) else {}),
                'execution_fact': {
                    'version': 1,
                    'nature': 'effect' if writing else 'read',
                    'control': 'stop' if writing else 'continue',
                    'error_category': category,
                    'error_code': code,
                },
            },
        })
    return result, status
