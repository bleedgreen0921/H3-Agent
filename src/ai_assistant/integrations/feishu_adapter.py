"""飞书协议及业务事实适配；不拥有轮次、预算或恢复调度。"""
from langchain_core.messages import ToolMessage
from ai_assistant.integrations.feishu_tasks import (
    TASK_WRITES, prepare_task, task_data, verification, payload, task_model_result,
)

IDEMPOTENCY = {
    'im_v1_message_create': ('data', 'uuid'), 'im_v1_message_reply': ('data', 'uuid'),
    'calendar_v4_calendarEvent_create': ('params', 'idempotency_key'),
    'task_v2_task_create': ('data', 'client_token'), 'task_v2_task_addMembers': ('data', 'client_token'),
    'bitable_v1_appTableRecord_create': ('params', 'client_token'),
}


def idempotency_path(entry):
    return IDEMPOTENCY.get(entry.original_name) if entry.settings.transport == 'feishu' else None


def task_write(entry):
    return entry.settings.transport == 'feishu' and entry.original_name in TASK_WRITES


def guidance(entries, context):
    names = {entry.original_name for entry in entries if entry.settings.transport == 'feishu'}
    text = ''
    if any(name.startswith(('docx_', 'wiki_')) for name in names):
        text += '飞书链接使用已认证 MCP；Wiki 先 getNode 解析 obj_token/obj_type，再读取正文；登录页不证明没有凭据。'
    if any(name.startswith(('drive_', 'wiki_', 'docx_')) for name in names):
        text += ('文档搜索按原生 schema 分页，展示标题、类型、定位和链接；有限结果不代表全部。'
                 '同名候选不唯一时让用户选择后再写入。Wiki 解析节点后读取底层文档，长正文用块列表分页。'
                 '移动必须明确源和目标；Wiki 源节点先检查子节点，有子节点时说明连同移动范围并取得确认。'
                 '支持云空间文件夹间、Wiki 节点间和云空间文档移入 Wiki，不支持 Wiki 移回云空间。'
                 '不支持重命名、删除、复制、新建文件夹、批量整理或权限变更。'
                 '异步移动仅报告已提交及任务 ID，使用 taskCheck/task_get 查询完成状态，不重发移动请求。'
                 '权限不足时说明工具所需权限，提示管理员开通后重新执行本地飞书 OAuth 授权，不自动扩大授权。')
    if any(name.startswith('task_v2_task_') for name in names):
        text += '任务时间先结合可信当前日期明确为 ISO，再调用 resolve_datetime；写入时间须来自本轮转换，日期不明确先澄清。身份使用可信身份工具，不从姓名推导 ID。主人飞书私聊创建时未传 members 遵循默认负责人配置，显式空列表不覆盖；群聊须明确时区与身份。patch 只更新 update_fields，负责人使用 addMembers。回执核验仅比较本次发送字段与本次接口返回；未返回字段表示未确认，接口接受不证明提醒实际送达。'
    return text


def email_source_key(entry, arguments, context):
    extra = arguments.get('data', {}).get('extra')
    if not isinstance(extra, str) or not extra.startswith('email-task:'):
        return None
    from ai_assistant.tools.mail import allowed
    from langchain_core.tools import ToolException
    if (entry.original_name != 'task_v2_task_create' or entry.settings.transport != 'feishu'
            or entry.settings.identity != 'user' or not allowed(context.caller)
            or getattr(context, 'mail', None) is None):
        raise ToolException('邮件来源任务仅允许主人私聊通过 user 身份创建。')
    return context.mail.validate(arguments)


async def prepare(entry, arguments, context, state, config=None):
    email_source_key(entry, arguments, context)
    return (prepare_task(context, entry, arguments) if task_write(entry) else {}), None


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


async def verify(entry, operation, arguments, expected, result, context, config=None):
    if not task_write(entry):
        return result, None
    detail = verification(expected, task_data(result), arguments.get('path', {}).get('task_guid'))
    if operation['status'] != 'succeeded':
        detail['verification'] = 'unverified'
    if isinstance(result, ToolMessage):
        result = result.model_copy(update={'status': 'error' if detail['verification'] == 'mismatch' else result.status,
            'artifact': {**(result.artifact if isinstance(result.artifact, dict) else {}), 'protected_receipt': True,
                         'verification': detail}})
    return result, detail


async def after_read(entry, arguments, result, context):
    if entry.settings.transport == 'feishu' and entry.original_name.startswith('task_v2_task_') and isinstance(result, ToolMessage):
        result = task_model_result(result, context)
    return result


async def unfinished_document_notice(context):
    """正文写入失败时从本轮成功创建回执恢复链接，避免丢失已创建文档。"""
    import re
    from types import SimpleNamespace
    if context.operations is None or context.mcp is None or context.input_id is None:
        return ''
    entries = context.mcp.entries
    if not any(entry.settings.transport == 'feishu' and entry.original_name == 'docx_v1_document_create'
               for entry in entries.values()):
        return ''
    links = []
    for row in await context.operations.list(context.input_id):
        entry = entries.get(row.get('tool'))
        if (entry is None or entry.settings.transport != 'feishu'
                or entry.original_name != 'docx_v1_document_create' or row['status'] != 'succeeded'):
            continue
        receipt = row.get('result') or {}
        remote = payload(SimpleNamespace(content=receipt.get('content'), artifact=receipt.get('artifact')))
        data = remote.get('data', remote) if isinstance(remote, dict) else {}
        document = data.get('document', {}) if isinstance(data, dict) else {}
        ident = document.get('document_id')
        if isinstance(ident, str) and re.fullmatch(r'[A-Za-z0-9_-]+', ident):
            links.append('https://feishu.cn/docx/' + ident)
    return ('\n已创建文档：' + '、'.join(dict.fromkeys(links)) +
            '。正文整理或写入尚未确认完成；未重新创建或重放，请读取目标核对后明确后续操作。') if links else ''
