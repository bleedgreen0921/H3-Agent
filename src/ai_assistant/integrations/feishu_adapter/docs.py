"""飞书文档/云空间读取与移动规则；失败只读取回执，不重放。"""
from ai_assistant.integrations.feishu_tasks import payload


def guidance(entries):
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
    return text


async def unfinished_document_notice(context):
    """正文写入失败时从本轮成功创建回执恢复链接，避免丢失已创建文档。"""
    import re
    from types import SimpleNamespace
    if context.operations is None or 'mcp' not in context.services or context.input_id is None:
        return ''
    entries = context.services['mcp'].entries
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
