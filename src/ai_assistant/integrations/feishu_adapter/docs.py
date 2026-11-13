"""飞书文档/云空间读取与移动规则；失败只读取回执，不重放。"""
from ai_assistant.integrations.tool_policy import PreflightIssue, ToolPreflightError

_DOC_LABELS = {
    'docx_append_markdown': '追加文档内容',
    'docx_v1_document_create': '创建文档',
    'docx_v1_document_get': '读取文档信息',
    'docx_v1_document_rawContent': '读取文档正文',
    'docx_v1_documentBlock_get': '读取文档块',
    'docx_v1_documentBlock_list': '读取文档块',
    'docx_v1_documentBlockChildren_create': '写入文档内容',
    'docx_v1_documentBlockDescendant_create': '写入文档内容',
    'docx_v1_document_convert': '转换 Markdown 为文档块',
    'docx_builtin_search': '搜索文档',
    'wiki_v1_node_search': '搜索 Wiki',
    'wiki_v2_space_getNode': '解析 Wiki 节点',
    'wiki_v2_space_list': '列出知识空间',
    'wiki_v2_spaceNode_list': '列出 Wiki 节点',
    'wiki_v2_task_get': '查询 Wiki 任务',
    'drive_v1_file_list': '列出云空间文件',
    'drive_v1_file_taskCheck': '检查移动任务',
}


def display_name(original_name):
    """给用户和进度卡看的业务名称；内部 stable id 不进入正文。"""
    return _DOC_LABELS.get(original_name, original_name or '外部工具')


def guidance(entries):
    names = {entry.original_name for entry in entries if entry.settings.transport == 'feishu'}
    text = ''
    if any(name.startswith(('docx_', 'wiki_')) for name in names):
        text += '飞书链接使用已认证 MCP；Wiki 先 getNode 解析 obj_token/obj_type，再读取正文；登录页不证明没有凭据。'
    if 'docx_append_markdown' in names:
        text += ('追加 Markdown 到飞书文档时优先使用 docx_append_markdown；'
                 '它自动转换并插入文档块，不要手写 block JSON。')
    if any(name.startswith(('drive_', 'wiki_', 'docx_')) for name in names):
        text += ('文档搜索按原生 schema 分页，展示标题、类型、定位和链接；有限结果不代表全部。'
                 '同名候选不唯一时让用户选择后再写入。Wiki 解析节点后读取底层文档，长正文用块列表分页。'
                 '移动必须明确源和目标；Wiki 源节点先检查子节点，有子节点时说明连同移动范围并取得确认。'
                 '支持云空间文件夹间、Wiki 节点间和云空间文档移入 Wiki，不支持 Wiki 移回云空间。'
                 '不支持重命名、删除、复制、新建文件夹、批量整理或权限变更。'
                 '异步移动仅报告已提交及任务 ID，使用 taskCheck/task_get 查询完成状态，不重发移动请求。'
                 '权限不足时说明工具所需权限，提示管理员开通后重新执行本地飞书 OAuth 授权，不自动扩大授权。')
    return text


def preflight(entry, arguments, context):
    """写入文档块前的本地结构校验；不触碰远端。"""
    name = getattr(entry, 'original_name', '')
    if name not in {'docx_v1_documentBlockChildren_create',
                    'docx_v1_documentBlockDescendant_create'}:
        return
    data = arguments.get('data') if isinstance(arguments, dict) else None
    blocks = data.get('children') if isinstance(data, dict) else None
    if not isinstance(blocks, list) or not blocks:
        blocks = data.get('descendants') if isinstance(data, dict) else None
    if not isinstance(blocks, list) or not blocks:
        raise ToolPreflightError(PreflightIssue(
            'invalid_arguments',
            '写入文档内容缺少 data.children/data.descendants；'
            '请优先使用 docx_append_markdown，避免手写 block JSON。',
            recoverable=True))


def references(entry, result):
    """从写入回执提取稳定资源引用；原始内容仍留在受保护回执中。"""
    from ai_assistant.integrations.feishu_tasks import payload
    name = getattr(entry, 'original_name', '')
    if name not in {'docx_v1_document_create', 'docx_append_markdown'}:
        return {}
    remote = payload(result)
    if not isinstance(remote, dict) or remote.get('code', 0) != 0:
        return {}
    data = remote.get('data', remote)
    data = data if isinstance(data, dict) else {}
    if name == 'docx_v1_document_create':
        document = data.get('document', {}) if isinstance(data, dict) else {}
        ident = document.get('document_id') if isinstance(document, dict) else None
        if isinstance(ident, str) and ident:
            return {'document_id': ident, 'url': 'https://feishu.cn/docx/' + ident,
                    'title': document.get('title')}
        return {}
    reference = {'document_id': data.get('document_id')}
    if data.get('revision_id') is not None:
        reference['revision_id'] = data['revision_id']
    return reference


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
        from ai_assistant.integrations.feishu_tasks import payload
        remote = payload(SimpleNamespace(content=receipt.get('content'), artifact=receipt.get('artifact')))
        data = remote.get('data', remote) if isinstance(remote, dict) else {}
        document = data.get('document', {}) if isinstance(data, dict) else {}
        ident = document.get('document_id')
        if isinstance(ident, str) and re.fullmatch(r'[A-Za-z0-9_-]+', ident):
            links.append('https://feishu.cn/docx/' + ident)
    return ('\n已创建文档：' + '、'.join(dict.fromkeys(links)) +
            '。正文整理或写入尚未确认完成；未重新创建或重放，请读取目标核对后明确后续操作。') if links else ''


__all__ = ['display_name', 'guidance', 'preflight', 'references', 'unfinished_document_notice']
