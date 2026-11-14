"""通用查询意图与能力元数据推导。

这里只使用动作/对象/领域的通用词汇，不识别具体产品或服务。
"""
import re
from dataclasses import dataclass

# 将中英文关键词归一到通用动作，供请求和能力描述共同使用。
ACTION_TERMS = {
    'read': ('read', 'get', 'download', 'open', 'load', 'fetch', 'raw', 'content', '读取', '查看', '打开', '下载', '抓取'),
    'summarize': ('summarize', 'summary', 'summarise', '分析', '总结', '概括'),
    'search': ('search', 'query', 'find', 'lookup', '搜索', '查询', '查找', '检索', '查'),
    'list': ('list', 'enumerate', '列出', '列表', '枚举'),
    'create': ('create', 'add', 'new', 'build', 'ingest', 'import', 'write', 'store', 'prepare', 'save', 'join', 'sync', '创建', '新建', '添加', '增加', '入库', '导入', '写入', '加入', '存入', '保存', '同步'),
    'update': ('update', 'patch', 'edit', 'modify', 'set', '更新', '修改', '编辑', '调整'),
    'delete': ('delete', 'remove', 'drop', '删除', '移除'),
    'upload': ('upload', '上传'),
    'send': ('send', 'post', '发送'),
}
# 含这些动作的意图或能力会被标记为可能产生写入效果。
WRITE_ACTIONS = frozenset({'create', 'update', 'delete', 'upload', 'send'})

# 领域描述业务范围，对象描述操作目标；一个词可以属于多个类别。
DOMAIN_TERMS = {
    'docs': ('document', 'docx', 'wiki', 'drive', 'markdown', '文档', '论文', 'pdf', '知识库'),
    'tasks': ('task', 'todo', '任务', '待办'),
    'bitable': ('bitable', 'base', 'record', 'field', 'table', 'sheet', '表格', '多维表格', '记录'),
    'calendar': ('calendar', 'event', 'schedule', 'meeting', '日历', '日程', '会议'),
    'files': ('file', 'drive', 'folder', 'upload', 'download', '文件', '网盘', '云盘'),
    'messages': ('message', 'chat', 'im', '消息', '群聊'),
    'contacts': ('contact', 'user', '联系人'),
    'knowledge': ('knowledge', 'rag', 'index', '知识库', '检索'),
    'mail': ('mail', 'email', 'imap', '邮件', '邮箱'),
    'web': ('web', 'http', 'url', 'search', 'fetch', '网页', '链接'),
    'skills': ('skill', 'guidance', '技能'),
}
OBJECT_TERMS = {
    'attachment': ('attachment', 'file reference', 'file_key', '附件'),
    'document': ('document', 'docx', 'wiki', 'markdown', '文档', '论文', 'pdf'),
    'task': ('task', '任务', '待办'),
    'record': ('record', 'field', 'table', 'sheet', '记录', '表格'),
    'message': ('message', 'chat', 'im', '消息'),
    'file': ('file', 'folder', 'drive', '文件', '网盘', '云盘'),
    'email': ('email', 'mail', '邮件'),
    'event': ('calendar', 'event', 'schedule', '日历', '日程', '会议'),
    'knowledge': ('knowledge', 'rag', 'index', '知识库'),
    'contact': ('contact', 'user', '联系人'),
    'web': ('web', 'http', 'url', '网页'),
}
ATTACHMENT_TERMS = ('file reference', 'file_key', 'attachment', '附件')

# 英文按词提取；中文先提取连续文本，再补充相邻双字词。
_WORD = re.compile(r'[a-z0-9_]+')
_CJK = re.compile(r'[\u4e00-\u9fff]+')


@dataclass(frozen=True)
class QueryIntent:
    """保存一次关键词推导出的意图，供能力路由使用。"""

    actions: frozenset[str] = frozenset()
    objects: frozenset[str] = frozenset()
    domains: frozenset[str] = frozenset()
    side_effect: str = 'read'
    attachments: bool = False
    confidence: float = 0.0

    def summary(self):
        # 排序后便于稳定展示和记录；此摘要不包含置信度。
        return {'actions': sorted(self.actions), 'objects': sorted(self.objects),
                'domains': sorted(self.domains), 'side_effect': self.side_effect,
                'attachments': self.attachments}


def _tokens(text):
    """提取英文词、连续中文片段及其相邻双字词。"""
    text = (text or '').casefold()
    tokens = set(_WORD.findall(text))
    for run in _CJK.findall(text):
        tokens.add(run)
        tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _matches(text, groups):
    """按子串匹配返回所有命中的类别，不做语义或否定判断。"""
    text = (text or '').casefold()
    return {name for name, terms in groups.items() if any(term in text for term in terms)}


def query_terms(query):
    """为检索生成原始词项，并补充命中的标准类别名称。"""
    terms = _tokens(query)
    for groups in (ACTION_TERMS, DOMAIN_TERMS, OBJECT_TERMS):
        for name, words in groups.items():
            if any(word in (query or '').casefold() for word in words):
                terms.add(name)
    return terms


def infer_descriptor(name, description, *, write_hint=False, use_description=True):
    """从工具/技能的名称与描述推导通用元数据。"""
    # 名称是稳定的 capability 标识；描述里的“不写入/执行其他任务”等句子会污染元数据。
    actions = _matches(name, ACTION_TERMS)
    objects = _matches(name, OBJECT_TERMS)
    domains = _matches(name, DOMAIN_TERMS)
    if use_description and not (actions or objects or domains):
        text = f'{name} {description}'
        actions = _matches(text, ACTION_TERMS)
        objects = _matches(text, OBJECT_TERMS)
        domains = _matches(text, DOMAIN_TERMS)
    side_effect = 'write' if write_hint or actions & WRITE_ACTIONS else 'read'
    return frozenset(actions), frozenset(objects), frozenset(domains), side_effect


def extract(query):
    """从用户请求推导动作、对象、领域、附件和读写倾向。"""
    text = (query or '').casefold()
    actions = _matches(text, ACTION_TERMS)
    objects = _matches(text, OBJECT_TERMS)
    domains = _matches(text, DOMAIN_TERMS)
    attachments = any(term in text for term in ATTACHMENT_TERMS)
    side_effect = 'write' if actions & WRITE_ACTIONS else 'read'

    # 置信度仅按命中类别数粗略估算，不代表模型概率或权限判断。
    matches = len(actions) + len(objects) + len(domains) + int(attachments)
    return QueryIntent(actions=frozenset(actions), objects=frozenset(objects),
                       domains=frozenset(domains), side_effect=side_effect,
                       attachments=attachments, confidence=min(1.0, matches / 4))
