"""飞书文档的高层写入能力：Markdown -> blocks -> 追加。

原生 ``documentBlockChildren.create`` 的 schema 包含全部块类型，直接暴露给
模型既昂贵又容易把 ``index``/``children`` 放错层级。这里提供一个语义工具，
模型只提交 Markdown 与目标块；转换和嵌套 block 组装在适配层完成。
"""
from __future__ import annotations

import json
import re
from typing import Annotated

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from ai_assistant.integrations.feishu_files import FeishuAPI, FeishuRejected

SEMANTIC_TOOLS = frozenset({'docx_append_markdown'})

_ID = re.compile(r'^[A-Za-z0-9_-]+$')


class AppendMarkdownArgs(BaseModel):
    document_id: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    block_id: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    markdown: str = Field(min_length=1, max_length=1_000_000)
    index: Annotated[int | None, Field(ge=0, le=100_000)] = None


def _clean_blocks(value):
    """移除只读字段；保留 block_id/children 供 descendants 关系解析。"""
    if isinstance(value, dict):
        return {key: _clean_blocks(item) for key, item in value.items()
                if key not in {'merge_info', 'comment_ids', 'parent_id'}}
    if isinstance(value, list):
        return [_clean_blocks(item) for item in value]
    return value


def _rejected(code, action):
    # 不把 provider 的 log_id/troubleshooter 带进模型上下文；只在回执里保留 code。
    if code == 99991679:
        return {'code': code,
                'msg': '缺少文档转换授权 docx:document.block:convert；'
                       '请重新执行飞书 OAuth 授权后再写入。'}
    return {'code': code, 'msg': f'飞书拒绝{action}（权限或参数），未确认写入。'}


async def append_markdown(api: FeishuAPI, document_id: str, block_id: str,
                          markdown: str, index: int | None = None,
                          convert_api: FeishuAPI | None = None) -> dict:
    if not _ID.fullmatch(document_id) or not _ID.fullmatch(block_id):
        raise ToolException('文档或块标识格式不正确。')
    # 转换是无状态内容处理，使用应用身份即可；真正写入仍使用用户身份。
    converter = convert_api or api
    try:
        converted = await converter.request(
            'POST', '/docx/v1/documents/blocks/convert',
            json={'content_type': 'markdown', 'content': markdown})
    except FeishuRejected as error:
        return _rejected(error.code, '转换 Markdown')
    blocks = converted.get('blocks') if isinstance(converted, dict) else None
    first_level = converted.get('first_level_block_ids') if isinstance(converted, dict) else None
    if not isinstance(blocks, list) or not blocks or not isinstance(first_level, list) or not first_level:
        return {'code': 1770001, 'msg': 'Markdown 转换没有返回可插入的文档块。'}
    payload = {'children_id': first_level, 'descendants': _clean_blocks(blocks)}
    if index is not None:
        payload['index'] = index
    try:
        created = await api.request(
            'POST', f'/docx/v1/documents/{document_id}/blocks/{block_id}/descendant',
            json=payload)
    except FeishuRejected as error:
        return _rejected(error.code, '写入文档')
    data = created if isinstance(created, dict) else {}
    revision = data.get('document_revision_id', data.get('revision_id'))
    return {'code': 0, 'data': {
        'document_id': document_id,
        'revision_id': revision,
        'block_ids': first_level,
        'inserted_count': len(first_level),
        'status': 'succeeded',
    }}


def create_semantic_tools(api: FeishuAPI, convert_api: FeishuAPI | None = None):
    """返回可在官方 Feishu MCP 服务下挂载的本地语义工具。

    ``api`` 用于用户身份写入；``convert_api`` 只用于无状态的 Markdown 转换，
    默认使用应用身份，避免要求用户 token 额外申请转换 scope。
    """

    async def append(document_id: str, block_id: str, markdown: str,
                     index: int | None = None) -> dict:
        return json.dumps(
            await append_markdown(api, document_id, block_id, markdown, index,
                                  convert_api=convert_api),
            ensure_ascii=False)

    return [StructuredTool.from_function(
        coroutine=append, name='docx_append_markdown',
        description=('将 Markdown 追加到飞书文档指定块。自动转换为飞书文档块并插入，'
                     '不需要手写 block JSON；只写入用户已授权的文档。'),
        args_schema=AppendMarkdownArgs)]


__all__ = ['SEMANTIC_TOOLS', 'create_semantic_tools']
