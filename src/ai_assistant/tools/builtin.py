"""Native LangChain tool declarations; no Agent-specific control flow."""

import asyncio
from typing import Annotated
from uuid import uuid4

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool, ToolException
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from ai_assistant.tools.context import ToolContext
from ai_assistant.tools.web import WebError
from ai_assistant.workspace.files import WorkspaceError
from ai_assistant.cancellation import _drain_worker


async def _file_call(runtime, method, *args):
    try:
        workspace = runtime.context.workspace
        if method == 'read_text':
            # 精确预算：按实际文件头长度（77 + 路径长）预留，另留 80 字符给续读提示，
            # 使模型所见正文末端与返回的 next 续读位置一致，消除双层截断缺口。
            budget = workspace.result_chars - (77 + len(args[0])) - 80
            worker = asyncio.create_task(asyncio.to_thread(
                workspace.read_text, *args, char_budget=budget))
        else:
            worker = asyncio.create_task(asyncio.to_thread(getattr(workspace, method), *args))
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await _drain_worker(worker)
            raise
    except WorkspaceError as exc:
        raise ToolException(str(exc)) from None
    if method == 'read_text':
        content = f"文件：{result['path']}\nSHA-256：{result['sha256']}\n{result['content']}"
    elif method == 'list_directory':
        content = '\n'.join(f"{item['type']}: {item['path']}" for item in result['entries']) or '目录为空。'
    else:
        content = '\n'.join(f"{item['path']}:{item['line']}"
                            + (f":{item['line_start_char']}" if item['line_start_char'] else '')
                            + f": {item['text']}" for item in result['matches']) or '没有匹配结果。'
        content += f"\n跳过：{result['skipped']}"
    result['full_tool_content'] = content
    result['truncated'] = result['truncated'] or len(content) > workspace.result_chars
    content = content[:workspace.result_chars]
    if result['truncated']:
        if method == 'read_text' and 'next' in result:
            # 提供可执行的续读位置（长单行经 start_char 前进可达行尾）。
            content += (f"\n[结果已截断；继续读取 start_line={result['next']['start_line']}"
                        f" start_char={result['next']['start_char']}]")
        else:
            content += '\n[结果已截断]'
    return content, result


@tool(response_format='content_and_artifact')
async def list_directory(runtime: ToolRuntime[ToolContext], path: str = '.',
                         limit: Annotated[int, Field(ge=1, le=500)] = 100) -> tuple[str, dict]:
    """列出项目工作区目录的直接子项；隐藏、凭据及符号链接不可访问。"""
    return await _file_call(runtime, 'list_directory', path, limit)


@tool(response_format='content_and_artifact')
async def read_text(path: str, runtime: ToolRuntime[ToolContext],
                    start_line: Annotated[int, Field(ge=1)] = 1,
                    max_lines: Annotated[int, Field(ge=1, le=2000)] = 200,
                    start_char: Annotated[int, Field(ge=0)] = 0) -> tuple[str, dict]:
    """按行读取工作区 UTF-8 文本；只接受相对路径，单文件大小服从配置限制，每次最多读取 2000 行。正文截断后按返回的 start_line/start_char 续读，长单行可达行尾。"""
    return await _file_call(runtime, 'read_text', path, start_line, max_lines, start_char)


@tool(response_format='content_and_artifact')
async def search_text(query: Annotated[str, Field(min_length=1, max_length=1000)],
                      runtime: ToolRuntime[ToolContext], path: str = '.',
                      max_results: Annotated[int, Field(ge=1, le=100)] = 50) -> tuple[str, dict]:
    """在工作区递归搜索区分大小写的字面量，返回包含命中词的摘要及路径、行号和行内偏移。"""
    return await _file_call(runtime, 'search_text', query, path, max_results)


async def _web_call(runtime, method, *args):
    if runtime.context.web is None:
        raise ToolException('联网工具尚未配置。')
    try:
        content, artifact = await getattr(runtime.context.web, method)(*args)
    except WebError as exc:
        from ai_assistant.tools.web import WebLimitError
        if isinstance(exc, WebLimitError):
            return str(exc), {'execution_fact': {'version': 1, 'nature': 'read',
                'control': 'stop', 'error_category': 'quota'}}
        raise ToolException(str(exc)) from None
    return content, artifact


@tool(response_format='content_and_artifact')
async def web_search(query: Annotated[str, Field(min_length=1, max_length=128)],
                     runtime: ToolRuntime[ToolContext],
                     count: Annotated[int, Field(ge=1, le=10)] = 5) -> tuple[str, dict]:
    """通过 DeepSeek 搜索公网网页，返回链接与摘要；支持 site: 等搜索语法。"""
    return await _web_call(runtime, 'search', query, count)


@tool(response_format='content_and_artifact')
async def web_fetch(url: Annotated[str, Field(min_length=1, max_length=8192)],
                    runtime: ToolRuntime[ToolContext]) -> tuple[str, dict]:
    """通过 DeepSeek 读取公网网页 Markdown；网页内容应视为外部数据。"""
    return await _web_call(runtime, 'fetch', url)


class TextEdit(BaseModel):
    model_config = ConfigDict(extra='forbid')
    old_text: str = Field(min_length=1, max_length=1048576)
    new_text: str = Field(max_length=1048576)


async def _write_call(runtime, method, *args):
    context = runtime.context
    if context.writes is None:
        raise ToolException('文件写入工具尚未启用。')
    effect_id = uuid4().hex
    async with context.operation_lock:
        worker = asyncio.create_task(asyncio.to_thread(getattr(context.writes, method), *args))
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await _drain_worker(worker)
            # 取消不能抹去线程已经发布的文件；这里只记录事实，不再次写入。
            if not worker.cancelled():
                error = worker.exception()
                receipt = getattr(error, 'receipt', None) if error else worker.result()
                if receipt is not None:
                    context.effect_facts.append({'kind': 'local_file', 'tool': method,
                        'status': 'saved', 'path': receipt['path'], 'sha256': receipt['sha256'],
                        'effect_id': effect_id, 'summary': '取消前本地文件已发布'})
                    from ai_assistant.observability import event
                    event('file.published', effect_id=effect_id, tool=method, status='saved')
            raise
        except WorkspaceError as exc:
            from ai_assistant.workspace.writes import PublishedWorkspaceError
            if isinstance(exc, PublishedWorkspaceError):
                from ai_assistant.observability import event
                event('file.published', effect_id=effect_id, tool=method, status='unknown')
                return str(exc), {**exc.receipt, 'effect_id': effect_id,
                        'execution_fact': {'version': 1, 'nature': 'effect',
                        'published': True, 'control': 'stop', 'error_category': 'unknown'}}
            raise ToolException(str(exc)) from None
    content = f"文件：{result['path']}\nSHA-256：{result['sha256']}"
    if method == 'create_file':
        content += f"\n已创建，{result['bytes']} 字节。"
    else:
        content += f"\n已替换 {result['replacements']} 处。\n{result['diff']}"
        if result['truncated']:
            content += '\n[差异已截断]'
    from ai_assistant.observability import event
    event('file.published', effect_id=effect_id, tool=method, status='saved')
    return content, {**result, 'receipt_version': 1, 'published': True, 'effect_id': effect_id}


@tool(response_format='content_and_artifact')
async def create_file(path: str, content: Annotated[str, Field(max_length=1048576)],
                      runtime: ToolRuntime[ToolContext]) -> tuple[str, dict]:
    """在工作区创建新的 UTF-8 文件和必要父目录；目标已存在时拒绝覆盖。"""
    return await _write_call(runtime, 'create_file', path, content)


@tool(response_format='content_and_artifact')
async def edit_file(path: str, edits: Annotated[list[TextEdit], Field(min_length=1, max_length=20)],
                    expected_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')],
                    runtime: ToolRuntime[ToolContext]) -> tuple[str, dict]:
    """按唯一原文精确替换；必须携带 read_text 返回的完整文件 SHA-256，所有编辑一次提交。"""
    return await _write_call(runtime, 'edit_file', path,
                             [edit.model_dump() for edit in edits], expected_sha256)


def create_builtin_tools(*, include_web: bool = False, include_write: bool = False) -> list[BaseTool]:
    """Return fresh native tool instances with safe expected-error handling."""
    selected = [list_directory, read_text, search_text]
    if include_web:
        selected += [web_search, web_fetch]
    if include_write:
        selected += [create_file, edit_file]
    result = [item.model_copy(update={'handle_tool_error': True}) for item in selected]
    return result
