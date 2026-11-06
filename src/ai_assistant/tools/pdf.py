"""主人私聊 PDF、DOCX、Markdown 附件临时阅读；不接入知识库或持久化工作区。"""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException

from ai_assistant.config import ConfigError
from ai_assistant.integrations.feishu_files import FileBridge
from ai_assistant.knowledge import IngestionError
from ai_assistant.knowledge.loading import parse_snapshot
from ai_assistant.tools.context import ToolContext
from ai_assistant.workspace.files import ReadOnlyWorkspace


class PDFAttachments:
    def __init__(self, api, timeout=60):
        self.directory = TemporaryDirectory(prefix='assistant-pdf-')
        self.root = Path(self.directory.name)
        self.bridge = FileBridge(api, ReadOnlyWorkspace(self.root))
        self.timeout = timeout
        self.cache = {}
        self.cursors = {}
        self.lock = asyncio.Lock()

    def close(self):
        self.cache.clear()
        self.cursors.clear()
        self.directory.cleanup()

    async def read_document(self, message_id, file_key, file_name, cursor=None):
        suffix = Path(file_name).suffix.lower()
        if suffix not in {".pdf", ".docx", ".md"}:
            raise ToolException("仅支持 PDF、DOCX、Markdown 附件。")
        return await self._read(message_id, file_key, cursor, suffix)

    async def _read(self, message_id, file_key, cursor, suffix):
        if not message_id or not file_key:
            raise ToolException('请提供附件所属 message_id 和 file_key。')
        key = (message_id, file_key, suffix)
        async with self.lock:
            position = (0, 0)
            if cursor is not None:
                saved = self.cursors.get(cursor)
                if saved is None or saved[0] != key:
                    raise ToolException('续读游标无效或不属于此附件，请从头读取。')
                position = saved[1]
            if key not in self.cache:
                path = None
                try:
                    result = await self.bridge.download_message_file(message_id, file_key, 'attachment' + suffix)
                    path = self.root / result['path']
                    with path.open('rb') as stream:
                        if suffix == '.pdf' and stream.read(5) != b'%PDF-':
                            raise ToolException('附件实际格式不是 PDF。')
                    pages = await parse_snapshot(path, timeout=self.timeout, attachment=True)
                    if suffix == '.pdf' and not any(page.page_content.strip() for page in pages):
                        raise ToolException('PDF 无可提取文字，可能是扫描件或空白页；暂不支持 OCR。')
                    self.cache[key] = pages
                except IngestionError as error:
                    raise ToolException(str(error)) from None
                except (httpx.HTTPError, ConfigError):
                    raise ToolException('飞书附件下载失败，请检查消息、文件标识及机器人消息资源权限。') from None
                except (OSError, ValueError):
                    raise ToolException('附件读取失败：文件不可用或超过 25 MiB。') from None
                finally:
                    if path is not None:
                        path.unlink(missing_ok=True)
            pages = self.cache[key]
            index, offset = position
            parts, remaining = [], 16000
            # 页内偏移与页码一起推进，长页不会丢失剩余正文。
            while index < len(pages) and remaining:
                page = pages[index]
                chunk = page.page_content[offset:offset + remaining]
                if chunk:
                    parts.append({'location': ({'page': index + 1} if suffix == '.pdf' else page.metadata),
                                  'offset': offset, 'text': chunk})
                    remaining -= len(chunk)
                offset += len(chunk)
                if offset >= len(page.page_content):
                    index, offset = index + 1, 0
            next_cursor = None
            if index < len(pages):
                next_cursor = uuid4().hex
                self.cursors[next_cursor] = (key, (index, offset))
            coverage = {'.pdf': '物理页码；图片未识别，暂不支持 OCR。',
                        '.docx': '仅正文段落及表格文字；不含图片、批注、页眉页脚、文本框等；不提供物理页码。',
                        '.md': 'UTF-8 原文；不渲染、不访问链接、不加载图片。'}
            return {'format': suffix[1:], 'parts': parts, 'total_parts': len(pages),
                    'gaps': [{'page': i + 1} for i, page in enumerate(pages) if page.metadata.get('gap')],
                    'coverage': coverage[suffix], 'next_cursor': next_cursor,
                    'complete': next_cursor is None}


@tool
async def read_document_attachment(message_id: str, file_key: str, file_name: str,
                                   runtime: ToolRuntime[ToolContext], cursor: str | None = None) -> dict:
    """读取飞书 PDF、DOCX、Markdown 附件正文及位置，每次最多 16000 字符，用 next_cursor 续读。仅主人私聊；不入库，不授权写入。"""
    caller = runtime.context.caller
    if (caller is None or not caller.owner or caller.scope != 'private'
            or caller.channel != 'feishu_private' or runtime.context.pdf is None):
        raise ToolException('附件阅读仅对已启用的主人飞书私聊开放。')
    return await runtime.context.pdf.read_document(message_id, file_key, file_name, cursor)


read_document_attachment = read_document_attachment.model_copy(update={"handle_tool_error": True})
