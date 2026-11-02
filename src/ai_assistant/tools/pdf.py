"""主人私聊 PDF 附件临时阅读；不接入知识库或持久化工作区。"""
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

    async def read(self, message_id, file_key, cursor=None):
        if not message_id or not file_key:
            raise ToolException('请提供附件所属 message_id 和 file_key。')
        key = (message_id, file_key)
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
                    result = await self.bridge.download_message_file(message_id, file_key, 'attachment.pdf')
                    path = self.root / result['path']
                    with path.open('rb') as stream:
                        if stream.read(5) != b'%PDF-':
                            raise ToolException('附件实际格式不是 PDF。')
                    pages = await parse_snapshot(path, timeout=self.timeout, attachment=True)
                    if not any(page.page_content.strip() for page in pages):
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
                    parts.append({'page': index + 1, 'offset': offset, 'text': chunk})
                    remaining -= len(chunk)
                offset += len(chunk)
                if offset >= len(page.page_content):
                    index, offset = index + 1, 0
            next_cursor = None
            if index < len(pages):
                next_cursor = uuid4().hex
                self.cursors[next_cursor] = (key, (index, offset))
            return {'pages': parts, 'total_pages': len(pages),
                    'gaps': [i + 1 for i, page in enumerate(pages) if page.metadata.get('gap')],
                    'next_cursor': next_cursor,
                    'complete': next_cursor is None,
                    'notice': '空白、扫描或提取失败页列于 gaps；图片内容未识别，暂不支持 OCR。'}


@tool
async def read_pdf_attachment(message_id: str, file_key: str, runtime: ToolRuntime[ToolContext],
                              cursor: str | None = None) -> dict:
    """读取飞书 PDF 附件，最多 16000 正文字符，按 next_cursor 续读。仅主人私聊；不入库。"""
    caller = runtime.context.caller
    if (caller is None or not caller.owner or caller.scope != 'private'
            or caller.channel != 'feishu_private' or runtime.context.pdf is None):
        raise ToolException('PDF 附件阅读仅对已启用的主人飞书私聊开放。')
    return await runtime.context.pdf.read(message_id, file_key, cursor)


read_pdf_attachment = read_pdf_attachment.model_copy(update={"handle_tool_error": True})
