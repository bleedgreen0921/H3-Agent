"""本地 Web 附件的持久元数据和文件读取。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4
import os
import tempfile

from ai_assistant.config import ConfigError
from ai_assistant.documents.formats import (
    CSV_EXTENSIONS, DOCX_EXTENSIONS, OFFICE_EXTENSIONS, PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
)


DDL = '''CREATE TABLE IF NOT EXISTS assistant_attachments (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    input_id text,
    original_name text NOT NULL,
    media_type text NOT NULL,
    bytes bigint NOT NULL,
    sha256 text NOT NULL,
    storage_path text NOT NULL,
    state text NOT NULL DEFAULT 'ready',
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    expires_at timestamptz
)'''


class AttachmentError(RuntimeError):
    """附件操作的稳定安全错误。"""


@dataclass(frozen=True)
class AttachmentRecord:
    id: str
    session_id: str
    input_id: str | None
    original_name: str
    media_type: str
    bytes: int
    sha256: str
    storage_path: str
    state: str

    @classmethod
    def from_row(cls, row):
        return cls(
            id=str(row["id"]), session_id=str(row["session_id"]),
            input_id=row.get("input_id"), original_name=row["original_name"],
            media_type=row["media_type"], bytes=row["bytes"],
            sha256=row["sha256"], storage_path=row["storage_path"],
            state=row["state"])


@asynccontextmanager
async def attachment_store(settings):
    """按会话数据库连接打开附件仓储；每次操作使用独立连接。"""
    try:
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row
    except ImportError:
        raise ConfigError("安装 sessions 依赖以使用附件。") from None
    if settings is None:
        raise ConfigError("附件需要配置 [sessions]。")
    conn = await AsyncConnection.connect(
        settings.database_url.get_secret_value(), autocommit=True,
        row_factory=dict_row, connect_timeout=10)
    try:
        await conn.execute("SELECT id FROM assistant_attachments LIMIT 0")
        yield conn
    except Exception as error:
        if isinstance(error, ConfigError):
            raise
        raise AttachmentError("附件存储不可用，请检查数据库并显式执行 init。") from None
    finally:
        await conn.close()


async def initialize(settings):
    from psycopg import AsyncConnection
    conn = await AsyncConnection.connect(settings.database_url.get_secret_value(),
                                         autocommit=True, connect_timeout=10)
    try:
        await conn.execute(DDL)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS assistant_attachments_session "
            "ON assistant_attachments(session_id, created_at DESC)")
    finally:
        await conn.close()


async def cleanup(settings, *, days: int, execute: bool = False):
    """删除超过保留期的附件文件和元数据；默认只预览。"""
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    conn = await AsyncConnection.connect(settings.database_url.get_secret_value(),
                                         autocommit=True, row_factory=dict_row,
                                         connect_timeout=10)
    try:
        rows = await (await conn.execute(
            "SELECT * FROM assistant_attachments WHERE created_at < now() - (%s * interval '1 day')",
            (days,))).fetchall()
        if execute:
            import shutil
            for row in rows:
                shutil.rmtree(Path(row["storage_path"]).parent, ignore_errors=True)
                await conn.execute("DELETE FROM assistant_attachments WHERE id=%s",
                                   (row["id"],))
        return {"mode": "executed" if execute else "preview", "count": len(rows)}
    finally:
        await conn.close()


def _safe_name(value: str) -> str:
    name = Path(value or "attachment").name
    if name in {".", ".."} or "\\" in name or "\x00" in name:
        raise AttachmentError("附件文件名无效。")
    return name[:200] or "attachment"


def _safe_suffix(name: str, allowed: list[str]) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in allowed:
        raise AttachmentError("不支持的附件类型。")
    return suffix


def _media_type(name: str) -> str:
    import mimetypes
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _coverage(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "PDF 物理页码；空白页记录为 gaps；不提供 OCR。"
    if suffix in DOCX_EXTENSIONS:
        return "DOCX 正文段落和表格行；不提供 Word 页码和 OCR。"
    if suffix in CSV_EXTENSIONS:
        return "CSV/TSV 记录号；首行作为表头；不计算公式。"
    if suffix == ".json":
        return "JSON 原文行号；已校验 JSON 语法。"
    if suffix == ".jsonl" or suffix == ".ndjson":
        return "JSON Lines 原文行号；逐行读取。"
    if suffix in TEXT_EXTENSIONS:
        return "UTF-8/UTF-16 文本行号；代码只读，不执行。"
    if suffix in OFFICE_EXTENSIONS:
        return "XLSX/PPTX 结构化文本；不执行公式。"
    return "本地文本内容；不提供 OCR。"


class AttachmentService:
    """管理附件文件、数据库记录和只读解析缓存。"""

    def __init__(self, config):
        self.config = config
        self.settings = config.attachments
        self.root = self.settings.root()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.cache = {}
        self.cursors = {}

    async def create(self, session_id: str, filename: str, data: bytes):
        name = _safe_name(filename)
        suffix = _safe_suffix(name, self.settings.allowed_extensions)
        if len(data) > self.settings.max_bytes:
            raise AttachmentError("附件超过大小限制。")
        if not data:
            raise AttachmentError("附件为空。")
        if suffix == ".pdf" and not data.startswith(b"%PDF-"):
            raise AttachmentError("附件实际格式不是 PDF。")
        if suffix == ".docx" and not data.startswith(b"PK"):
            raise AttachmentError("附件实际格式不是 DOCX。")
        if suffix == ".doc":
            raise AttachmentError("旧版 .doc 不支持，请另存为 .docx。")
        if suffix in TEXT_EXTENSIONS | CSV_EXTENSIONS:
            from ai_assistant.documents.parsing import decode_text_bytes
            from ai_assistant.knowledge import IngestionError
            try:
                text = decode_text_bytes(data)
                if suffix == ".json":
                    import json
                    json.loads(text)
            except IngestionError as error:
                raise AttachmentError(str(error) if str(error) else "附件文本格式无效。") from None
            except ValueError:
                raise AttachmentError("JSON 文件语法无效。") from None
        ident = str(uuid4())
        directory = self.root / ident
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        path = directory / ("original" + suffix)
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        except BaseException:
            import shutil
            shutil.rmtree(directory, ignore_errors=True)
            raise
        digest = sha256(data).hexdigest()
        async with attachment_store(self.config.sessions) as conn:
            row = await (await conn.execute(
                '''INSERT INTO assistant_attachments
                (id,session_id,original_name,media_type,bytes,sha256,storage_path,state)
                VALUES(%s,%s,%s,%s,%s,%s,%s,'ready') RETURNING *''',
                (ident, session_id, name, _media_type(name), len(data), digest,
                 str(path)))).fetchone()
        return AttachmentRecord.from_row(row)

    async def get(self, attachment_id: str, session_id: str):
        try:
            ident = str(UUID(str(attachment_id)))
            session = str(UUID(str(session_id)))
        except (ValueError, TypeError, AttributeError):
            raise AttachmentError("附件标识无效。") from None
        async with attachment_store(self.config.sessions) as conn:
            row = await (await conn.execute(
                "SELECT * FROM assistant_attachments WHERE id=%s AND session_id=%s",
                (ident, session))).fetchone()
        if not row:
            raise AttachmentError("附件不存在或不属于当前会话。")
        return AttachmentRecord.from_row(row)

    async def list(self, session_id: str):
        async with attachment_store(self.config.sessions) as conn:
            rows = await (await conn.execute(
                "SELECT * FROM assistant_attachments WHERE session_id=%s ORDER BY created_at",
                (str(session_id),))).fetchall()
        return [AttachmentRecord.from_row(row) for row in rows]

    async def delete_session_files(self, session_id: str):
        import shutil
        for record in await self.list(session_id):
            shutil.rmtree(Path(record.storage_path).parent, ignore_errors=True)

    async def read(self, attachment_id: str, session_id: str, cursor: str | None = None):
        record = await self.get(attachment_id, session_id)
        path = Path(record.storage_path)
        if path.is_symlink() or not path.is_file():
            raise AttachmentError("附件文件不可用。")
        key = record.id
        if key not in self.cache:
            from ai_assistant.knowledge.loading import parse_snapshot
            try:
                self.cache[key] = await parse_snapshot(
                    path, timeout=self.settings.parse_timeout, granularity='read')
            except Exception:
                raise AttachmentError("附件解析失败，请检查格式或离线解析资源。") from None
        pages = self.cache[key]
        index, offset = 0, 0
        if cursor:
            saved = self.cursors.get(cursor)
            if saved is None or saved[0] != key:
                raise AttachmentError("续读游标无效或不属于此附件，请从头读取。")
            index, offset = saved[1]
        parts, remaining = [], self.settings.read_chars
        while index < len(pages) and remaining:
            page = pages[index]
            chunk = page.page_content[offset:offset + remaining]
            if chunk:
                parts.append({
                    "location": ({"page": index + 1} if path.suffix == ".pdf"
                                 else page.metadata),
                    "offset": offset,
                    "text": chunk,
                })
                remaining -= len(chunk)
            offset += len(chunk)
            if offset >= len(page.page_content):
                index, offset = index + 1, 0
        next_cursor = None
        if index < len(pages):
            next_cursor = str(uuid4())
            self.cursors[next_cursor] = (key, (index, offset))
        return {
            "attachment_id": record.id,
            "name": record.original_name,
            "format": path.suffix[1:],
            "parts": parts,
            "total_parts": len(pages),
            "gaps": [{"page": i + 1} for i, page in enumerate(pages)
                     if page.metadata.get("gap")],
            "coverage": _coverage(path.suffix),
            "next_cursor": next_cursor,
            "complete": next_cursor is None,
            "external_data": True,
        }

    async def delete_file(self, record: AttachmentRecord):
        import shutil
        shutil.rmtree(Path(record.storage_path).parent, ignore_errors=True)
