"""附件和知识库共用的文档解析入口。"""
from __future__ import annotations

import codecs
import csv
import io
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

from langchain_core.documents import Document

from ai_assistant.documents.formats import (
    CSV_EXTENSIONS, DOCX_EXTENSIONS, OFFICE_EXTENSIONS, PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
)
from ai_assistant.knowledge import IngestionError

MAX_EXPANDED_BYTES = 100 * 1024**2
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def _xml(raw):
    declarations = raw.replace(b'\x00', b'').upper()
    if b'<!DOCTYPE' in declarations or b'<!ENTITY' in declarations:
        raise ValueError('XML entities are unsupported')
    return ET.fromstring(raw)


def _docx_text(element):
    return ''.join(node.text or '' if node.tag == W + 't' else
                   '\t' if node.tag == W + 'tab' else '\n'
                   for node in element.iter()
                   if node.tag in {W + 't', W + 'tab', W + 'br', W + 'cr'})


def decode_text_bytes(data: bytes) -> str:
    """只接受 UTF-8/UTF-16 BOM 文本；其他编码明确失败。"""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = 'utf-16'
    else:
        encoding = 'utf-8-sig'
    try:
        text = data.decode(encoding)
    except UnicodeError:
        raise IngestionError(
            'text_encoding', f'文本必须是 UTF-8 或带 BOM 的 UTF-16；当前不是 {encoding}。') from None
    if '\x00' in text:
        raise IngestionError('text_binary', '文件包含 NUL 字节，不能按文本读取。')
    return text


def _line_docs(text, suffix, granularity):
    if granularity == 'ingest':
        return [Document(page_content=text, metadata={'format': suffix[1:]})]
    return [
        Document(page_content=line, metadata={'line': number})
        for number, line in enumerate(text.splitlines(keepends=True), 1)
    ]


def _json_docs(text, suffix, granularity):
    if suffix == '.json':
        try:
            json.loads(text)
        except (TypeError, ValueError):
            raise IngestionError('json_invalid', 'JSON 文件语法无效。') from None
    return _line_docs(text, suffix, granularity)


def _csv_docs(text, suffix, granularity):
    delimiter = '\t' if suffix == '.tsv' else ','
    old_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(1024 * 1024)
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error:
        raise IngestionError('csv_invalid', 'CSV/TSV 记录格式无效或字段过大。') from None
    finally:
        csv.field_size_limit(old_limit)
    if not rows:
        return []
    header = rows[0]
    docs = []
    for index, row in enumerate(rows[1:], 1):
        if not any(value.strip() for value in row):
            continue
        content = ' | '.join(
            f'{header[i] if i < len(header) else i + 1}: {value}'
            for i, value in enumerate(row)
        )
        docs.append(Document(page_content=content,
                             metadata={'record': index, 'csv_record': index, 'columns': len(row)}))
    return docs


def _pdf_docs(path):
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise IngestionError('pdf_encrypted', 'PDF 已加密，暂不支持读取。')
        docs = []
        for number, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ''
                gap = not text.strip()
            except Exception:
                text, gap = '', True
            docs.append(Document(page_content=text,
                                 metadata={'page_number': number, 'gap': gap}))
        return docs
    except PdfReadError:
        raise IngestionError('pdf_invalid', 'PDF 已损坏或格式无效。') from None


def _docx_docs(path, granularity):
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError('Duplicate ZIP entries')
            if any(entry.flag_bits & 1 for entry in entries):
                raise ValueError('Encrypted ZIP')
            if sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES:
                raise IngestionError('docx_too_large', 'DOCX 解压总量超过 100 MiB。')
            required = {'[Content_Types].xml', '_rels/.rels', 'word/document.xml'}
            if not required.issubset(names):
                raise ValueError('Not a Word package')
            types = _xml(archive.read('[Content_Types].xml'))
            if not any(node.get('PartName') == '/word/document.xml' and node.get('ContentType') ==
                       'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'
                       for node in types):
                raise ValueError('Invalid Word content type')
            relations = _xml(archive.read('_rels/.rels'))
            if not any(node.get('Type', '').endswith('/officeDocument') and
                       node.get('Target') in {'word/document.xml', '/word/document.xml'} and
                       node.get('TargetMode') != 'External' for node in relations):
                raise ValueError('Invalid Word relationship')
            total = 0
            for entry in entries:
                with archive.open(entry) as stream:
                    while chunk := stream.read(65536):
                        total += len(chunk)
                        if total > MAX_EXPANDED_BYTES:
                            raise IngestionError('docx_too_large', 'DOCX 解压总量超过 100 MiB。')
            root = _xml(archive.read('word/document.xml'))
            body = root.find(W + 'body') if root.tag == W + 'document' else None
            if body is None:
                raise ValueError('Missing Word body')
            docs = []
            paragraph = table = 0
            for element in body:
                if element.tag == W + 'p':
                    paragraph += 1
                    docs.append(Document(page_content=_docx_text(element),
                                         metadata={'paragraph': paragraph}))
                elif element.tag == W + 'tbl':
                    table += 1
                    for row, tr in enumerate(element.findall(W + 'tr'), 1):
                        text = ' | '.join(
                            '\n'.join(_docx_text(p) for p in cell.iter(W + 'p'))
                            for cell in tr.findall(W + 'tc'))
                        docs.append(Document(page_content=text,
                                             metadata={'table': table, 'row': row}))
            if granularity == 'ingest':
                joined = '\n'.join(doc.page_content for doc in docs)
                return [Document(page_content=joined,
                                 metadata={'paragraph_count': paragraph,
                                           'table_count': table})]
            return docs
    except IngestionError:
        raise
    except (zipfile.BadZipFile, ET.ParseError, ValueError, KeyError,
            RuntimeError, NotImplementedError):
        raise IngestionError('docx_invalid', 'DOCX 已损坏、加密或不是有效的 Word 文档包。') from None


def _xlsx_docs(path):
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter
    book = load_workbook(str(path), read_only=True, data_only=False)
    values = load_workbook(str(path), read_only=True, data_only=True)
    docs = []
    try:
        for sheet in book:
            header = None
            for index, (row, display) in enumerate(zip(
                    sheet.iter_rows(values_only=True),
                    values[sheet.title].iter_rows(values_only=True)), 1):
                if header is None:
                    header = row
                if not any(value is not None for value in row):
                    continue
                parts = []
                for i, value in enumerate(row):
                    shown = display[i] if i < len(display) and display[i] is not None else value
                    part = f'{header[i] if i < len(header) else i + 1}: {shown}'
                    if isinstance(value, str) and value.startswith('='):
                        part += f'（公式：{value}）'
                    parts.append(part)
                docs.append(Document(
                    page_content=' | '.join(parts),
                    metadata={'sheet_name': sheet.title,
                              'cell_range': f'A{index}:{get_column_letter(max(1, len(row)))}{index}'}))
        return docs
    finally:
        book.close()
        values.close()


def _pptx_docs(path):
    from pptx import Presentation
    presentation = Presentation(str(path))
    docs = []
    for index, slide in enumerate(presentation.slides, 1):
        values = []
        for shape in slide.shapes:
            if hasattr(shape, 'text') and shape.text.strip():
                values.append(shape.text.strip())
        if values:
            docs.append(Document(page_content='\n'.join(values),
                                 metadata={'slide_number': index}))
    return docs


def parse_document(path, *, granularity='ingest'):
    """按扩展名解析；read 保留引用位置，ingest 合并为可切分文本/块。"""
    if granularity not in {'read', 'ingest'}:
        raise ValueError('granularity must be read or ingest')
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return _pdf_docs(path)
    if suffix in DOCX_EXTENSIONS:
        return _docx_docs(path, granularity)
    if suffix in CSV_EXTENSIONS:
        text = decode_text_bytes(path.read_bytes())
        return _csv_docs(text, suffix, granularity)
    if suffix in TEXT_EXTENSIONS:
        text = decode_text_bytes(path.read_bytes())
        if suffix == '.json':
            return _json_docs(text, suffix, granularity)
        return _line_docs(text, suffix, granularity)
    if suffix in OFFICE_EXTENSIONS:
        if suffix == '.xlsx':
            return _xlsx_docs(path)
        return _pptx_docs(path)
    raise IngestionError('attachment_format_unsupported',
                         f'不支持的附件格式：{suffix or "无扩展名"}。')


def parse_text_attachment(path):
    """兼容旧附件解析入口；固定返回 read 粒度。"""
    return parse_document(path, granularity='read')
