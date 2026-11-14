"""附件专用标准库解析；由可终止的解析子进程调用。"""
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

from langchain_core.documents import Document

from ai_assistant.knowledge import IngestionError

# 限制 DOCX ZIP 的总展开字节数；W 为 Word 正文 XML 命名空间。
MAX_EXPANDED_BYTES = 100 * 1024**2
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def _xml(raw):
    # 禁止实体声明，避免实体展开绕过 ZIP 大小限制。
    declarations = raw.replace(b'\x00', b'').upper()
    if b'<!DOCTYPE' in declarations or b'<!ENTITY' in declarations:
        raise ValueError('XML entities are unsupported')
    return ET.fromstring(raw)


def _text(element):
    """按文档顺序提取文本、制表符和换行。"""
    return ''.join(node.text or '' if node.tag == W + 't' else
                   '\t' if node.tag == W + 'tab' else '\n'
                   for node in element.iter() if node.tag in {W+'t', W+'tab', W+'br', W+'cr'})


def parse_text_attachment(path):
    """解析 Markdown 或 DOCX 附件，返回带位置的 Document 列表。"""
    path = Path(path)

    # Markdown 仅接受 UTF-8/BOM 文本，逐行保留原始换行和行号。
    if path.suffix == '.md':
        try:
            text = path.read_bytes().decode('utf-8-sig')
            if '\x00' in text or text.startswith(('%PDF-', 'PK\x03\x04')):
                raise ValueError('Binary content')
        except (UnicodeError, ValueError):
            raise IngestionError('markdown_invalid', 'Markdown 必须是 UTF-8 文本，不能是二进制附件。') from None
        return [Document(page_content=line, metadata={'line': number})
                for number, line in enumerate(text.splitlines(keepends=True), 1)]

    # DOCX 是 ZIP 包：先验证成员、解压预算及 Word 文档关系。
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

            # 流式校验所有成员的实际解压大小和 CRC，不解压到磁盘。
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

            # 只提取正文段落及表格行；记录段落号或表格、行号供引用定位。
            docs = []
            paragraph = table = 0
            for element in body:
                if element.tag == W + 'p':
                    paragraph += 1
                    docs.append(Document(page_content=_text(element), metadata={'paragraph': paragraph}))
                elif element.tag == W + 'tbl':
                    table += 1
                    for row, tr in enumerate(element.findall(W + 'tr'), 1):
                        text = ' | '.join('\n'.join(_text(p) for p in cell.iter(W + 'p'))
                                          for cell in tr.findall(W + 'tc'))
                        docs.append(Document(page_content=text, metadata={'table': table, 'row': row}))
            return docs
    except IngestionError:
        raise
    except (zipfile.BadZipFile, ET.ParseError, ValueError, KeyError, RuntimeError, NotImplementedError):
        # 不将底层包解析异常直接暴露给用户。
        raise IngestionError('docx_invalid', 'DOCX 已损坏、加密或不是有效的 Word 文档包。') from None
