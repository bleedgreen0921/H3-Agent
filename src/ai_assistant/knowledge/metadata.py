"""Stable semantic metadata, independent of parser extraction quality."""
import json
from collections.abc import Mapping
from langchain_core.documents import Document

RANGES = ('page', 'line', 'block')
FILTERS = {'document_id', 'media_type', 'tags', 'source_name'}


def normalize(document: Document, *, identity: dict, parser: str,
              mapping: Mapping[str, str] | None = None, zero_based: tuple[str, ...] = ()) -> Document:
    """Only explicitly mapped positions acquire shared semantics.

    Unstructured page_number is physical, one-based. Other loaders must declare
    their mapping and zero-based fields; arbitrary parser IDs never become IDs.
    """
    raw = json.loads(json.dumps(document.metadata, ensure_ascii=False, default=str))
    metadata = dict(identity)
    metadata.update(metadata_schema_version=1, parser_name=parser, parser_metadata=raw)
    mapped = {'page_start': 'page_number', 'page_end': 'page_number'} if parser == 'unstructured' else {}
    mapped.update(mapping or {})
    if parser == 'unstructured' and isinstance(raw.get('element_id'), str):
        metadata['element_refs'] = [raw['element_id']]
    for target, source in mapped.items():
        if source not in raw or raw[source] is None:
            continue
        value = raw[source]
        if target in {f'{r}_{s}' for r in RANGES for s in ('start', 'end')} or target == 'csv_record':
            if type(value) is not int:
                raise ValueError(f'{target} must be an integer')
            value += int(source in zero_based)
            if value < 1:
                raise ValueError(f'{target} must be one-based')
        elif target in ('heading_path', 'element_refs'):
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise ValueError(f'{target} must be a string list')
        elif target == 'page_label':
            if not isinstance(value, str):
                raise ValueError('page_label must be a string')
        else:
            raise ValueError(f'Unsupported semantic field: {target}')
        metadata[target] = value
    for name in RANGES:
        start, end = metadata.get(f'{name}_start'), metadata.get(f'{name}_end')
        if start is not None and end is not None and end < start:
            raise ValueError('Invalid inclusive range')
    for key in ('knowledge_base_id', 'document_id', 'document_version_id', 'source_uri',
                'source_name', 'media_type', 'parse_config_fingerprint', 'index_version'):
        if not isinstance(metadata.get(key), str):
            raise ValueError(f'{key} must be a string')
    for key in ('tags', 'heading_path'):
        value = metadata.setdefault(key, [])
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ValueError(f'{key} must be a string list')
    return Document(page_content=document.page_content, metadata=metadata)


def validate_filters(filters: dict | None) -> dict:
    filters = dict(filters or {})
    if filters.keys() - FILTERS:
        raise ValueError('Unsupported knowledge filter')
    for key, value in filters.items():
        if key == 'tags':
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise ValueError('tags must be a string list')
        elif not isinstance(value, str):
            raise ValueError(f'{key} must be a string')
    return filters


def location(metadata: dict) -> str:
    parts = []
    for key, label in (('sheet_name','工作表'),('cell_range','单元格'),('table_name','数据表'),('record_id','记录')):
        if metadata.get(key):
            parts.append(f'{label} {metadata[key]}')
    for name, label in (('page', '页'), ('line', '行'), ('block', '块')):
        if f'{name}_start' in metadata:
            parts.append(f'{label} {metadata[f"{name}_start"]}–{metadata.get(f"{name}_end", metadata[f"{name}_start"])}')
    if 'page_label' in metadata:
        parts.append(f'页标签 {metadata["page_label"]}')
    if 'csv_record' in metadata:
        parts.append(f'CSV 记录 {metadata["csv_record"]}')
    return '；'.join(parts) or '未提供精确定位'


def terms(text: str) -> str:
    import jieba
    import re
    pieces = re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\x00-\x7f]+', text)
    tokens = []
    for piece in pieces:
        tokens.extend(jieba.cut(piece) if re.search(r'[\u3400-\u9fff]', piece) else [piece])
    return ' '.join(word.lower() for word in tokens if re.search(r'\w', word))
