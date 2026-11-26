"""兼容入口：附件读取的实际解析在 ai_assistant.documents.parsing。"""
from ai_assistant.documents.parsing import parse_document, parse_text_attachment

__all__ = ['parse_document', 'parse_text_attachment']
