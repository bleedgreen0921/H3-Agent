"""轻量证据展示，不加载检索服务。"""
from ai_assistant.knowledge.metadata import location


def render_evidence(docs, *, limit=24000):
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(f"片段：{doc.id}\n来源：{m.get('source_name', '')} ({m.get('source_url') or m.get('source_uri', '')})\n"
                     f"文档：{m.get('document_id', '')}；版本：{m.get('document_version_id', '')}；构建：{m.get('index_version', '')}\n"
                     f"定位：{location(m)}\n{doc.page_content}")
    content = '\n\n'.join(parts) or '未找到相关证据。'
    marker = '\n[展示已截断；artifact 保留完整证据]'
    return content if len(content) <= limit else content[:limit-len(marker)] + marker
