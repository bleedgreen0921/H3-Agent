"""Web 展示层 Markdown 渲染与安全边界。"""
from __future__ import annotations

from html import escape
import re


def safe_html(text: str) -> str:
    """渲染受控 Markdown，并给表格加横向滚动容器。"""
    import markdown
    rendered = markdown.markdown(escape(text or ""), extensions=[
        "fenced_code", "tables", "nl2br", "sane_lists"])
    # 原始 HTML 已转义；这里再移除可能由扩展产生的危险标签。
    rendered = re.sub(r'<(img|script|style|iframe)[^>]*>', "", rendered,
                      flags=re.IGNORECASE)
    rendered = re.sub(r'href="(?!https?://|/|#|mailto:)[^"]*"',
                      'href="#"', rendered)
    # 宽表在消息气泡内自行滚动，不挤压正文布局。
    rendered = re.sub(r'(<table\b[^>]*>)', r'<div class="table-wrap">\1', rendered)
    rendered = rendered.replace("</table>", "</table></div>")
    return rendered
