"""可组合的模型提示；核心只提供通用规则，领域规则由组件贡献。"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PromptSection:
    """可按条件启用、排序的独立提示片段。"""
    key: str
    priority: int
    text: str
    when: Callable[[object], bool] | None = None


def compose_prompt(base, sections=(), context=None):
    """筛选已启用片段并按优先级与键稳定拼接。"""
    selected = [section for section in sections
                if section.text and (section.when is None or section.when(context))]
    selected.sort(key=lambda section: (section.priority, section.key))
    return '\n\n'.join([base.strip(), *(section.text.strip() for section in selected)]).strip()
