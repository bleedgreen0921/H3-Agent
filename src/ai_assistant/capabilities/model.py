"""能力描述、意图与暴露计划；不持有具体工具实现。"""
from dataclasses import dataclass, field
from pathlib import Path

from ai_assistant.capabilities.intent import QueryIntent


@dataclass(frozen=True)
class CapabilityDescriptor:
    id: str
    kind: str                     # tool | skill
    name: str
    original_name: str
    description: str
    provider: str
    domains: frozenset[str] = frozenset()
    actions: frozenset[str] = frozenset()
    objects: frozenset[str] = frozenset()
    side_effect: str = 'read'
    schema_tokens: int = 0
    guidance_tokens: int = 0
    core: bool = False
    write: bool = False
    tool: object | None = None
    skill: object | None = None

    def summary(self):
        return {'id': self.id, 'kind': self.kind, 'name': self.name,
                'original_name': self.original_name, 'provider': self.provider,
                'description': self.description[:120], 'write': self.write}


@dataclass
class ExposurePlan:
    tool_ids: list[str] = field(default_factory=list)
    skill_directory_ids: list[str] = field(default_factory=list)
    active_skill_ids: list[str] = field(default_factory=list)
    reason: str = 'retrieval'
    intent: QueryIntent | None = None
    tool_schema_tokens: int = 0
    skill_directory_tokens: int = 0


@dataclass
class CapabilityTurnState:
    """一轮对话的能力状态；plan 在首次模型调用前生成。"""
    registry: object | None = None
    plan: ExposurePlan | None = None
    search_calls: int = 0
    preflight_failures: int = 0

    @property
    def initialized(self):
        return self.plan is not None


@dataclass(frozen=True)
class CapabilityPolicy:
    """部署级策略；不是应用分支。"""
    core_tools: tuple[str, ...] = (
        'read_document_attachment', 'read_text', 'list_directory', 'search_text',
        'web_search', 'web_fetch', 'search_knowledge', 'search_tools',
        'search_skills', 'load_skill', 'read_skill_resource', 'unload_skill',
        'resolve_datetime',
    )
    max_schema_tokens: int = 25000
    top_k: int = 8
    max_search_calls: int = 3


def resolve_log_path(value, base):
    if not value:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()
