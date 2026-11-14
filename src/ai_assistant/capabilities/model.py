"""能力描述、意图与暴露计划；不持有具体工具实现。"""
from dataclasses import dataclass, field
from pathlib import Path

from ai_assistant.capabilities.intent import QueryIntent


@dataclass(frozen=True)
class CapabilityDescriptor:
    """统一描述工具或 Skill，供注册表和能力选择使用。"""

    # 能力标识、展示信息及来源；original_name 保留服务侧原名。
    id: str
    kind: str                     # tool | skill
    name: str
    original_name: str
    description: str
    provider: str
    server: str | None = None
    identity: str | None = None

    # 通用语义标签用于匹配请求意图；读写标记参与能力筛选。
    domains: frozenset[str] = frozenset()
    actions: frozenset[str] = frozenset()
    objects: frozenset[str] = frozenset()
    side_effect: str = 'read'

    # 估算工具 schema 与 Skill 目录说明的 token 开销。
    schema_tokens: int = 0
    guidance_tokens: int = 0

    # 核心能力标记，以及与该描述关联的工具或 Skill 对象。
    core: bool = False
    write: bool = False
    tool: object | None = None
    skill: object | None = None

    def summary(self):
        # 输出精简摘要，避免记录完整工具对象或过长描述。
        return {'id': self.id, 'kind': self.kind, 'name': self.name,
                'original_name': self.original_name, 'provider': self.provider,
                'description': self.description[:120], 'write': self.write}


@dataclass
class ExposurePlan:
    """保存本轮模型可见能力及后续扩展所需的约束。"""

    # Skill 出现在目录中不代表已激活；分别记录目录和激活状态。
    tool_ids: list[str] = field(default_factory=list)
    skill_directory_ids: list[str] = field(default_factory=list)
    active_skill_ids: list[str] = field(default_factory=list)

    # 记录计划来源、识别意图和预计占用的 token。
    reason: str = 'retrieval'
    intent: QueryIntent | None = None
    tool_schema_tokens: int = 0
    skill_directory_tokens: int = 0

    # 规划任务可锁定资源身份与效果，扩展时继续遵守同一约束。
    server: str | None = None
    identity: str | None = None
    effect: str | None = None
    max_schema_tokens: int = 0
    forbid_create: bool = False


@dataclass
class CapabilityTurnState:
    """一轮对话的能力状态；plan 在首次模型调用前生成。"""

    registry: object | None = None
    plan: ExposurePlan | None = None

    # 累计能力搜索次数及工具执行前检查失败次数。
    search_calls: int = 0
    preflight_failures: int = 0

    # 规划中的只读任务不能通过 search_tools 扩展到写能力。
    read_only: bool = False

    @property
    def initialized(self):
        # 有计划即表示能力规划已经初始化。
        return self.plan is not None


@dataclass(frozen=True)
class CapabilityPolicy:
    """部署级策略；不是应用分支。"""

    # 默认核心工具名称；实际可用性仍取决于注册表中的工具。
    core_tools: tuple[str, ...] = (
        'read_document_attachment', 'read_text', 'list_directory', 'search_text',
        'web_search', 'web_fetch', 'search_knowledge', 'search_tools',
        'search_skills', 'load_skill', 'read_skill_resource', 'unload_skill',
        'resolve_datetime', 'read_tool_result',
    )

    # 工具 schema 预算、候选数量和能力搜索次数的默认上限。
    max_schema_tokens: int = 25000
    top_k: int = 8
    max_search_calls: int = 3


def resolve_log_path(value, base):
    """将日志路径解析为绝对路径；未配置时返回 None。"""
    if not value:
        return None
    path = Path(value).expanduser()

    # 相对路径以调用方提供的基准目录为准。
    return (path if path.is_absolute() else Path(base) / path).resolve()
