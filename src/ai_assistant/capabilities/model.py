"""能力描述、显式元数据与暴露计划；不持有具体工具实现。"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ToolMetadata:
    """内置工具必须显式声明的能力边界元数据。"""

    # 读写效果与资源类型只来自声明，不能从名称或描述推导。
    side_effect: Literal['read', 'write']
    resource_kinds: frozenset[str] = frozenset()  # document | file | record
    append_capable: bool = False
    creates: bool = False
    available: bool = True


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

    # 读写、资源类型和可用性必须由装配方显式提供。
    side_effect: str | None = None
    resource_kinds: frozenset[str] = frozenset()
    append_capable: bool = False
    creates: bool = False
    available: bool = True

    # 估算工具 schema 与 Skill 目录说明的 token 开销。
    schema_tokens: int = 0
    guidance_tokens: int = 0

    # 核心能力标记，以及与该描述关联的工具或 Skill 对象。
    core: bool = False
    tool: object | None = None
    skill: object | None = None

    @property
    def write(self):
        """读写判断的唯一来源是显式 side_effect。"""
        return self.side_effect == 'write'

    def summary(self):
        # 输出精简摘要，避免记录完整工具对象或过长描述。
        return {'id': self.id, 'kind': self.kind, 'name': self.name,
                'original_name': self.original_name, 'provider': self.provider,
                'description': self.description[:120],
                'effect': self.side_effect, 'write': self.write,
                'resource_kinds': sorted(self.resource_kinds),
                'available': self.available}


@dataclass
class ExposurePlan:
    """保存本轮模型可见能力及后续扩展所需的约束。"""

    # Skill 出现在目录中不代表已激活；分别记录目录和激活状态。
    tool_ids: list[str] = field(default_factory=list)
    skill_directory_ids: list[str] = field(default_factory=list)
    active_skill_ids: list[str] = field(default_factory=list)

    # 记录计划来源、选择结果和预计占用的 token。
    reason: str = 'core'
    selection_source: str = 'core'  # core | llm
    candidate_tool_count: int = 0
    candidate_skill_count: int = 0
    requires_write: bool = False
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


def resolve_log_path(value, base):
    """将日志路径解析为绝对路径；未配置时返回 None。"""
    if not value:
        return None
    path = Path(value).expanduser()

    # 相对路径以调用方提供的基准目录为准。
    return (path if path.is_absolute() else Path(base) / path).resolve()
