"""Caller context is created by trusted entrypoints, never by model arguments."""
from dataclasses import dataclass, field
from typing import Literal
from langchain_core.tools import ToolException


@dataclass(frozen=True)
class Caller:
    channel: str = 'cli'
    owner: bool = True
    binding: str | None = None
    resources: dict[str, tuple[str, ...]] = field(default_factory=dict)

    scope: Literal['local', 'private', 'shared'] | None = field(default=None, kw_only=True)
    skill_ids: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self):
        # 外部入口必须显式提供可信范围和平台命名空间绑定。
        if self.channel == 'cli' and self.scope is None:
            object.__setattr__(self, 'scope', 'local')
        if self.scope not in {'local', 'private', 'shared'}:
            raise ValueError('外部入口必须明确会话范围。')
        if self.channel != 'cli' and (self.scope == 'local' or not self.binding or ':' not in self.binding):
            raise ValueError('外部入口必须提供私人或共享范围及平台命名空间绑定。')

    @property
    def personal(self):
        return self.scope == 'local' or self.scope == 'private' and self.owner


def operation_principal(caller):
    """操作回执使用本地主体默认值；外部身份必须通过 Caller 校验。"""
    if caller is None:
        return 'local'
    if caller.channel == 'cli':
        return caller.binding or 'local'
    caller.__post_init__()
    return caller.binding


def authorize(caller, server_name, server, tool, arguments):
    if not caller.owner:
        raise ToolException('Only the configured owner can use office tools.')
    if tool not in server.tools:
        raise ToolException('Tool is outside the configured allow list.')
    if tool in server.write_tools and not server.allow_write:
        raise ToolException('Writes are disabled for this server.')
    if caller.scope != 'shared':
        return
    if not shared_service_allowed(server):
        raise ToolException('Private services are unavailable in groups.')
    allowed = caller.resources.get(server_name, ())
    paths = server.resource_arguments.get(tool, ())
    if not paths or not allowed:
        raise ToolException('Group resources must be explicitly configured for this tool.')
    for path in paths:
        value = arguments
        for key in path.split('.'):
            if not isinstance(value, dict) or key not in value:
                raise ToolException('Missing resource selector.')
            value = value[key]
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(v, str) or v not in allowed for v in values):
            raise ToolException('Target resource is outside this group grant.')


def shared_service_allowed(server):
    """共享连接身份由服务适配验证；未知适配默认拒绝。"""
    if server.transport in {'feishu', 'feishu_files'}:
        from ai_assistant.integrations.feishu_adapter import shared_identity_allowed
        return shared_identity_allowed(server)
    return False
