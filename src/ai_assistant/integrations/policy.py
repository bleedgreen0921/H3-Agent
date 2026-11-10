"""Caller context is created by trusted entrypoints, never by model arguments."""
from dataclasses import dataclass, field
from typing import Literal
from langchain_core.tools import ToolException


@dataclass(frozen=True)
class Caller:
    channel: str = 'cli'
    owner: bool = True
    binding: str | None = None

    scope: Literal['local', 'private'] | None = field(default=None, kw_only=True)

    def __post_init__(self):
        # 外部入口必须显式提供可信范围和平台命名空间绑定。
        if self.channel == 'cli' and self.scope is None:
            object.__setattr__(self, 'scope', 'local')
        if self.channel == 'feishu_group':
            raise ValueError('群聊入口已移除。')
        if self.scope not in {'local', 'private'}:
            raise ValueError('外部入口必须明确会话范围。')
        if self.channel != 'cli' and (self.scope == 'local' or not self.binding or ':' not in self.binding):
            raise ValueError('外部入口必须提供私聊范围及平台命名空间绑定。')

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


def authorize(caller, server, tool):
    """只允许可信主人调用显式开放的工具；写入另受服务开关限制。"""
    caller.__post_init__()
    if not caller.owner:
        raise ToolException('Only the configured owner can use office tools.')
    if tool not in server.tools:
        raise ToolException('Tool is outside the configured allow list.')
    if tool in server.write_tools and not server.allow_write:
        raise ToolException('Writes are disabled for this server.')
