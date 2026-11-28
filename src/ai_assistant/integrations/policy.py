"""Trusted local caller context; created by entrypoints, never by model arguments."""
from dataclasses import dataclass

from langchain_core.tools import ToolException


@dataclass(frozen=True)
class Caller:
    """本地入口的可信调用者；owner 仅作为外部工具授权的防御性开关。"""
    owner: bool = True


def operation_principal():
    """所有本地入口使用固定本地主体；不存在外部聊天身份。"""
    return 'local'


def authorize(caller, server, tool):
    """只允许可信主人调用显式开放的工具；写入另受服务开关限制。"""
    if not caller.owner:
        raise ToolException('Only the configured owner can use external tools.')
    if tool not in server.tools:
        raise ToolException('Tool is outside the configured allow list.')
    if tool in server.write_tools and not server.allow_write:
        raise ToolException('Writes are disabled for this server.')
