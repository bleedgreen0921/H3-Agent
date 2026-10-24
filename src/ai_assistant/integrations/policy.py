"""Caller context is created by trusted entrypoints, never by model arguments."""
from dataclasses import dataclass, field
from typing import Literal
from langchain_core.tools import ToolException


@dataclass(frozen=True)
class Caller:
    channel: Literal['cli', 'feishu_private', 'feishu_group'] = 'cli'
    owner: bool = True
    binding: str = 'local'
    resources: dict[str, tuple[str, ...]] = field(default_factory=dict)


def authorize(caller, server_name, server, tool, arguments):
    if not caller.owner:
        raise ToolException('Only the configured owner can use office tools.')
    if tool not in server.tools:
        raise ToolException('Tool is outside the configured allow list.')
    if tool in server.write_tools and not server.allow_write:
        raise ToolException('Writes are disabled for this server.')
    if caller.channel != 'feishu_group':
        return
    if server.identity != 'bot' or server.transport != 'feishu':
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
