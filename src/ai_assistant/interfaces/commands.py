"""通用命令协议：解析、权限、处理结果与注册表；业务 handler 可逐步迁移。"""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CommandSpec:
    name: str
    usage: str
    parse: Callable[[str], Any]
    handler: Callable | None = None
    aliases: tuple[str, ...] = ()
    immediate: bool = True
    turn: bool = False
    legacy: bool = True


@dataclass
class CommandContext:
    worker: object
    row: dict
    caller: object
    text: str
    parsed: Any = None
    recover: bool = False


@dataclass
class CommandResult:
    handled: bool = True
    reply: str | None = None
    turn_text: str | None = None
    skill_ids: list[str] = field(default_factory=list)
    recover: bool = False


class CommandRegistry:
    def __init__(self, specs=()):
        self._specs = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: CommandSpec):
        for name in (spec.name, *spec.aliases):
            if name in self._specs:
                raise ValueError(f'duplicate command: {name}')
            self._specs[name] = spec
        return spec

    def get(self, name):
        return self._specs.get(name)

    def names(self):
        return tuple(sorted(self._specs))

    def immutable(self):
        return tuple(self._specs.values())
