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
    preempt: bool = False
    progress: bool = True
    recover_rag: bool = False
    turn_check: Callable[[Any], bool] | None = None


@dataclass
class CommandContext:
    worker: object
    row: dict
    caller: object
    text: str
    value: Any = None
    recover: bool = False


@dataclass
class CommandResult:
    reply: str | None = None
    turn_text: str | None = None
    skill_ids: list[str] = field(default_factory=list)
    start_turn: bool = False
    recover: bool = False
    turn_kwargs: dict = field(default_factory=dict)


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

    def parse(self, text):
        """返回 (spec | None, value | error)。解析失败时 spec 为 None。"""
        text = (text or '').strip()
        if not text.startswith('/'):
            return None, None
        head, _, rest = text.partition(' ')
        spec = self.get(head)
        if spec is None:
            return None, ('error', '未支持的指令。')
        try:
            return spec, spec.parse(rest.strip())
        except (ValueError, TypeError, AttributeError):
            return None, ('error', '用法：' + spec.usage)

    def names(self):
        return tuple(sorted(self._specs))

    def immutable(self):
        return tuple(self._specs.values())
