"""通用命令协议：解析、权限、处理结果与注册表；业务 handler 可逐步迁移。"""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CommandSpec:
    """声明命令语法、处理函数及调度标记。"""

    # 基本命令信息与参数解析器。
    name: str
    usage: str
    parse: Callable[[str], Any]
    handler: Callable | None = None
    aliases: tuple[str, ...] = ()

    # 由渠道工作循环解释这些标记，例如启动轮次或抢占当前任务。
    immediate: bool = True
    turn: bool = False
    preempt: bool = False
    progress: bool = True
    recover_rag: bool = False
    turn_check: Callable[[Any], bool] | None = None


@dataclass
class CommandContext:
    """传给命令处理函数的输入、调用者及解析结果。"""
    worker: object
    row: dict
    caller: object
    text: str
    value: Any = None
    recover: bool = False


@dataclass
class CommandResult:
    """命令可直接返回回复，也可请求启动携带参数的模型轮次。"""
    reply: str | None = None
    turn_text: str | None = None
    skill_ids: list[str] = field(default_factory=list)
    start_turn: bool = False
    recover: bool = False
    turn_kwargs: dict = field(default_factory=dict)


class CommandRegistry:
    """登记命令和别名，并将输入解析为命令及其参数。"""

    def __init__(self, specs=()):
        self._specs = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: CommandSpec):
        # 命令名和别名共用命名空间，重复时明确报错。
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

        # 非斜杠输入交给普通对话流程；未知命令返回可读错误。
        if not text.startswith('/'):
            return None, None
        head, _, rest = text.partition(' ')
        spec = self.get(head)
        if spec is None:
            return None, ('error', '未支持的指令。')

        # 参数解析失败时只返回用法，不暴露原始异常。
        try:
            return spec, spec.parse(rest.strip())
        except (ValueError, TypeError, AttributeError):
            return None, ('error', '用法：' + spec.usage)

    def names(self):
        return tuple(sorted(self._specs))

    def immutable(self):
        return tuple(self._specs.values())
