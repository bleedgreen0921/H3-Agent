"""工具输入预检与模型可见输出投影的通用契约。

业务适配器只描述字段和规则；原始回执、校验事实和模型可见内容
在这一层明确分离，避免各个工具各自打脱敏补丁。
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from langchain_core.tools import ToolException


@dataclass(frozen=True)
class PreflightIssue:
    """尚未产生外部副作用时的可读校验问题。"""

    code: str
    message: str
    recoverable: bool = True
    field: str | None = None


class ToolPreflightError(ToolException):
    """预检失败；是否允许模型修正由 recoverable 决定。"""

    def __init__(self, issue: PreflightIssue):
        self.issue = issue
        super().__init__(issue.message)

    @property
    def code(self) -> str:
        return self.issue.code

    @property
    def recoverable(self) -> bool:
        return self.issue.recoverable

    @property
    def field(self) -> str | None:
        return self.issue.field


@dataclass(frozen=True)
class RedactionRule:
    """按路径后缀匹配敏感字段；`*` 匹配任意一层。"""

    path: tuple[str, ...]
    replacement: str = "[账号信息已隐藏]"


@dataclass(frozen=True)
class RedactionPolicy:
    """账号字段、业务字段和文本兜底规则的集合。"""

    keys: frozenset[str] = frozenset()
    paths: tuple[RedactionRule, ...] = ()
    omit_paths: tuple[tuple[str, ...], ...] = ()
    patterns: tuple[re.Pattern[str], ...] = ()
    replacement: str = "[账号信息已隐藏]"
    preserve_keys: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PolicySet:
    """同一工具族的上下文投影与最终输出投影。"""

    context: RedactionPolicy | None
    output: RedactionPolicy = RedactionPolicy()



# 默认账号字段：只覆盖直接字段；嵌套 id 由 paths 精确控制，避免误删业务 ID。
ACCOUNT_KEYS = frozenset({
    "open_id", "union_id", "user_id", "owner_id", "creator_id", "sender_id",
    "employee_no", "email", "mobile", "phone",
})

ACCOUNT_PATHS = (
    RedactionRule(("creator", "id")),
    RedactionRule(("members", "*", "id")),
    RedactionRule(("sender", "id")),
    RedactionRule(("attendees", "*", "id")),
    RedactionRule(("owner", "id")),
)

# 业务定位标识：即使包含 id 语义也不应被脱敏。
BUSINESS_KEYS = frozenset({
    "guid", "task_id", "document_id", "reminder_id", "operation_id",
    "outbox_id", "input_id", "sync_id", "job_id", "source_id", "build_id",
    "revision_id", "url", "start", "due", "reminders", "effect_id",
    "execution_fact", "registration", "task_model_view", "protected_receipt",
})

ACCOUNT_TEXT_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])(?:ou|on)_[A-Za-z0-9]+"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # 保守匹配中国大陆手机号；13 位时间戳不会命中。
    re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
)

DEFAULT_REDACTION_POLICY = RedactionPolicy(
    keys=ACCOUNT_KEYS,
    paths=ACCOUNT_PATHS,
    patterns=ACCOUNT_TEXT_PATTERNS,
    preserve_keys=BUSINESS_KEYS,
)

# 投影时需要重新解析为 JSON 的 artifact 字段（飞书工具把原始正文放在这里）。
_JSON_ARTIFACT_KEYS = frozenset({
    "content", "full_tool_content", "structured_content", "structuredContent", "raw",
})


def _path_matches(pattern: Sequence[str], path: Sequence[str]) -> bool:
    """路径后缀匹配，使嵌套在 data/task 下的同名字段也能命中。"""
    if len(pattern) > len(path):
        return False
    tail = path[-len(pattern):]
    return all(expected == "*" or expected == actual
               for expected, actual in zip(pattern, tail))


def _is_sensitive_path(path: Sequence[str], policy: RedactionPolicy) -> bool:
    if path and path[-1] in policy.preserve_keys:
        return False
    if path and path[-1] in policy.keys:
        return True
    return any(_path_matches(rule.path, path) for rule in policy.paths)


def _should_omit(path: Sequence[str], policy: RedactionPolicy) -> bool:
    return any(_path_matches(pattern, path) for pattern in policy.omit_paths)


def redact_text(text: str, policy: RedactionPolicy = DEFAULT_REDACTION_POLICY) -> str:
    """对非结构化文本做账号信息兜底替换。"""
    for pattern in policy.patterns:
        text = pattern.sub(policy.replacement, text)
    return text


def project_output_text(text: str, policy: RedactionPolicy = DEFAULT_REDACTION_POLICY) -> str:
    """最终用户可见文本的兜底投影。"""
    return redact_text(text, policy) if isinstance(text, str) else text


def redact_value(value: Any, policy: RedactionPolicy = DEFAULT_REDACTION_POLICY,
                 path: tuple[str, ...] = ()) -> Any:
    """递归复制并脱敏 dict/list/str；不修改原对象。"""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            child = (*path, str(key))
            if _should_omit(child, policy):
                continue
            if _is_sensitive_path(child, policy):
                result[key] = policy.replacement
            else:
                result[key] = redact_value(item, policy, child)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_value(item, policy, (*path, str(index)))
                for index, item in enumerate(value)]
    if isinstance(value, str):
        return redact_text(value, policy)
    return value


def project_content(value: Any, policy: RedactionPolicy = DEFAULT_REDACTION_POLICY) -> Any:
    """投影 ToolMessage 的 content；兼容 JSON 文本块和普通文本。"""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return redact_text(value, policy)
        if not isinstance(parsed, (dict, list)):
            return redact_text(value, policy)
        return json.dumps(redact_value(parsed, policy), ensure_ascii=False)
    if isinstance(value, list):
        blocks = []
        for block in value:
            if isinstance(block, Mapping):
                current = dict(block)
                if isinstance(current.get("text"), str):
                    current["text"] = project_content(current["text"], policy)
                blocks.append(current)
            else:
                blocks.append(project_content(block, policy))
        return blocks
    return redact_value(value, policy)


def project_artifact(value: Any, policy: RedactionPolicy = DEFAULT_REDACTION_POLICY,
                     path: tuple[str, ...] = ()) -> Any:
    """投影 artifact，同时保留执行事实和业务定位标识。"""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            child = (*path, str(key))
            if _should_omit(child, policy):
                continue
            if key in _JSON_ARTIFACT_KEYS:
                result[key] = project_content(item, policy)
            else:
                result[key] = project_artifact(item, policy, child)
        return result
    if isinstance(value, (list, tuple)):
        return [project_artifact(item, policy, (*path, str(index)))
                for index, item in enumerate(value)]
    if isinstance(value, str):
        return redact_text(value, policy)
    return value


def project_tool_message(message: Any, policy: RedactionPolicy = DEFAULT_REDACTION_POLICY) -> Any:
    """返回模型可见的 ToolMessage 副本；原始对象保持不变。"""
    content = project_content(getattr(message, "content", ""), policy)
    artifact = getattr(message, "artifact", None)
    projected_artifact = project_artifact(copy.deepcopy(artifact), policy) if artifact is not None else None
    update = {"content": content}
    if artifact is not None:
        update["artifact"] = projected_artifact
    model_copy = getattr(message, "model_copy", None)
    if model_copy is not None:
        return model_copy(update=update)
    # 兼容测试或轻量消息对象；不修改原对象。
    clone = copy.copy(message)
    for key, value in update.items():
        setattr(clone, key, value)
    return clone


__all__ = [
    "ACCOUNT_KEYS", "ACCOUNT_PATHS", "ACCOUNT_TEXT_PATTERNS", "BUSINESS_KEYS",
    "DEFAULT_REDACTION_POLICY", "PolicySet", "PreflightIssue", "RedactionPolicy", "RedactionRule",
    "ToolPreflightError", "project_artifact", "project_content", "project_output_text",
    "project_tool_message", "redact_text", "redact_value",
]
