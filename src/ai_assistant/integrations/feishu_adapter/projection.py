"""飞书工具族的模型上下文与最终输出投影策略。"""
from __future__ import annotations

from ai_assistant.integrations.tool_policy import (
    ACCOUNT_KEYS,
    ACCOUNT_PATHS,
    ACCOUNT_TEXT_PATTERNS,
    BUSINESS_KEYS,
    RedactionPolicy,
    RedactionRule,
    project_output_text,
    project_tool_message,
)


# 最终输出：所有账号字段和文本兜底都隐藏；任务昵称/来源不构成业务结果。
FEISHU_OUTPUT_POLICY = RedactionPolicy(
    keys=ACCOUNT_KEYS,
    paths=ACCOUNT_PATHS,
    omit_paths=(("origin",),),
    patterns=ACCOUNT_TEXT_PATTERNS,
    preserve_keys=BUSINESS_KEYS,
)

# 任务上下文：保留 GUID、时间、提醒和链接；隐藏创建人、成员 ID 和昵称。
FEISHU_TASK_CONTEXT_POLICY = RedactionPolicy(
    keys=ACCOUNT_KEYS,
    paths=ACCOUNT_PATHS,
    omit_paths=(("creator",), ("origin",), ("members", "*", "id")),
    patterns=ACCOUNT_TEXT_PATTERNS,
    preserve_keys=BUSINESS_KEYS,
)

# 通讯录是后续任务分配的引用来源：保留 open_id/union_id，隐藏其他账号字段。
FEISHU_REFERENCE_CONTEXT_POLICY = RedactionPolicy(
    keys=frozenset({"user_id", "email", "mobile", "phone", "employee_no"}),
    paths=(),
    patterns=(),
    preserve_keys=BUSINESS_KEYS,
)

FEISHU_CALENDAR_CONTEXT_POLICY = RedactionPolicy(
    keys=ACCOUNT_KEYS,
    paths=ACCOUNT_PATHS + (
        RedactionRule(("organizer", "id")),
        RedactionRule(("attendees", "*", "id")),
    ),
    patterns=ACCOUNT_TEXT_PATTERNS,
    preserve_keys=BUSINESS_KEYS,
)


def context_policy(entry):
    """按飞书工具族选择上下文策略；未知工具保持原样。"""
    name = getattr(entry, "original_name", "")
    if name.startswith("task_v2_"):
        return FEISHU_TASK_CONTEXT_POLICY
    if name.startswith("contact_v3_"):
        return FEISHU_REFERENCE_CONTEXT_POLICY
    if name.startswith("calendar_v4_"):
        return FEISHU_CALENDAR_CONTEXT_POLICY
    return None


def project(entry, result, context=None, *, mode="context"):
    """投影工具结果；context 模式只处理需要安全的工具族。"""
    policy = FEISHU_OUTPUT_POLICY if mode == "output" else context_policy(entry)
    if policy is None:
        return result
    return project_tool_message(result, policy)


def project_output(text):
    """最终飞书文本使用与工具输出相同的分类。"""
    return project_output_text(text, FEISHU_OUTPUT_POLICY)


__all__ = [
    "FEISHU_CALENDAR_CONTEXT_POLICY", "FEISHU_OUTPUT_POLICY",
    "FEISHU_REFERENCE_CONTEXT_POLICY", "FEISHU_TASK_CONTEXT_POLICY",
    "context_policy", "project", "project_output",
]
