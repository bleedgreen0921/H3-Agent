"""飞书工具族的模型上下文与最终输出投影策略。"""
from __future__ import annotations

from ai_assistant.integrations.tool_policy import (
    ACCOUNT_KEYS,
    ACCOUNT_PATHS,
    ACCOUNT_TEXT_PATTERNS,
    BUSINESS_KEYS,
    DEFAULT_REDACTION_POLICY,
    RedactionPolicy,
    RedactionRule,
    project_output_text,
    project_tool_message,
)


# 最终输出：所有账号字段和文本兜底都隐藏；任务昵称/来源不构成业务结果。
FEISHU_OUTPUT_POLICY = RedactionPolicy(
    keys=ACCOUNT_KEYS,
    paths=ACCOUNT_PATHS,
    omit_paths=(("origin",), ("creator",), ("node_creator",),
                ("log_id",), ("troubleshooter",), ("method_id",),
                ("error", "log_id"), ("error", "troubleshooter"),
                ("error", "helps"), ("error", "method_id")),
    patterns=ACCOUNT_TEXT_PATTERNS,
    preserve_keys=BUSINESS_KEYS,
)

# 文档上下文：保留业务 token、标题、时间和链接；隐藏创建人、账号字段和
# provider 排查信息。docx/wiki/drive 原先没有策略，导致 open_id 进入模型
# 上下文和 checkpoint，这里统一收口。
FEISHU_DOC_CONTEXT_POLICY = RedactionPolicy(
    keys=ACCOUNT_KEYS,
    paths=ACCOUNT_PATHS,
    omit_paths=(("origin",), ("creator",), ("node_creator",),
                ("log_id",), ("troubleshooter",), ("method_id",),
                ("error", "log_id"), ("error", "troubleshooter"),
                ("error", "helps"), ("error", "method_id")),
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

# 日历上下文额外隐藏组织者和参与者的身份字段。
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
    """按飞书工具族选择上下文策略；未知工具使用默认兜底。"""
    name = getattr(entry, "original_name", "")
    if name.startswith(("docx_", "wiki_", "drive_")):
        return FEISHU_DOC_CONTEXT_POLICY
    if name.startswith("task_v2_"):
        return FEISHU_TASK_CONTEXT_POLICY
    if name.startswith("contact_v3_"):
        return FEISHU_REFERENCE_CONTEXT_POLICY
    if name.startswith("calendar_v4_"):
        return FEISHU_CALENDAR_CONTEXT_POLICY
    return DEFAULT_REDACTION_POLICY


def project(entry, result, context=None, *, mode="context"):
    """投影工具结果；所有 MCP 结果都有默认兜底，文档族使用专门策略。"""
    # 最终输出使用统一策略；模型上下文按工具族选择策略。
    policy = FEISHU_OUTPUT_POLICY if mode == "output" else context_policy(entry)
    if policy is None:
        return result
    return project_tool_message(result, policy)


def project_output(text):
    """最终飞书文本使用与工具输出相同的分类。"""
    return project_output_text(text, FEISHU_OUTPUT_POLICY)


__all__ = [
    "FEISHU_CALENDAR_CONTEXT_POLICY", "FEISHU_DOC_CONTEXT_POLICY",
    "FEISHU_OUTPUT_POLICY", "FEISHU_REFERENCE_CONTEXT_POLICY",
    "FEISHU_TASK_CONTEXT_POLICY", "context_policy", "project", "project_output",
]
