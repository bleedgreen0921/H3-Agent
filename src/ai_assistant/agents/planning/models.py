"""规划模式的数据契约、状态和稳定状态集合。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_assistant.agents.outcomes import TurnState

BLOCKING = frozenset({'failed', 'unknown', 'blocked', 'cancelled', 'needs_input'})


class PlanValidationError(ValueError):
    """计划结构无效；只包含安全摘要，不包含模型原始输出。"""


class ResourceRef(BaseModel):
    """用户请求中的稳定资源引用；计划任务只引用 ID，不复制长链接。"""

    model_config = ConfigDict(extra='forbid')

    id: str = Field(pattern=r'^r[1-9][0-9]*$')
    kind: Literal['feishu_wiki', 'feishu_doc', 'feishu_sheet', 'feishu_base',
                  'file', 'url', 'other'] = 'other'
    url: str | None = Field(default=None, max_length=8192)
    token: str | None = Field(default=None, max_length=512)
    label: str | None = Field(default=None, max_length=200)
    server: str | None = Field(default=None, max_length=128)
    identity: Literal['user', 'bot'] | None = None


class PlanTask(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    id: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]{0,31}$')
    description: str = Field(min_length=1, max_length=2000)
    deps: list[str] = Field(default_factory=list, max_length=12)
    intent: str = Field(default='', max_length=800)
    expected_output: str = Field(default='', max_length=1200)
    effect: Literal['read', 'write'] = Field(default='read', alias='side_effect')
    resources: list[str] = Field(default_factory=list, max_length=12)
    creates: bool = False
    append: bool = False
    replaces: list[str] = Field(default_factory=list, max_length=12)


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid')

    goal: str = Field(min_length=1, max_length=2000)
    tasks: list[PlanTask] = Field(default_factory=list, max_length=20)
    questions: list[str] = Field(default_factory=list, max_length=12)


class PlanIntent(BaseModel):
    """规划模式路由器：只判断用户最后一句话的意图。"""

    model_config = ConfigDict(extra='forbid')

    intent: Literal['execute', 'exit', 'status', 'continue']
    confidence: float = Field(ge=0, le=1)


class TaskExecutionPlan(BaseModel):
    """任务在派发前解析一次的能力、效果和资源绑定。"""

    model_config = ConfigDict(extra='forbid')

    task_id: str
    effect: Literal['read', 'write']
    tool_ids: list[str] = Field(default_factory=list)
    schema_tokens: int = 0
    resources: list[ResourceRef] = Field(default_factory=list)
    server: str | None = None
    identity: Literal['user', 'bot'] | None = None
    forbid_create: bool = False
    reason: str = ''


class TaskReport(BaseModel):
    """任务 agent 的最终输出契约：用户摘要与下游产物分离。"""

    model_config = ConfigDict(extra='forbid')

    summary: str = Field(min_length=1, max_length=600)
    output: str = Field(default='', max_length=200_000)
    references: list[str] = Field(default_factory=list, max_length=20)
    questions: list[str] = Field(default_factory=list, max_length=12)


def merge_task_results(left, right):
    """Reducer：同一计划内合并任务结果，新计划整体替换旧结果。"""
    left = left or {}
    right = right or {}
    if not right:
        return left
    if left.get('plan_id') != right.get('plan_id'):
        return right
    items = dict(left.get('items') or {})
    items.update(right.get('items') or {})
    return {'plan_id': right['plan_id'], 'items': items}


class PlanState(TurnState):
    """父图状态；planning 跨轮保留，task_results 按 plan_id 隔离。"""

    planning: dict | None
    task_results: Annotated[dict, merge_task_results]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
