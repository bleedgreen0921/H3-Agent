"""记忆协议：事实值与展示正文分离，不兼容旧格式。"""
from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FORMAT = 3
Cardinality = Literal['single', 'multiple']


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Evidence(MemoryModel):
    source_id: str
    quote: str = Field(min_length=1, max_length=1200)
    timestamp: datetime

    @field_validator('timestamp')
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None:
            raise ValueError('时间必须包含时区')
        return value


class EvidenceReference(MemoryModel):
    source_id: str
    quote: str = Field(min_length=1, max_length=1200)


class ProjectEntry(MemoryModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(default_factory=list)
    revision: int = 1
    evidence: list[Evidence] = Field(default_factory=list)


class AttributeEntry(MemoryModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(default_factory=list)
    cardinality: Cardinality
    revision: int = 1


class MemoryRequest(MemoryModel):
    category: Literal['preference', 'personal_fact', 'project', 'decision']
    topic: str = Field(min_length=1, max_length=100)
    values: list[str] = Field(min_length=1, max_length=30)
    cardinality: Cardinality
    quote: str = Field(min_length=1, max_length=1200)
    context_evidence: list[EvidenceReference] = Field(default_factory=list, max_length=5)
    action: Literal['add', 'replace', 'append', 'remove', 'invalidate'] = 'add'
    replace_all: bool = False
    memory_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    attribute_id: str | None = None
    project_id: str | None = None
    project_name: str | None = Field(default=None, min_length=1, max_length=100)
    scope: Literal['global', 'project', 'session'] = 'global'
    kind: Literal['fact', 'plan'] = 'fact'
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    @field_validator('values')
    @classmethod
    def canonical_values(cls, values):
        # 仅清理边界和集合重复；不猜测词义、不折叠大小写。
        cleaned = [v.strip() for v in values]
        if any(not v or len(v) > 600 for v in cleaned) or sum(map(len, cleaned)) > 1800:
            raise ValueError('事实值为空或超出长度限制')
        return sorted(set(cleaned))

    @field_validator('valid_from', 'valid_until')
    @classmethod
    def aware(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError('时间必须包含时区')
        return value

    @model_validator(mode='after')
    def validate_change(self):
        if self.cardinality == 'single' and len(self.values) != 1:
            raise ValueError('单值属性只能有一个值')
        if self.cardinality == 'single' and self.action in {'append', 'remove'}:
            raise ValueError('单值属性不支持集合增删')
        if self.valid_from and self.valid_until and self.valid_until <= self.valid_from:
            raise ValueError('生效结束时间必须晚于开始时间')
        if self.action == 'add' and (self.memory_id is not None or self.expected_revision is not None):
            raise ValueError('新增不能携带更新目标或 revision')
        if self.action != 'add' and (not self.memory_id or not self.expected_revision):
            raise ValueError('变更需要记忆 ID 和预期 revision')
        if self.action == 'replace' and self.cardinality == 'multiple' and not self.replace_all:
            raise ValueError('替换整个集合需要明确的 replace_all 授权语义')
        if self.scope == 'project' and not (self.project_id or self.project_name):
            raise ValueError('项目范围需要明确项目身份')
        return self


class MemoryChange(MemoryRequest):
    source_id: str


class MemoryRecord(MemoryModel):
    format: int = FORMAT
    id: str = Field(default_factory=lambda: str(uuid4()))
    revision: int = 1
    epoch: int
    category: str
    topic: str
    content: str
    values: list[str]
    cardinality: Cardinality
    attribute_id: str
    project_id: str | None = None
    scope: Literal['global', 'project', 'session']
    session_id: str | None = None
    kind: Literal['fact', 'plan'] = 'fact'
    status: Literal['active', 'inactive'] = 'active'
    evidence: list[Evidence]
    observed_at: datetime
    valid_from: datetime | None = None
    valid_until: datetime | None = None


REASONS = {
    'committed': '持久化完成', 'same_values': '事实值和生效规则未变化',
    'revision_mismatch': '记录版本已变化，请重新查询', 'identity_mismatch': '主体、属性或范围不一致',
    'cardinality_mismatch': '属性的单值/多值性质不一致', 'invalid_current_evidence': '当前来源无效或不允许保存',
    'invalid_change': '变更参数不符合记忆协议', 'search_required': '保存前请先查找相关记忆',
    'existing_attribute_requires_target': '属性已有记录，请使用原 ID 和 revision',
    'project_identity': '无法确定项目身份', 'attribute_identity': '无法确定属性身份',
    'project_name_evidence': '请保留用户明确命名的完整项目名称',
    'deleted': '该记忆已删除，正文和历史不可用', 'deleted_evidence': '旧证据不能恢复已删除记忆',
    'stale_epoch': '旧格式或清空前的候选已失效', 'stale_evidence': '来源早于记忆管理屏障',
    'commit_not_confirmed': '提交状态未知，请查询核实，不自动重放',
    'management_not_confirmed': '管理操作状态未知，请查询核实',
    'unavailable': '记忆存储暂不可用，无法确认结果', 'not_found': '未找到符合条件的记忆',
    'found': '已找到记录', 'memory_disabled': '当前入口未启用个人记忆',
    'delete': '已删除正文、历史和关联回执', 'clear': '已清空当前主人记忆',
    'pause': '已暂停自动候选，显式新请求仍可保存', 'resume': '已恢复自动候选，不回扫旧消息',
    'selected': '已选择项目上下文', 'alias_registered': '已关联简称并选择项目',
    'invalid_filter': '查询过滤条件无效', 'invalid_offset': '分页位置无效', 'invalid_action': '不支持的记忆管理动作',
    'missing_session': '缺少可信会话身份', 'paused': '自动记忆已暂停',
}


class MemoryResult(MemoryModel):
    status: Literal['saved', 'unchanged', 'needs_clarification', 'rejected', 'conflict', 'failed', 'unknown']
    reason: str
    summary: str
    memory_id: str | None = None
    revision: int | None = None


def applicable(record, *, session_id, project_id, now):
    """当前读取不自动复活被永久替换的历史。"""
    return (record['status'] == 'active' and record['kind'] == 'fact'
            and (record['scope'] != 'session' or record['session_id'] == session_id)
            and (record['project_id'] is None or record['project_id'] == project_id)
            and (record.get('valid_from') is None or datetime.fromisoformat(record['valid_from']) <= now)
            and (record.get('valid_until') is None or now < datetime.fromisoformat(record['valid_until'])))
