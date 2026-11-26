"""全量能力目录：显式元数据、核心集合和资源工具种子。"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages.utils import count_tokens_approximately

from ai_assistant.capabilities.model import CapabilityDescriptor
from ai_assistant.capabilities.policy import CapabilityPolicy

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """不可变能力目录；选择由 LLM 完成，程序只维护边界和预算。"""

    def __init__(self, tools, descriptors, skills, settings, policy, log_path=None):
        # 保存输入映射的副本，并预先整理核心工具和 Skill 的 ID。
        self.tools = dict(tools)
        self.descriptors = dict(descriptors)
        self.skills = dict(skills)
        self.settings = settings
        self.policy = policy
        self.log_path = Path(log_path) if log_path else None
        self.core_ids = tuple(name for name, item in self.descriptors.items()
                              if item.core and item.kind == 'tool')
        self.skill_ids = tuple(name for name, item in self.descriptors.items()
                               if item.kind == 'skill')

    @classmethod
    def from_sources(cls, tools, *, mcp=None, skills=None, metadata=None,
                     settings=None, log_path=None):
        """将工具、MCP 信息和 Skills 整理为显式能力目录。"""
        maximum = int(getattr(settings, 'max_schema_tokens', 25000) or 0)
        top_k = int(getattr(settings, 'top_k', 8) or 8)
        max_search = int(getattr(settings, 'max_search_calls', 3) or 0)
        policy = CapabilityPolicy(max_schema_tokens=maximum, top_k=top_k,
                                  max_search_calls=max_search)
        entries = getattr(mcp, 'entries', {}) if mcp is not None else {}
        configured_core = tuple(getattr(settings, 'core', ()) or policy.core_tools)
        declared = dict(metadata or {})
        descriptors = {}
        tool_map = {}

        # 每个工具保留来源、身份、读写效果和 schema 成本。
        for tool in tools:
            name = tool.name
            entry = entries.get(name)
            if entry is not None:
                service = getattr(entry, 'settings', None)
                original = getattr(entry, 'original_name', None) or name
                server = getattr(entry, 'server', None)
                identity = getattr(service, 'identity', None)
                write_tools = set(getattr(service, 'write_tools', ()) or ())
                effect = 'write' if original in write_tools else 'read'
                available = not (effect == 'write'
                                 and not bool(getattr(service, 'allow_write', False)))
                resource_kinds, append_capable, creates = frozenset(), False, False
                provider = f'mcp:{server}' if server else 'builtin'
            else:
                original = name
                server = identity = None
                meta = declared.get(name)
                if meta is None:
                    logger.warning('Missing capability metadata for tool: %s', name)
                    effect, resource_kinds = 'unknown', frozenset()
                    append_capable = creates = False
                    available = False
                else:
                    effect = meta.side_effect
                    resource_kinds = meta.resource_kinds
                    append_capable = meta.append_capable
                    creates = meta.creates
                    available = meta.available
                provider = 'builtin'

            descriptors[name] = CapabilityDescriptor(
                id=name, kind='tool', name=name, original_name=original,
                description=getattr(tool, 'description', ''), provider=provider,
                server=server, identity=identity,
                side_effect=effect, resource_kinds=resource_kinds,
                append_capable=append_capable, creates=creates,
                available=available,
                schema_tokens=count_tokens_approximately([], tools=[tool]),
                core=name in configured_core, tool=tool)
            tool_map[name] = tool

        # Skill 使用同一描述结构，但预算按目录说明而非工具 schema 估算。
        skill_map = {}
        for skill in (getattr(skills, 'skills', {}) or {}).values():
            description = str(skill.metadata.get('description', ''))
            line = cls.skill_directory_line(skill)
            descriptors[skill.id] = CapabilityDescriptor(
                id=skill.id, kind='skill', name=skill.id,
                original_name=skill.metadata.get('name', skill.id),
                description=description, provider='local-skill',
                side_effect=None, available=True,
                guidance_tokens=cls._tokens(line), core=False, tool=None,
                skill=skill)
            skill_map[skill.id] = skill

        # 核心工具未装配时仅告警，不因可选能力缺失中断注册。
        missing = [name for name in configured_core if name not in tool_map]
        if missing:
            logger.warning('Unknown core capability names: %s', ', '.join(missing))
        return cls(tool_map, descriptors, skill_map, settings, policy, log_path)

    def validate(self):
        """生产装配时校验显式元数据、核心工具和核心 schema 预算。"""
        from ai_assistant.config import ConfigError
        unknown = sorted(name for name, item in self.descriptors.items()
                         if item.kind == 'tool' and item.side_effect not in {'read', 'write'})
        if unknown:
            raise ConfigError(
                '工具缺少显式能力元数据：' + '、'.join(unknown)
                + '；请在组件或 MCP 配置中声明 side_effect。')
        bad_core = sorted(name for name in self.core_ids
                          if not self.descriptors[name].available
                          or self.descriptors[name].side_effect != 'read')
        if bad_core:
            raise ConfigError('核心工具必须显式声明为只读且可用：' + '、'.join(bad_core))
        maximum = int(getattr(self.settings, 'max_schema_tokens', 0) or 0)
        if maximum and self.schema_tokens(self.core_ids) > maximum:
            raise ConfigError('核心工具 schema 超出 max_schema_tokens；请提高预算或缩小核心集合。')
        return self

    @staticmethod
    def skill_directory_line(skill):
        """生成展示给模型的单行 Skill 目录信息。"""
        name = skill.metadata.get('name', skill.id)
        return f"{skill.id} | {name}: {skill.metadata.get('description', '')}"

    @staticmethod
    def _tokens(text):
        """按消息格式近似估算文本 token 数。"""
        from langchain_core.messages import SystemMessage
        return count_tokens_approximately([SystemMessage(content=text)])

    def build_policy(self):
        """从已登记的显式效果构建运行时读写分类策略。"""
        return CapabilityPolicy(
            {name: item.side_effect for name, item in self.descriptors.items()
             if item.kind == 'tool'},
            max_schema_tokens=getattr(self.settings, 'max_schema_tokens', 25000) or 0,
            top_k=getattr(self.settings, 'top_k', 8) or 8,
            max_search_calls=getattr(self.settings, 'max_search_calls', 3) or 0)

    def eligible_tools(self, *, effect=None, server=None, identity=None,
                       forbid_create=False, exclude=()):
        """按硬约束构造 LLM 候选；不在此处做语义排序。"""
        excluded = set(self.core_ids) | set(exclude)
        maximum = int(getattr(self.settings, 'max_schema_tokens', 0) or 0)
        items = []
        for name, item in self.descriptors.items():
            if item.kind != 'tool' or name in excluded:
                continue
            if not item.available or item.side_effect not in {'read', 'write'}:
                continue
            if effect == 'read' and item.side_effect == 'write':
                continue
            if forbid_create and item.creates:
                continue
            if item.server is not None:
                if server is not None and item.server != server:
                    continue
                if identity is not None and item.identity != identity:
                    continue
            if maximum and item.schema_tokens > maximum:
                continue
            items.append(item)
        return sorted(items, key=lambda item: item.id)

    def eligible_skills(self, *, exclude=()):
        """返回未激活、未在目录中的 Skill 候选。"""
        excluded = set(exclude)
        return sorted((item for name, item in self.descriptors.items()
                       if item.kind == 'skill' and name not in excluded
                       and item.available), key=lambda item: item.id)

    def seed_resource_tools(self, plan, *, resource_kinds, server=None,
                            identity=None, effect=None, forbid_create=False,
                            append=False):
        """资源任务的必备工具由显式 resource_kinds 确定。"""
        kinds = frozenset(resource_kinds or ())
        if not kinds:
            return plan
        candidates = []
        for item in self.descriptors.values():
            if item.kind != 'tool' or not item.available:
                continue
            if not (item.resource_kinds & kinds):
                continue
            if server is not None and item.server is not None and item.server != server:
                continue
            if identity is not None and item.identity is not None and item.identity != identity:
                continue
            if effect == 'read' and item.write:
                continue
            if forbid_create and item.creates:
                continue
            candidates.append(item)

        readers = sorted((item for item in candidates if not item.write),
                         key=lambda item: (item.schema_tokens, item.id))
        writes = [item for item in candidates if item.write]
        action = []
        if effect == 'write':
            if append:
                action = sorted((item for item in writes if item.append_capable),
                                key=lambda item: (item.schema_tokens, item.id))[:1]
                if not action:
                    raise ValueError(
                        '目标资源没有显式声明 append_capable 的写入工具；'
                        '请检查资源工具元数据。')
            else:
                # 非追加写任务优先非追加工具；只暴露一个动作工具，避免扩大写入面。
                ordered = sorted(writes, key=lambda item: (
                    item.append_capable, item.creates, item.schema_tokens, item.id))
                action = ordered[:1]
                if not action:
                    raise ValueError(
                        '目标资源没有显式声明的写入工具；'
                        '请检查资源工具元数据。')
        required = action + readers

        # 动作工具是必达能力；readers 只能在预算内补充。
        selected = list(plan.tool_ids)
        used = self.schema_tokens(selected)
        maximum = int(plan.max_schema_tokens
                      or getattr(self.settings, 'max_schema_tokens', 0) or 0)
        action_ids = {item.id for item in action}
        for item in required:
            if item.id in selected:
                continue
            if maximum and used + item.schema_tokens > maximum:
                if item.id in action_ids:
                    raise ValueError(
                        '目标资源写入工具超出本轮 schema 预算，请缩小范围或提高预算。')
                continue
            selected.append(item.id)
            used += item.schema_tokens

        plan.tool_ids = list(dict.fromkeys(selected))
        plan.tool_schema_tokens = self.schema_tokens(plan.tool_ids)
        plan.reason = 'resource_seeded'
        plan.effect = effect or plan.effect
        plan.requires_write = plan.effect == 'write'
        plan.server = server
        plan.identity = identity
        plan.forbid_create = forbid_create
        return plan

    def seed_create_tools(self, plan, *, resource_kinds, server=None, identity=None):
        """创建任务的工具由显式 creates 元数据确定。"""
        kinds = frozenset(resource_kinds or ())
        if not kinds:
            raise ValueError('创建任务缺少显式资源类型，无法选择创建工具。')
        candidates = [item for item in self.descriptors.values()
                      if item.kind == 'tool' and item.available and item.write
                      and item.creates and item.resource_kinds & kinds]
        if server is not None:
            candidates = [item for item in candidates
                          if item.server is None or item.server == server]
        if identity is not None:
            candidates = [item for item in candidates
                          if item.identity is None or item.identity == identity]
        candidates.sort(key=lambda item: (item.schema_tokens, item.id))
        if not candidates:
            raise ValueError(
                '目标资源没有显式声明的创建工具；'
                '请检查资源工具元数据。')
        selected = list(plan.tool_ids)
        used = self.schema_tokens(selected)
        maximum = int(plan.max_schema_tokens
                      or getattr(self.settings, 'max_schema_tokens', 0) or 0)
        for item in candidates:
            if maximum and used + item.schema_tokens > maximum:
                continue
            selected.append(item.id)
            used += item.schema_tokens
            break
        else:
            raise ValueError('目标资源创建工具超出本轮 schema 预算。')
        plan.tool_ids = list(dict.fromkeys(selected))
        plan.tool_schema_tokens = self.schema_tokens(plan.tool_ids)
        plan.reason = 'create_seeded'
        plan.effect = 'write'
        plan.requires_write = True
        plan.server = server
        plan.identity = identity
        return plan

    # 以下方法按 ID 取回能力对象、目录文本和预算统计。
    def all_tool_ids(self):
        return [name for name, item in self.descriptors.items() if item.kind == 'tool']

    def tools_for(self, ids):
        return [self.tools[name] for name in ids if name in self.tools]

    def descriptors_for(self, ids):
        return [self.descriptors[name] for name in ids if name in self.descriptors]

    def schema_tokens(self, ids):
        return sum(self.descriptors[name].schema_tokens for name in ids
                   if name in self.descriptors)

    def skill_directory_tokens(self, ids):
        return sum(self.descriptors[name].guidance_tokens for name in ids
                   if name in self.descriptors)

    def skill_lines(self, ids):
        return [self.skill_directory_line(self.skills[name]) for name in ids
                if name in self.skills]

    def record_plan(self, plan, input_id=None, session_id=None, *,
                    change='initial', execution_thread_id=None, search_calls=0):
        """记录一次能力计划快照；不记录用户正文或工具描述。"""
        plan_hash = self._plan_hash(plan)
        from ai_assistant.observability import csv_ids, event
        # 先写 trace：即使本地选择日志不可用，能力选择仍可在观测侧看到。
        event('capability.plan', input_id=input_id, session_id=session_id,
              execution_thread_id=execution_thread_id, plan_hash=plan_hash, change=change,
              tool_count=len(self.all_tool_ids()), selected_count=len(plan.tool_ids),
              schema_tokens=plan.tool_schema_tokens,
              skill_count=len(self.skill_ids), selected_skill_count=len(plan.skill_directory_ids),
              skill_directory_tokens=plan.skill_directory_tokens,
              active_skill_count=len(plan.active_skill_ids),
              selected_tools=csv_ids(plan.tool_ids),
              selected_skills=csv_ids(plan.skill_directory_ids),
              search_calls=search_calls, reason=plan.reason,
              selection_source=plan.selection_source,
              candidate_tool_count=plan.candidate_tool_count,
              candidate_skill_count=plan.candidate_skill_count,
              requires_write=plan.requires_write, effect=plan.effect)
        if self.log_path is None:
            return

        # 可选 JSONL 日志记录计划摘要；写入失败不阻断当前业务。
        record = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'input_id': input_id,
            'session_id': session_id,
            'execution_thread_id': execution_thread_id,
            'reason': plan.reason,
            'change': change,
            'plan_hash': plan_hash,
            'search_calls': search_calls,
            'selection_source': plan.selection_source,
            'candidate_tool_count': plan.candidate_tool_count,
            'candidate_skill_count': plan.candidate_skill_count,
            'requires_write': plan.requires_write,
            'effect': plan.effect,
            'tools': list(plan.tool_ids),
            'tool_schema_tokens': plan.tool_schema_tokens,
            'skill_directory': list(plan.skill_directory_ids),
            'active_skills': list(plan.active_skill_ids),
            'skill_directory_tokens': plan.skill_directory_tokens,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        except OSError:
            logger.warning('Capability selection log unavailable: %s', self.log_path)

    @staticmethod
    def _plan_hash(plan):
        """对排序后的能力 ID 计算短哈希，便于关联同一计划快照。"""
        payload = {'tools': sorted(plan.tool_ids),
                   'skills': sorted(plan.skill_directory_ids),
                   'active': sorted(plan.active_skill_ids)}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]
