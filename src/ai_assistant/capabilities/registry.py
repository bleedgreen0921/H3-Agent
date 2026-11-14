"""全量能力注册表：工具与 Skills 统一登记，权限已在装配前过滤。"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages.utils import count_tokens_approximately

from ai_assistant.capabilities.intent import infer_descriptor, query_terms
from ai_assistant.capabilities.model import CapabilityDescriptor, CapabilityPolicy

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """不可变能力目录；选择计划由 router 生成。"""

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
    def from_sources(cls, tools, *, mcp=None, skills=None, settings=None, log_path=None):
        """将已装配的工具、MCP 信息和 Skills 整理为统一能力目录。"""
        policy = CapabilityPolicy(
            max_schema_tokens=getattr(settings, 'max_schema_tokens', 25000),
            top_k=getattr(settings, 'top_k', 8),
            max_search_calls=getattr(settings, 'max_search_calls', 3),
        )
        entries = getattr(mcp, 'entries', {}) if mcp is not None else {}
        configured_core = tuple(getattr(settings, 'core', ()) or policy.core_tools)
        descriptors = {}
        tool_map = {}

        # 每个工具保留来源、身份、读写效果和 schema 成本。
        for tool in tools:
            name = tool.name
            entry = entries.get(name)
            original = getattr(entry, 'original_name', None) or name
            server = getattr(entry, 'server', None)
            identity = getattr(getattr(entry, 'settings', None), 'identity', None) if entry else None
            write_tools = getattr(getattr(entry, 'settings', None), 'write_tools', ()) if entry else ()
            write_hint = entry is not None and original in write_tools
            actions, objects, domains, side_effect = infer_descriptor(
                original, getattr(tool, 'description', ''), write_hint=write_hint)
            provider = f'mcp:{server}' if server else 'builtin'
            schema_tokens = count_tokens_approximately([], tools=[tool])
            descriptors[name] = CapabilityDescriptor(
                id=name, kind='tool', name=name, original_name=original,
                description=getattr(tool, 'description', ''), provider=provider,
                server=server, identity=identity,
                domains=domains, actions=actions, objects=objects,
                side_effect=side_effect, schema_tokens=schema_tokens,
                core=name in configured_core, write=side_effect == 'write', tool=tool)
            tool_map[name] = tool

        # Skill 使用同一描述结构，但预算按目录说明而非工具 schema 估算。
        skill_map = {}
        for skill in (getattr(skills, 'skills', {}) or {}).values():
            description = str(skill.metadata.get('description', ''))
            line = cls.skill_directory_line(skill)
            actions, objects, domains, side_effect = infer_descriptor(
                skill.metadata.get('name', skill.id), description)
            descriptors[skill.id] = CapabilityDescriptor(
                id=skill.id, kind='skill', name=skill.id,
                original_name=skill.metadata.get('name', skill.id),
                description=description, provider='local-skill',
                domains=domains, actions=actions, objects=objects,
                side_effect=side_effect,
                guidance_tokens=cls._tokens(line),
                core=False, tool=None, skill=skill)
            skill_map[skill.id] = skill

        # 核心工具未装配时仅告警，不因可选能力缺失中断注册。
        missing = [name for name in configured_core if name not in tool_map]
        if missing:
            logger.warning('Unknown core capability names: %s', ', '.join(missing))
        return cls(tool_map, descriptors, skill_map, settings, policy, log_path)

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
        """从已登记的工具效果构建运行时读写分类策略。"""
        from ai_assistant.capabilities.policy import CapabilityPolicy
        return CapabilityPolicy({name: item.side_effect for name, item in self.descriptors.items()
                                 if item.kind == 'tool'})

    def plan(self, query, *, directory_budget=6000, active_skill_ids=(),
             server=None, identity=None, effect=None):
        """生成初始计划，再应用资源身份、读写和 schema 预算约束。"""
        from ai_assistant.capabilities.router import build_plan
        plan = build_plan(self, query, directory_budget=directory_budget,
                          active_skill_ids=active_skill_ids, effect=effect)
        # 资源任务只暴露同一 MCP 服务/身份；内置工具不参与身份过滤。
        selected = []
        for name in plan.tool_ids:
            descriptor = self.descriptors.get(name)
            if descriptor is None:
                continue
            if descriptor.kind == 'tool' and descriptor.server:
                if server is not None and descriptor.server != server:
                    continue
                if identity is not None and descriptor.identity != identity:
                    continue
            if effect == 'read' and descriptor.write:
                continue
            selected.append(name)
        plan.server, plan.identity, plan.effect = server, identity, plan.effect
        configured = int(getattr(self.settings, 'max_schema_tokens', None) or 0)
        plan.max_schema_tokens = configured
        # 即使 router 发生 fallback，也不能让工具 schema 突破硬预算。
        if configured:
            bounded, used = [], 0
            for name in selected:
                cost = self.descriptors[name].schema_tokens if name in self.descriptors else 0
                if used + cost > configured:
                    continue
                bounded.append(name)
                used += cost
            selected = bounded
            if len(selected) != len(plan.tool_ids):
                plan.reason = 'budget_trimmed'
        plan.tool_ids = selected
        plan.tool_schema_tokens = self.schema_tokens(selected)
        return plan

    @staticmethod
    def _is_doc_descriptor(descriptor):
        """根据服务、标签和原始名称识别文档或文件相关能力。"""
        return bool(descriptor.server and (
            descriptor.domains & {'docs', 'files'}
            or descriptor.objects & {'document', 'file'}
            or any(word in descriptor.original_name.casefold()
                   for word in ('docx', 'wiki', 'drive'))
        ))

    @staticmethod
    def _creates_document(descriptor):
        """根据原始工具名识别受创建限制的能力。"""
        original = descriptor.original_name.casefold()
        return any(term in original for term in
                   ('document_create', 'doc_create', 'spacenode_create'))

    def seed_resource_tools(self, plan, *, server=None, identity=None,
                            effect=None, forbid_create=False, append=False):
        """资源任务的必备工具由资源解析确定，不依赖查询检索是否命中。"""
        # 先筛选目标服务和身份下符合读写、创建限制的文档工具。
        candidates = []
        for descriptor in self.descriptors.values():
            if descriptor.kind != 'tool' or not self._is_doc_descriptor(descriptor):
                continue
            if server is not None and descriptor.server != server:
                continue
            if identity is not None and descriptor.identity != identity:
                continue
            if forbid_create and self._creates_document(descriptor):
                continue
            if effect == 'read' and descriptor.write:
                continue
            candidates.append(descriptor)

        # 追加任务优先使用高层语义工具；没有时回退到原生 block 工具。
        def priority(item):
            original = item.original_name.casefold()
            if effect == 'write' and item.write:
                if any(term in original for term in
                       ('append_markdown', 'append_blocks')):
                    return 0
                if any(term in original for term in
                       ('blockchildren_create', 'blockchild', 'block_create', 'append')):
                    return 1
                return 2
            return 3

        candidates.sort(key=lambda item: (priority(item), item.schema_tokens, item.id))
        maximum = int(plan.max_schema_tokens or
                      getattr(self.settings, 'max_schema_tokens', 0) or 0)
        readers = [item for item in candidates if not item.write]
        if effect == 'write':
            semantic = [item for item in candidates if priority(item) == 0]
            native_append = [item for item in candidates if priority(item) == 1]
            other = [item for item in candidates if priority(item) == 2]
            if append and semantic:
                action = semantic
            elif append and native_append:
                action = native_append
            else:
                action = other or semantic or native_append
            required = action + readers
        else:
            action = []
            required = readers
        action_ids = {item.id for item in action}
        if effect == 'write' and not action_ids:
            raise ValueError('目标文档没有可用的写入工具。')

        # 动作工具是必达能力；readers 可以在预算内补充，但不能把必达工具
        # 以外的 schema 挤进模型上下文。动作工具本身超预算时明确失败。
        selected, used = [], 0
        for item in required:
            cost = item.schema_tokens
            if maximum and used + cost > maximum:
                if item.id in action_ids:
                    raise ValueError(
                        '目标文档写入工具超出本轮 schema 预算，请缩小范围或启用语义写入工具。')
                continue
            if item.id not in selected:
                selected.append(item.id)
                used += cost
        semantic_append = bool(action) and any(
            'append_markdown' in item.original_name.casefold() for item in action)

        # 语义追加已覆盖块构造，不再把原生 block 工具作为可选项塞回上下文。
        existing = [item.id for item in candidates if item.id not in selected]
        existing += [name for name in plan.tool_ids if name not in selected]
        for name in existing:
            descriptor = self.descriptors.get(name)
            if descriptor is None:
                continue
            if semantic_append and any(
                    term in descriptor.original_name.casefold()
                    for term in ('blockchildren_create', 'blockchild', 'block_create')):
                continue
            if effect == 'write' and descriptor.write and descriptor.id not in action_ids:
                # 写任务只保留已解析的动作工具；patch/move 等不属于当前意图的
                # 写能力不进入上下文，避免模型在追加任务里改文档或移动节点。
                continue
            if forbid_create and self._creates_document(descriptor):
                continue
            if effect == 'read' and descriptor.write:
                continue
            cost = descriptor.schema_tokens
            if maximum and used + cost > maximum:
                continue
            selected.append(name)
            used += cost
        plan.tool_ids = list(dict.fromkeys(selected))
        plan.tool_schema_tokens = self.schema_tokens(plan.tool_ids)
        plan.reason = 'resource_seeded'
        return plan

    def extend_plan(self, plan, query, *, limit=8, effect=None):
        """按同一身份/效果/预算扩展 ExposurePlan；不突破 schema 上限。"""
        from ai_assistant.capabilities.intent import query_terms
        terms = query_terms(query)
        effective = effect or plan.effect
        maximum = int(plan.max_schema_tokens or getattr(self.settings, 'max_schema_tokens', 0) or 0)
        known = set(plan.tool_ids)
        candidates = []

        # 排除已绑定或不符合当前计划约束的工具，再对相关候选评分。
        for name, descriptor in self.descriptors.items():
            if (descriptor.kind != 'tool' or descriptor.core
                    or name == 'search_tools' or name in known):
                continue
            if effective == 'read' and descriptor.write:
                continue
            if plan.forbid_create:
                original = descriptor.original_name.casefold()
                if any(term in original for term in
                       ('document_create', 'doc_create', 'spacenode_create')):
                    continue
            if descriptor.server:
                if plan.server is not None and descriptor.server != plan.server:
                    continue
                if plan.identity is not None and descriptor.identity != plan.identity:
                    continue
            score = self.score(descriptor, terms)
            if score > 0:
                candidates.append((score, descriptor))
        candidates.sort(key=lambda item: (-item[0], item[1].id))

        # 按分数加入预算允许的工具，并报告因预算跳过的数量。
        added, skipped = [], 0
        used = plan.tool_schema_tokens
        for _, descriptor in candidates:
            if len(added) >= max(1, min(limit, 30)):
                break
            if maximum and used + descriptor.schema_tokens > maximum:
                skipped += 1
                continue
            plan.tool_ids.append(descriptor.id)
            added.append(descriptor)
            used += descriptor.schema_tokens
        plan.tool_schema_tokens = used
        return added, skipped, bool(skipped and not added)

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

    def score(self, descriptor, terms, intent=None):
        """按名称、描述和可选意图标签计算工具相关性分数。"""
        if descriptor.kind != 'tool':
            return 0
        name = descriptor.name.casefold()
        original = descriptor.original_name.casefold()
        text = (name + ' ' + original + ' ' + descriptor.description).casefold()
        # 名称命中权重高于描述；意图标签再按领域、动作、对象加权。
        score = sum(3 if term in name else 2 if term in original else 1
                    for term in terms if term in text)
        if intent is not None:
            score += len(descriptor.domains & intent.domains) * 4
            score += len(descriptor.actions & intent.actions) * 3
            score += len(descriptor.objects & intent.objects) * 2
        if descriptor.side_effect == 'write':
            score -= 1
        return score

    def search_tools(self, query, limit=8):
        """返回匹配的非核心工具摘要，不直接修改能力计划。"""
        terms = query_terms(query)
        items = []
        for name, descriptor in self.descriptors.items():
            if descriptor.kind != 'tool' or descriptor.core or name == 'search_tools':
                continue
            score = self.score(descriptor, terms)
            if score > 0:
                items.append((score, name, descriptor))
        items.sort(key=lambda item: (-item[0], item[1]))
        return [item[2].summary() for item in items[:max(1, min(limit, 30))]]

    def record_plan(self, plan, input_id=None, session_id=None, *,
                    change='initial', execution_thread_id=None, search_calls=0):
        """记录一次能力计划快照；初始计划、扩展和 Skill 变更共用同一结构。"""
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
              search_calls=search_calls, reason=plan.reason)
        if self.log_path is None:
            return

        # 可选 JSONL 日志记录计划摘要；写入失败不阻断当前业务。
        record = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'input_id': input_id,
            'session_id': session_id,
            'execution_thread_id': execution_thread_id,
            'mode': getattr(self.settings, 'mode', 'retrieval'),
            'reason': plan.reason,
            'change': change,
            'plan_hash': plan_hash,
            'search_calls': search_calls,
            'intent': plan.intent.summary() if plan.intent else None,
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
