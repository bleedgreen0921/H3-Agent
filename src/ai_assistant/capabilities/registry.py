"""全量能力注册表：工具与 Skills 统一登记，权限已在装配前过滤。"""
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
        policy = CapabilityPolicy(
            max_schema_tokens=getattr(settings, 'max_schema_tokens', 25000),
            top_k=getattr(settings, 'top_k', 8),
            max_search_calls=getattr(settings, 'max_search_calls', 3),
        )
        entries = getattr(mcp, 'entries', {}) if mcp is not None else {}
        configured_core = tuple(getattr(settings, 'core', ()) or policy.core_tools)
        descriptors = {}
        tool_map = {}
        for tool in tools:
            name = tool.name
            entry = entries.get(name)
            original = getattr(entry, 'original_name', None) or name
            server = getattr(entry, 'server', None)
            write_tools = getattr(getattr(entry, 'settings', None), 'write_tools', ()) if entry else ()
            write_hint = entry is not None and original in write_tools
            actions, objects, domains, side_effect = infer_descriptor(
                original, getattr(tool, 'description', ''), write_hint=write_hint)
            provider = f'mcp:{server}' if server else 'builtin'
            schema_tokens = count_tokens_approximately([], tools=[tool])
            descriptors[name] = CapabilityDescriptor(
                id=name, kind='tool', name=name, original_name=original,
                description=getattr(tool, 'description', ''), provider=provider,
                domains=domains, actions=actions, objects=objects,
                side_effect=side_effect, schema_tokens=schema_tokens,
                core=name in configured_core, write=side_effect == 'write', tool=tool)
            tool_map[name] = tool

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

        missing = [name for name in configured_core if name not in tool_map]
        if missing:
            logger.warning('Unknown core capability names: %s', ', '.join(missing))
        return cls(tool_map, descriptors, skill_map, settings, policy, log_path)

    @staticmethod
    def skill_directory_line(skill):
        name = skill.metadata.get('name', skill.id)
        return f"{skill.id} | {name}: {skill.metadata.get('description', '')}"

    @staticmethod
    def _tokens(text):
        from langchain_core.messages import SystemMessage
        return count_tokens_approximately([SystemMessage(content=text)])

    def build_policy(self):
        from ai_assistant.capabilities.policy import CapabilityPolicy
        return CapabilityPolicy({name: item.side_effect for name, item in self.descriptors.items()
                                 if item.kind == 'tool'})

    def plan(self, query, *, directory_budget=6000, active_skill_ids=()):
        from ai_assistant.capabilities.router import build_plan
        return build_plan(self, query, directory_budget=directory_budget,
                          active_skill_ids=active_skill_ids)

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
        if descriptor.kind != 'tool':
            return 0
        name = descriptor.name.casefold()
        original = descriptor.original_name.casefold()
        text = (name + ' ' + original + ' ' + descriptor.description).casefold()
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

    def record_plan(self, plan, input_id=None, session_id=None):
        if self.log_path is None:
            return
        record = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'input_id': input_id,
            'session_id': session_id,
            'mode': getattr(self.settings, 'mode', 'retrieval'),
            'reason': plan.reason,
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
