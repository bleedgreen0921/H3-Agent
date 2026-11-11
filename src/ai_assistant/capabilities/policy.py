"""统一能力读写分类；权限白名单仍由 integrations.policy.authorize 执行。"""
from ai_assistant.capabilities.intent import infer_descriptor


class CapabilityPolicy:
    def __init__(self, effects=None):
        self.effects = dict(effects or {})

    @classmethod
    def from_tools(cls, tools):
        effects = {}
        for tool in tools:
            actions, objects, domains, side_effect = infer_descriptor(
                tool.name, getattr(tool, 'description', ''), use_description=False)
            effects[tool.name] = side_effect if (actions or objects or domains) else 'unknown'
        return cls(effects)

    @classmethod
    def from_context(cls, context):
        state = getattr(context, 'capabilities', None)
        registry = getattr(state, 'registry', None)
        if registry is None:
            return cls()
        return cls({name: item.side_effect for name, item in registry.descriptors.items()
                    if item.kind == 'tool'})

    def effect(self, name):
        return self.effects.get(name, 'unknown')

    def requires_approval(self, name):
        return self.effect(name) == 'write'

    def can_try(self, caller, name, context=None):
        # 真正的白名单/身份/写开关仍由 integrations.policy 校验。
        return True
