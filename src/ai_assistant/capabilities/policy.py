"""统一能力读写分类；权限白名单仍由 integrations.policy.authorize 执行。"""
from ai_assistant.capabilities.intent import infer_descriptor


class CapabilityPolicy:
    """维护工具名称到读写效果的映射，不负责实际授权。"""

    def __init__(self, effects=None):
        # 复制传入的映射，避免外部修改原字典影响当前策略。
        self.effects = dict(effects or {})

    @classmethod
    def from_tools(cls, tools):
        """只根据工具名称推导效果；无法匹配任何类别时标为 unknown。"""
        effects = {}
        for tool in tools:
            # 禁用描述匹配，避免描述中的其他动作或否定表达污染分类。
            actions, objects, domains, side_effect = infer_descriptor(
                tool.name, getattr(tool, 'description', ''), use_description=False)
            effects[tool.name] = side_effect if (actions or objects or domains) else 'unknown'
        return cls(effects)

    @classmethod
    def from_context(cls, context):
        """从当前轮次的能力注册表读取工具效果。"""
        state = getattr(context, 'capabilities', None)
        registry = getattr(state, 'registry', None)

        # 没有注册表时返回空策略，未知工具的查询结果为 unknown。
        if registry is None:
            return cls()
        return cls({name: item.side_effect for name, item in registry.descriptors.items()
                    if item.kind == 'tool'})

    def effect(self, name):
        """查询工具效果；未登记时返回 unknown。"""
        return self.effects.get(name, 'unknown')

    def requires_approval(self, name):
        # 这里只返回写入标记，不发起审批或授予执行权限。
        return self.effect(name) == 'write'

    def can_try(self, caller, name, context=None):
        # 此方法当前不限制尝试；实际身份、白名单和写开关由授权层校验。
        return True
