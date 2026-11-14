"""每轮只规划一次；Skills 与工具绑定消费同一份 ExposurePlan。"""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage


def _last_user_text(messages):
    """从消息历史中倒序查找最近一条用户消息的文本。"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text if hasattr(message, 'text') else str(message.content)
    return ''


class CapabilityPlannerMiddleware(AgentMiddleware):
    """为当前轮次生成并记录工具与 Skills 共用的能力计划。"""

    def __init__(self, registry):
        self.registry = registry

    def prepare(self, request):
        # 没有能力状态时不干预；计划已存在时复用，避免每次模型调用重复规划。
        context = getattr(getattr(request, 'runtime', None), 'context', None)
        state = getattr(context, 'capabilities', None)
        if state is None or state.plan is not None:
            return request
        if state.registry is None:
            state.registry = self.registry

        # 已激活的 Skills 和最近一条用户消息共同参与计划生成。
        records = request.state.get('active_skills', {}) if request.state else {}
        active = [key for key, value in records.items() if value is not None]
        text = _last_user_text(request.messages)
        from ai_assistant.capabilities.intent import extract
        intent = extract(text)

        # 文档意图可携带当前服务的文档身份，供注册表选择相关能力。
        identity = None
        if intent.domains & {'docs'} or intent.objects & {'document'}:
            feishu = getattr(context, 'services', {}).get('feishu')
            identity = getattr(feishu, 'document_identity', None)
        state.plan = state.registry.plan(
            text,
            directory_budget=getattr(context, 'skills_directory_tokens', 6000),
            active_skill_ids=active,
            identity=identity,
            effect=intent.side_effect,
        )

        # 记录初始计划及关联标识，供后续诊断和追踪使用。
        state.registry.record_plan(
            state.plan, input_id=getattr(context, 'input_id', None),
            session_id=getattr(context, 'session_id', None),
            change='initial',
            execution_thread_id=getattr(context, 'execution_thread_id', None),
            search_calls=state.search_calls,
        )
        return request

    # 同步与异步调用均在调用模型前准备同一份计划。
    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))


class ToolBindingMiddleware(AgentMiddleware):
    """按照当前能力计划，为每次模型请求绑定可见工具。"""

    def prepare(self, request):
        context = getattr(getattr(request, 'runtime', None), 'context', None)
        state = getattr(context, 'capabilities', None)
        plan = getattr(state, 'plan', None)
        if plan is None or state.registry is None:
            return request

        # 只读状态或只读计划不向模型暴露登记为写入的工具。
        selected_ids = list(plan.tool_ids)
        if getattr(state, 'read_only', False) or getattr(plan, 'effect', None) == 'read':
            selected_ids = [name for name in selected_ids
                            if getattr(state.registry.descriptors.get(name), 'write', False) is False]

        # 禁止创建时，额外排除名称匹配的文档和知识空间节点创建工具。
        if getattr(plan, 'forbid_create', False):
            selected_ids = [
                name for name in selected_ids
                if not any(term in getattr(state.registry.descriptors.get(name),
                                           'original_name', '').casefold()
                           for term in ('document_create', 'doc_create', 'spacenode_create'))
            ]
        selected = state.registry.tools_for(selected_ids)

        # 提示模型可渐进搜索工具；重复模型调用时不重复追加提示。
        original = request.system_message
        text = original.text if original else ''
        if '当前工具按本轮任务按需暴露' not in text:
            text += ('\n当前工具按本轮任务按需暴露；现有工具不足时调用 search_tools 查询更多能力，'
                     '命中后下一次模型请求即可使用。')
            original = SystemMessage(text, additional_kwargs=original.additional_kwargs if original else {})
        return request.override(tools=selected, system_message=original)

    # 同步与异步路径使用相同的工具绑定逻辑。
    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))
