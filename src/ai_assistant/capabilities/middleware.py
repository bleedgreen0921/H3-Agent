"""每轮只规划一次；Skills 与工具绑定消费同一份 ExposurePlan。"""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from ai_assistant.capabilities.plan import core_plan, llm_plan
from ai_assistant.capabilities.selector import selection_text
from ai_assistant.runtime.execution import raise_if_fatal


class CapabilityPlannerMiddleware(AgentMiddleware):
    """在首次模型调用前调用 LLM 选择能力；失败时降级到核心工具。"""

    def __init__(self, registry):
        self.registry = registry

    async def prepare_plan(self, request):
        # 没有能力状态时不干预；计划已存在时复用，避免每次模型调用重复规划。
        context = getattr(getattr(request, 'runtime', None), 'context', None)
        state = getattr(context, 'capabilities', None)
        if state is None or state.plan is not None:
            return request
        if state.registry is None:
            state.registry = self.registry

        # 已激活的 Skills 和最近对话共同参与计划生成。
        records = request.state.get('active_skills', {}) if request.state else {}
        active = [key for key, value in records.items() if value is not None]
        hard_effect = 'read' if getattr(state, 'read_only', False) else None
        candidates = state.registry.eligible_tools(effect=hard_effect)
        skills = state.registry.eligible_skills(exclude=active)
        selector = (getattr(context, 'services', {}) or {}).get('capability_selector')
        directory_budget = getattr(context, 'skills_directory_tokens', 6000)
        settings = state.registry.settings
        max_tools = int(getattr(settings, 'top_k', 8) or 8)
        schema_budget = int(getattr(settings, 'max_schema_tokens', 0) or 0)

        if selector is None:
            state.plan = core_plan(
                state.registry, effect=hard_effect or 'read',
                reason='selector_unavailable', active_skill_ids=active)
        elif not candidates and not skills:
            state.plan = core_plan(
                state.registry, effect=hard_effect or 'read',
                reason='no_candidates', active_skill_ids=active)
        else:
            try:
                selection = await selector.select(
                    selection_text(request.messages),
                    tools=candidates, skills=skills, max_tools=max_tools,
                    schema_budget=schema_budget, directory_budget=directory_budget)
                state.plan = llm_plan(
                    state.registry, selection,
                    eligible_tools=candidates, eligible_skills=skills,
                    directory_budget=directory_budget,
                    active_skill_ids=active, effect=hard_effect,
                    reason='llm')
            except Exception as error:
                raise_if_fatal(error)
                state.plan = core_plan(
                    state.registry, effect=hard_effect or 'read',
                    reason=getattr(error, 'category', 'selector_error'),
                    active_skill_ids=active)

        # 记录初始计划及关联标识，供后续诊断和追踪使用。
        state.registry.record_plan(
            state.plan, input_id=getattr(context, 'input_id', None),
            session_id=getattr(context, 'session_id', None),
            change='initial',
            execution_thread_id=getattr(context, 'execution_thread_id', None),
            search_calls=state.search_calls,
        )
        return request

    # 规划需要调用异步模型，因此只实现异步入口；同步调用由基类明确报错。
    async def awrap_model_call(self, request, handler):
        return await handler(await self.prepare_plan(request))


class ToolBindingMiddleware(AgentMiddleware):
    """按照当前能力计划，为每次模型请求绑定可见工具。"""

    def _prepare(self, request, *, fallback):
        context = getattr(getattr(request, 'runtime', None), 'context', None)
        state = getattr(context, 'capabilities', None)
        if state is None or state.registry is None:
            return request

        # 预算估算阶段没有计划时保留全量工具作为保守上界；
        # 真正调用模型前若计划仍缺失，只绑定核心工具，绝不回退到全量。
        if state.plan is None:
            if not fallback:
                return request
            state.plan = core_plan(state.registry, effect='read', reason='plan_missing')

        # 只读状态或只读计划不向模型暴露登记为写入的工具。
        selected_ids = list(state.plan.tool_ids)
        if getattr(state, 'read_only', False) or getattr(state.plan, 'effect', None) == 'read':
            selected_ids = [name for name in selected_ids
                            if getattr(state.registry.descriptors.get(name),
                                       'side_effect', None) != 'write']
        if getattr(state.plan, 'forbid_create', False):
            selected_ids = [
                name for name in selected_ids
                if not getattr(state.registry.descriptors.get(name),
                               'creates', False)
            ]
        selected_ids = [name for name in selected_ids
                        if getattr(state.registry.descriptors.get(name),
                                   'available', True)]
        selected = state.registry.tools_for(selected_ids)

        # 提示模型可渐进搜索工具；重复模型调用时不重复追加提示。
        original = request.system_message
        text = original.text if original else ''
        if '当前工具按本轮任务按需暴露' not in text:
            text += ('\n当前工具按本轮任务按需暴露；现有工具不足时调用 search_tools 或 '
                     'search_skills 查询更多能力，命中后下一次模型请求即可使用。')
            original = SystemMessage(text, additional_kwargs=original.additional_kwargs if original else {})
        return request.override(tools=selected, system_message=original)

    def prepare(self, request):
        """供预算估算和已有计划绑定使用；缺少计划时保留全量上界。"""
        return self._prepare(request, fallback=False)

    # 同步与异步路径使用相同的纯绑定逻辑。
    def wrap_model_call(self, request, handler):
        return handler(self._prepare(request, fallback=True))

    async def awrap_model_call(self, request, handler):
        return await handler(self._prepare(request, fallback=True))
