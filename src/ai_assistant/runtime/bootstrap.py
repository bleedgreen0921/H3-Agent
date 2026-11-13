"""按配置和调用者权限组装资源，并统一管理会话退出时的清理。"""
import asyncio
import logging
from pathlib import Path
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import replace

from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent, load_persona
from ai_assistant.config import ConfigError
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.requests import model_session
from ai_assistant.tools.builtin import create_builtin_tools
from ai_assistant.tools.context import tool_context
from ai_assistant.cancellation import _drain_worker
from ai_assistant.storage.sessions import session_storage
from ai_assistant.runtime.session import AssistantSession
from ai_assistant.observability import current_service, ObservationService

logger = logging.getLogger(__name__)


def validate_context_config(config, selected):
    if config.context is None:
        return
    if config.models[selected].max_input_tokens is None:
        raise ConfigError('启用上下文管理的模型必须显式配置 max_input_tokens。')


@asynccontextmanager
async def assistant_session(config, model_name=None, *, retriever=None, session_id=None, caller=None, resources=None):
    from ai_assistant.integrations.policy import Caller
    caller = caller or Caller()
    persona = load_persona(config) if caller.personal else ''
    # 本地及主人私聊是仅有入口，不再为群聊构造另一套能力集合。
    caller.__post_init__()
    if not caller.personal:
        raise ConfigError('仅支持本地入口和主人私聊。')
    selected = model_name or config.default_model
    if selected not in config.models:
        raise ConfigError('Unknown model profile.')
    if session_id is not None and config.sessions is None:
        raise ConfigError('Opening a session requires [sessions].')
    validate_context_config(config, selected)
    if config.mcp and any(s.tools for s in config.mcp.servers.values()):
        if config.models[selected].max_input_tokens is None:
            raise ConfigError('启用 MCP 工具的模型必须显式配置 max_input_tokens。')
    async with AsyncExitStack() as stack:
        if current_service() is None:
            from ai_assistant.observability import configuration_secrets
            await stack.enter_async_context(ObservationService(config.observability, configuration_secrets(config)))
        repository = await stack.enter_async_context(session_storage(config.sessions)) if config.sessions else None
        usage_store = None
        if repository:
            from ai_assistant.storage.usage import usage_storage
            try:
                usage_store = await stack.enter_async_context(usage_storage(config.sessions))
            except Exception as error:
                logger.warning('Usage unavailable: phase=connect category=%s', type(error).__name__)
        saver = repository.saver if repository else InMemorySaver()
        context = await stack.enter_async_context(tool_context(config, retriever=retriever,
            skill_snapshot=getattr(resources, "skills", None) if caller.personal else None))
        from ai_assistant.tools.context import FeishuService
        feishu = (FeishuService(config.integrations.feishu.owner_open_id,
                                config.integrations.feishu.default_task_assignee,
                                config.integrations.feishu.task_members,
                                config.integrations.feishu.default_calendar_id)
                  if caller.personal and config.integrations.feishu else FeishuService())
        context = replace(context, usage_store=usage_store,
                          services={**context.services, 'feishu': feishu},
                          timezone=config.agent.timezone if caller.personal else 'UTC',
                          caller=caller,
                          skills_directory_tokens=config.skills.directory_tokens if config.skills else 6000)
        from ai_assistant.storage.integrations import operation_storage
        operation_store = (await stack.enter_async_context(operation_storage(config.sessions))
                           if config.sessions else None)
        # ToolContext 与组件只依赖操作回执仓储；inbox/outbox 仍由渠道使用完整 store。
        operations = operation_store.operations if operation_store is not None else None
        context = replace(context, operations=operations)
        model = await stack.enter_async_context(model_session(config, selected,
            config.models[selected].parameters.get('timeout', 180), factory=create_model))

        from ai_assistant.runtime.components import ComponentContext, build_components
        component_context = ComponentContext(
            config=config, caller=caller, stack=stack, resources=resources,
            operations=operations, context=context,
            max_input_tokens=config.models[selected].max_input_tokens)
        tools, prompt_sections, component_middleware, failure_notices, services = [], [], [], [], {}
        for component in build_components(config, caller, retriever=retriever):
            await component.open(component_context)
            contribution = component.contribute(component_context)
            tools.extend(contribution.tools)
            prompt_sections.extend(contribution.prompt_sections)
            component_middleware.extend(contribution.middleware)
            failure_notices.extend(contribution.failure_notices)
            services.update(contribution.services)
        context = component_context.context
        if services:
            context = replace(context, services={**context.services, **services})

        from ai_assistant.capabilities.model import CapabilityTurnState, resolve_log_path
        from ai_assistant.capabilities.registry import CapabilityRegistry
        base = config.config_path.parent if config.config_path else Path.cwd()
        log_path = resolve_log_path(config.tools.selection.log_path, base)
        skills = context.services.get('skills')
        skill_ids = tuple(sorted(getattr(skills, 'skills', {}) or {}))
        key = (caller.binding, tuple(tool.name for tool in tools), skill_ids,
               repr(config.tools.selection), str(log_path))
        cache = getattr(resources, 'capability_cache', None) if resources is not None else None
        registry = cache.get(key) if cache is not None else None
        if registry is None:
            registry = CapabilityRegistry.from_sources(
                tools, mcp=context.services.get('mcp'), skills=skills,
                settings=config.tools.selection, log_path=log_path)
            if cache is not None:
                cache[key] = registry
        context = replace(context, capabilities=CapabilityTurnState(registry=registry))
        agent = create_assistant_agent(model, tools, checkpointer=saver, settings=config.agent, persona=persona,
            context_settings=config.context,
            max_input_tokens=config.models[selected].max_input_tokens,
            model_parameters=config.models[selected].parameters,
            prompt_sections=prompt_sections, component_middleware=component_middleware,
            failure_notices=failure_notices, capability_registry=registry,
            capability_policy=registry.build_policy())
        session = AssistantSession(agent, context, config.agent, model_name=selected,
                                   repository=repository)
        await session.open(session_id)
        watcher = asyncio.create_task(session.watch_connection()) if repository else None
        try:
            yield session
        finally:
            if session._active_task is not None:
                task = session._active_task
                if not task.cancelling():
                    task.cancel()
                await _drain_worker(task)
            session._closed = True
            if watcher:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
            if repository is None:
                saver.storage.clear()
                saver.writes.clear()
                saver.blobs.clear()
