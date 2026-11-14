"""运行时功能组件：资源、工具、提示和中间件的可复用装配单元。"""
from dataclasses import dataclass, field
from dataclasses import replace

from ai_assistant.capabilities.search import search_tools
from ai_assistant.runtime.prompt import PromptSection
from ai_assistant.runtime.prompts import (
    attachment_section, core_sections, feishu_docs_section, feishu_tasks_section, knowledge_section, skills_section,
)


@dataclass
class ComponentContext:
    """组件装配时共享的配置、调用者、资源栈和工具上下文。"""
    config: object
    caller: object
    stack: object
    resources: object | None
    operations: object | None
    context: object
    services: dict = field(default_factory=dict)
    max_input_tokens: int | None = None

    def update_context(self, **values):
        self.context = replace(self.context, **values)
        return self.context


@dataclass
class Contribution:
    """组件向 Agent 提供的工具、提示、中间件和附加服务。"""
    tools: list = field(default_factory=list)
    prompt_sections: list[PromptSection] = field(default_factory=list)
    middleware: list = field(default_factory=list)
    services: dict = field(default_factory=dict)
    failure_notices: list = field(default_factory=list)


class Component:
    """组件约定：判断启用、打开资源并贡献能力。"""
    name = 'component'

    def enabled(self, config, caller, retriever=None):
        return True

    async def open(self, ctx: ComponentContext):
        return None

    def contribute(self, ctx: ComponentContext) -> Contribution:
        return Contribution()


class CoreComponent(Component):
    """贡献基础文件、可选联网、时间解析和能力搜索工具。"""
    name = 'core'

    def contribute(self, ctx):
        from ai_assistant.runtime.bootstrap import create_builtin_tools
        from ai_assistant.tools.datetime import resolve_datetime
        tools = create_builtin_tools(include_web=ctx.config.tools.web is not None,
            include_write=ctx.config.tools.files.enabled)
        tools.append(resolve_datetime.model_copy(update={'handle_tool_error': True}))
        tools.append(search_tools)
        return Contribution(tools=tools, prompt_sections=core_sections())


class SkillsComponent(Component):
    """在配置 Skills 时贡献目录工具和指导中间件。"""
    name = 'skills'

    def enabled(self, config, caller, retriever=None):
        return config.skills is not None

    def contribute(self, ctx):
        catalog = ctx.services.get('skills') or ctx.context.services.get('skills')
        if ctx.config.skills is None or catalog is None:
            return Contribution()
        from ai_assistant.agents.skills import SkillsMiddleware
        from ai_assistant.tools.skills import create_skill_tools
        return Contribution(tools=create_skill_tools(),
            prompt_sections=[skills_section()],
            middleware=[SkillsMiddleware(ctx.max_input_tokens)],
            services={'skills': catalog})


class MCPComponent(Component):
    """复用或打开 MCP 目录，并按调用者筛选工具。"""
    name = 'mcp'

    def enabled(self, config, caller, retriever=None):
        return bool(config.mcp and config.mcp.servers)

    async def open(self, ctx):
        from ai_assistant.integrations.mcp import mcp_resources
        directory = getattr(ctx.resources, 'catalog', None) if ctx.resources is not None else None
        if directory is None:
            directory = await ctx.stack.enter_async_context(
                mcp_resources(ctx.config, ctx.context.workspace, caller=ctx.caller))
        ctx.services['mcp'] = directory.for_caller(ctx.caller)

    def contribute(self, ctx):
        from ai_assistant.agents.mcp import MCPMiddleware
        catalog = ctx.services.get('mcp')
        return Contribution(tools=[entry.tool for entry in catalog.entries.values()] if catalog else [],
                            middleware=[MCPMiddleware()], services={'mcp': catalog} if catalog else {})


class KnowledgeComponent(Component):
    """贡献知识检索、可选管理工具及其调用限制。"""
    name = 'knowledge'

    def enabled(self, config, caller, retriever=None):
        return config.knowledge is not None or retriever is not None

    async def open(self, ctx):
        from ai_assistant.tools.context import KnowledgeService
        current = ctx.context.services.get('knowledge')
        ctx.services['knowledge'] = KnowledgeService(ctx.config, current.retriever if current else None)

    def contribute(self, ctx):
        from ai_assistant.knowledge.tools import knowledge_middleware, knowledge_tools
        return Contribution(tools=knowledge_tools(include_management=ctx.config.knowledge is not None),
            prompt_sections=[knowledge_section()],
            middleware=knowledge_middleware(run_limit=ctx.config.agent.knowledge_call_limit),
            services={'knowledge': ctx.services.get('knowledge')})


class MailComponent(Component):
    """按身份和配置启用邮件服务，并要求持久操作回执。"""
    name = 'mail'

    def enabled(self, config, caller, retriever=None):
        from ai_assistant.tools.mail import allowed
        return bool(config.mail.enabled and allowed(caller))

    async def open(self, ctx):
        if ctx.operations is None:
            from ai_assistant.config import ConfigError
            raise ConfigError('启用邮箱需要持久回执 [sessions]，请配置并显式 init。')
        try:
            await ctx.operations.by_source('mail:schema-check')
        except Exception:
            from ai_assistant.config import ConfigError
            raise ConfigError('邮箱来源回执结构不可用，请检查存储并显式 init 后重启。') from None
        from ai_assistant.tools.mail import MailService
        mail = MailService(ctx.config.mail, ctx.config.agent.timezone)
        ctx.stack.push_async_callback(mail.aclose)
        ctx.services['mail'] = mail

    def contribute(self, ctx):
        from ai_assistant.tools.mail import email_tools
        return Contribution(tools=email_tools(), services={'mail': ctx.services.get('mail')})


class PDFComponent(Component):
    """为符合条件的飞书主人私聊装配附件读取服务。"""
    name = 'pdf'

    def enabled(self, config, caller, retriever=None):
        return bool(config.tools.pdf.enabled and caller.owner
                    and caller.scope == 'private' and caller.channel == 'feishu_private'
                    and config.integrations.feishu is not None)

    async def open(self, ctx):
        import httpx
        from ai_assistant.integrations.feishu_files import FeishuAPI
        from ai_assistant.tools.pdf import PDFAttachments
        client = await ctx.stack.enter_async_context(httpx.AsyncClient(
            follow_redirects=False, trust_env=False, timeout=60))
        pdf = PDFAttachments(FeishuAPI(ctx.config.integrations.feishu, client),
                             ctx.config.tools.pdf.timeout)
        ctx.stack.callback(pdf.close)
        ctx.services['pdf'] = pdf

    def contribute(self, ctx):
        pdf = ctx.services.get('pdf')
        if pdf is None:
            return Contribution()
        from ai_assistant.tools.pdf import read_document_attachment
        return Contribution(tools=[read_document_attachment],
            prompt_sections=[attachment_section()], services={'pdf': pdf})


class FeishuIdentityComponent(Component):
    """装配飞书身份工具和未完成文档提示。"""
    name = 'feishu_identity'

    def enabled(self, config, caller, retriever=None):
        return config.integrations.feishu is not None

    def contribute(self, ctx):
        from ai_assistant.integrations.feishu_identity import identity_tools
        notices = []
        if ctx.config.integrations.feishu is not None:
            from ai_assistant.integrations.feishu_adapter import unfinished_document_notice
            notices.append(unfinished_document_notice)
        return Contribution(tools=identity_tools(ctx.caller), failure_notices=notices)


def build_components(config, caller, *, retriever=None):
    """按配置和可信调用者选择本次会话启用的组件。"""
    candidates = [CoreComponent(), SkillsComponent(), MCPComponent(), KnowledgeComponent(),
                  MailComponent(), PDFComponent(), FeishuIdentityComponent()]
    return [component for component in candidates if component.enabled(config, caller, retriever)]
