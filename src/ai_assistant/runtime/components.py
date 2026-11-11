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
    tools: list = field(default_factory=list)
    prompt_sections: list[PromptSection] = field(default_factory=list)
    middleware: list = field(default_factory=list)
    services: dict = field(default_factory=dict)
    failure_notices: list = field(default_factory=list)


class Component:
    name = 'component'

    def enabled(self, config, caller, retriever=None):
        return True

    async def open(self, ctx: ComponentContext):
        return None

    def contribute(self, ctx: ComponentContext) -> Contribution:
        return Contribution()


class CoreComponent(Component):
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
    name = 'skills'

    def enabled(self, config, caller, retriever=None):
        return config.skills is not None

    def contribute(self, ctx):
        if ctx.config.skills is None or ctx.context.skills is None:
            return Contribution()
        from ai_assistant.agents.skills import SkillsMiddleware
        from ai_assistant.tools.skills import create_skill_tools
        return Contribution(tools=create_skill_tools(),
            prompt_sections=[skills_section()],
            middleware=[SkillsMiddleware(ctx.max_input_tokens)])


class MCPComponent(Component):
    name = 'mcp'

    def enabled(self, config, caller, retriever=None):
        return bool(config.mcp and config.mcp.servers)

    async def open(self, ctx):
        from ai_assistant.integrations.mcp import mcp_resources
        directory = getattr(ctx.resources, 'catalog', None) if ctx.resources is not None else None
        if directory is None:
            directory = await ctx.stack.enter_async_context(
                mcp_resources(ctx.config, ctx.context.workspace, caller=ctx.caller))
        ctx.update_context(mcp=directory.for_caller(ctx.caller))

    def contribute(self, ctx):
        from ai_assistant.agents.mcp import MCPMiddleware
        catalog = getattr(ctx.context, 'mcp', None)
        return Contribution(tools=[entry.tool for entry in catalog.entries.values()] if catalog else [],
                            middleware=[MCPMiddleware()])


class KnowledgeComponent(Component):
    name = 'knowledge'

    def enabled(self, config, caller, retriever=None):
        return config.knowledge is not None or retriever is not None

    async def open(self, ctx):
        ctx.update_context(knowledge_config=ctx.config)

    def contribute(self, ctx):
        from ai_assistant.knowledge.tools import knowledge_middleware, knowledge_tools
        return Contribution(tools=knowledge_tools(include_management=ctx.config.knowledge is not None),
            prompt_sections=[knowledge_section()],
            middleware=knowledge_middleware(run_limit=ctx.config.agent.knowledge_call_limit),
            services={'knowledge': ctx.context.knowledge_config})


class MailComponent(Component):
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
        ctx.update_context(mail=mail)

    def contribute(self, ctx):
        from ai_assistant.tools.mail import email_tools
        return Contribution(tools=email_tools(), services={'mail': ctx.services.get('mail')})


class PDFComponent(Component):
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
        ctx.update_context(pdf=pdf)

    def contribute(self, ctx):
        if ctx.context.pdf is None:
            return Contribution()
        from ai_assistant.tools.pdf import read_document_attachment
        return Contribution(tools=[read_document_attachment],
            prompt_sections=[attachment_section()], services={'pdf': ctx.context.pdf})


class FeishuIdentityComponent(Component):
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
    candidates = [CoreComponent(), SkillsComponent(), MCPComponent(), KnowledgeComponent(),
                  MailComponent(), PDFComponent(), FeishuIdentityComponent()]
    return [component for component in candidates if component.enabled(config, caller, retriever)]
