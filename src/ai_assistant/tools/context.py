"""Trusted tool dependencies and explicit asynchronous resource ownership."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.retrievers import BaseRetriever

import httpx

from ai_assistant.config import AppConfig
from ai_assistant.workspace.files import ReadOnlyWorkspace
from ai_assistant.tools.web import DeepSeekWeb
from ai_assistant.workspace.writes import WritableWorkspace


class ToolContextError(RuntimeError):
    """工具请求了未配置的运行时服务。"""


@dataclass(frozen=True)
class KnowledgeService:
    config: AppConfig | None = None
    retriever: BaseRetriever | None = None


@dataclass(frozen=True)
class FeishuService:
    owner_open_id: str | None = None
    default_task_assignee: str = 'none'
    task_members: str = 'model'
    default_calendar_id: str | None = None


@dataclass(frozen=True)
class ToolContext:
    workspace: ReadOnlyWorkspace
    effect_facts: list = field(default_factory=list, kw_only=True, repr=False, compare=False)
    inflight_tools: set = field(default_factory=set, kw_only=True, repr=False, compare=False)
    executed_tool_calls: set = field(default_factory=set, kw_only=True, repr=False, compare=False)
    web: DeepSeekWeb | None = None
    writes: WritableWorkspace | None = None
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    caller: object | None = None
    operations: object | None = None
    project_rules: str = ''
    input_id: str | None = None
    session_id: str | None = None
    execution_thread_id: str | None = None
    skills_directory_tokens: int = 6000
    timezone: str = 'Asia/Shanghai'
    resolved_datetimes: set[tuple[str, bool]] = field(default_factory=set, compare=False)
    usage_store: object | None = None
    capabilities: object | None = None
    services: dict = field(default_factory=dict, kw_only=True, compare=False)

    def service(self, name):
        try:
            return self.services[name]
        except KeyError:
            raise ToolContextError(f'Tool service not configured: {name}') from None


@asynccontextmanager
async def tool_context(config: AppConfig, *, retriever: BaseRetriever | None = None, skill_snapshot=None):
    base = config.config_path.parent if config.config_path else Path.cwd()
    root = Path(config.workspace.root)
    workspace = ReadOnlyWorkspace(
        root if root.is_absolute() else base / root,
        exclude_patterns=config.workspace.exclude_patterns, config_path=config.config_path, config_files=config.config_files,
        **config.tools.files.model_dump(exclude={"enabled"}),
    )
    lock = asyncio.Lock()
    async with AsyncExitStack() as stack:
        skills = None
        if config.skills is not None:
            from ai_assistant.skills.catalog import SkillCatalog, configured_roots
            skills = skill_snapshot or SkillCatalog(configured_roots(config), disabled=config.skills.disabled, config_files=config.config_files)
            if skill_snapshot is None:
                stack.callback(skills.close)
        settings = config.tools.web
        web = None
        if settings is not None:
            profile = config.models[settings.credential_profile]
            client = await stack.enter_async_context(httpx.AsyncClient(follow_redirects=False, trust_env=False))
            web = DeepSeekWeb(client, profile.api_key.get_secret_value(), settings.timeout,
                              search_chars=settings.search_chars, fetch_chars=settings.fetch_chars)
        services = {'knowledge': KnowledgeService(config, retriever)}
        if skills is not None:
            services['skills'] = skills
        yield ToolContext(workspace=workspace, web=web,
                          writes=WritableWorkspace(workspace) if config.tools.files.enabled else None,
                          operation_lock=lock,
                          skills_directory_tokens=config.skills.directory_tokens if config.skills else 6000,
                          services=services)
