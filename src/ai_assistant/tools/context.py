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
    """知识库配置及可选的已注入 Retriever。"""
    config: AppConfig | None = None
    retriever: BaseRetriever | None = None


@dataclass(frozen=True)
class FeishuService:
    """工具调用需要的飞书主人身份和默认业务策略。"""
    owner_open_id: str | None = None
    default_task_assignee: str = 'none'
    task_members: str = 'model'
    default_calendar_id: str | None = None
    document_identity: str = 'user'


@dataclass(frozen=True)
class ToolContext:
    """由可信运行时构造的工具依赖、调用身份和轮次状态。"""
    workspace: ReadOnlyWorkspace

    # 可变集合记录本轮效果、资源身份及执行中的工具，不进入常规对象展示。
    effect_facts: list = field(default_factory=list, kw_only=True, repr=False, compare=False)
    # 本轮已创建/已绑定的外部资源 -> (server, identity)；防止 bot 创建后 user 写入。
    resource_bindings: dict = field(default_factory=dict, kw_only=True, repr=False, compare=False)
    inflight_tools: set = field(default_factory=set, kw_only=True, repr=False, compare=False)
    executed_tool_calls: set = field(default_factory=set, kw_only=True, repr=False, compare=False)

    # 可选联网与写入服务；写操作通过上下文锁协调。
    web: DeepSeekWeb | None = None
    writes: WritableWorkspace | None = None
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    # 身份、操作存储和关联 ID 由运行时提供，而非模型参数决定。
    caller: object | None = None
    operations: object | None = None
    project_rules: str = ''
    input_id: str | None = None
    session_id: str | None = None
    execution_thread_id: str | None = None
    task_id: str | None = None
    skills_directory_tokens: int = 6000
    timezone: str = 'Asia/Shanghai'

    # 可信时间转换、用量、能力计划及其他按名称访问的服务。
    resolved_datetimes: set[tuple[str, bool]] = field(default_factory=set, compare=False)
    usage_store: object | None = None
    capabilities: object | None = None
    services: dict = field(default_factory=dict, kw_only=True, compare=False)

    def service(self, name):
        """获取已装配服务；缺失时给出明确错误。"""
        try:
            return self.services[name]
        except KeyError:
            raise ToolContextError(f'Tool service not configured: {name}') from None


@asynccontextmanager
async def tool_context(config: AppConfig, *, retriever: BaseRetriever | None = None, skill_snapshot=None):
    """按显式配置装配工作区和可选服务，并托管本作用域创建的资源。"""
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
            # 复用外部快照，或自行创建并在退出时关闭 Skill 目录。
            from ai_assistant.skills.catalog import SkillCatalog, configured_roots
            skills = skill_snapshot or SkillCatalog(configured_roots(config), disabled=config.skills.disabled, config_files=config.config_files)
            if skill_snapshot is None:
                stack.callback(skills.close)
        settings = config.tools.web
        web = None
        if settings is not None:
            # 只有显式配置联网工具时才创建 HTTP 客户端和服务。
            profile = config.models[settings.credential_profile]
            client = await stack.enter_async_context(httpx.AsyncClient(follow_redirects=False, trust_env=False))
            web = DeepSeekWeb(client, profile.api_key.get_secret_value(), settings.timeout,
                              search_chars=settings.search_chars, fetch_chars=settings.fetch_chars)
        services = {'knowledge': KnowledgeService(config, retriever)}
        if skills is not None:
            services['skills'] = skills

        # 文件写入仅在配置启用时装配；其余运行时身份由后续调用方注入。
        yield ToolContext(workspace=workspace, web=web,
                          writes=WritableWorkspace(workspace) if config.tools.files.enabled else None,
                          operation_lock=lock,
                          skills_directory_tokens=config.skills.directory_tokens if config.skills else 6000,
                          services=services)
