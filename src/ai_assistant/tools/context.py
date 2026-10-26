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
from ai_assistant.tools.terminal import TerminalService


@dataclass(frozen=True)
class ToolContext:
    workspace: ReadOnlyWorkspace
    web: DeepSeekWeb | None = None
    writes: WritableWorkspace | None = None
    terminal: TerminalService | None = None
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    retriever: BaseRetriever | None = None
    skills: object | None = None
    mcp: object | None = None
    caller: object | None = None
    operations: object | None = None
    input_id: str | None = None
    knowledge_config: object | None = None
    memory: object | None = None
    checkpoint_history: object | None = None
    mcp_schema_tokens: int = 6000
    timezone: str = 'Asia/Shanghai'
    owner_open_id: str | None = None
    default_task_assignee: str = 'none'
    resolved_datetimes: set[tuple[str, bool]] = field(default_factory=set, compare=False)
    task_outcomes: dict = field(default_factory=dict, compare=False)
    usage_store: object | None = None


@asynccontextmanager
async def tool_context(config: AppConfig, *, retriever: BaseRetriever | None = None, skill_snapshot=None):
    base = config.config_path.parent if config.config_path else Path.cwd()
    root = Path(config.workspace.root)
    workspace = ReadOnlyWorkspace(
        root if root.is_absolute() else base / root,
        exclude_patterns=config.workspace.exclude_patterns, config_path=config.config_path,
    )
    lock = asyncio.Lock()
    terminal_settings = config.tools.terminal
    terminal = TerminalService(workspace, timeout=terminal_settings.timeout, lock=lock,
        allow_network=terminal_settings.allow_network,
        env={k: v.get_secret_value() for k, v in terminal_settings.env.items()},
        read_paths=[p.expanduser() if p.expanduser().is_absolute() else base / p
                    for p in terminal_settings.read_paths]) if terminal_settings else None
    async with AsyncExitStack() as stack:
        skills = None
        if config.skills is not None:
            from ai_assistant.skills.catalog import SkillCatalog, configured_roots
            skills = skill_snapshot or SkillCatalog(configured_roots(config))
            if skill_snapshot is None:
                stack.callback(skills.close)
        settings = config.tools.web
        web = None
        if settings is not None:
            profile = config.models[settings.credential_profile]
            client = await stack.enter_async_context(httpx.AsyncClient(follow_redirects=False, trust_env=False))
            web = DeepSeekWeb(client, profile.api_key.get_secret_value(), settings.timeout)
        if terminal is not None:
            stack.push_async_callback(terminal.aclose)
        yield ToolContext(workspace, web,
                          WritableWorkspace(workspace) if config.tools.files.enabled else None,
                          terminal, lock, retriever, skills)
