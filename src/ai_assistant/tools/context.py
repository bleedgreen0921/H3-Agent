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


@asynccontextmanager
async def tool_context(config: AppConfig, *, retriever: BaseRetriever | None = None):
    base = config.config_path.parent if config.config_path else Path.cwd()
    root = Path(config.workspace.root)
    workspace = ReadOnlyWorkspace(
        root if root.is_absolute() else base / root,
        exclude_patterns=config.workspace.exclude_patterns, config_path=config.config_path,
    )
    lock = asyncio.Lock()
    terminal_settings = config.tools.terminal
    terminal = TerminalService(workspace, profile=terminal_settings.profile,
                               timeout=terminal_settings.timeout, lock=lock) if terminal_settings else None
    async with AsyncExitStack() as stack:
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
                          terminal, lock, retriever)
