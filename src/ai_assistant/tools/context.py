"""Trusted tool dependencies and explicit asynchronous resource ownership."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from ai_assistant.config import AppConfig
from ai_assistant.workspace.files import ReadOnlyWorkspace
from ai_assistant.tools.web import DeepSeekWeb


@dataclass(frozen=True)
class ToolContext:
    workspace: ReadOnlyWorkspace
    web: DeepSeekWeb | None = None


@asynccontextmanager
async def tool_context(config: AppConfig):
    base = config.config_path.parent if config.config_path else Path.cwd()
    root = Path(config.workspace.root)
    workspace = ReadOnlyWorkspace(
        root if root.is_absolute() else base / root,
        exclude_patterns=config.workspace.exclude_patterns, config_path=config.config_path,
    )
    settings = config.tools.web
    if settings is None:
        yield ToolContext(workspace)
    else:
        profile = config.models[settings.credential_profile]
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
            yield ToolContext(workspace, DeepSeekWeb(
                client, profile.api_key.get_secret_value(), settings.timeout,
            ))
