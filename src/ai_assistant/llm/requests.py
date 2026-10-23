"""Diagnostic request preparation and shared model-client ownership."""

from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from langchain_core.language_models import BaseChatModel

from ai_assistant.config import AppConfig, ConfigError


REQUEST_CONTROLS = frozenset({
    "max_tokens", "max_completion_tokens", "max_output_tokens", "timeout",
    "stream", "streaming", "response_format", "tools", "tool_choice",
})


def prepare_request(config: AppConfig, selected: str, timeout: float, max_tokens: int) -> AppConfig:
    """Copy diagnostic limits while preserving the profile SDK retry policy."""
    profile = config.models[selected]
    parameters = dict(profile.parameters)
    for container in ("extra_body", "model_kwargs"):
        nested = parameters.get(container, {})
        if not isinstance(nested, dict) or (REQUEST_CONTROLS | {"max_retries"}).intersection(nested):
            raise ConfigError("Request controls must not be nested in extra_body or model_kwargs.")
    for key in REQUEST_CONTROLS:
        parameters.pop(key, None)
    parameters.update(timeout=timeout, max_tokens=max_tokens, cache=False)
    copied = profile.model_copy(update={"parameters": parameters})
    return config.model_copy(update={"models": {**config.models, selected: copied}})


@asynccontextmanager
async def model_session(
    prepared: AppConfig, selected: str, timeout: float,
    *, factory: Callable[..., BaseChatModel],
):
    """Close owned clients on every exit; caller-supplied clients stay open."""
    profile = prepared.models[selected]
    parameters = dict(profile.parameters)
    async with AsyncExitStack() as stack:
        if profile.provider in {"openai", "deepseek", "glm", "qwen", "kimi"}:
            if "http_async_client" not in parameters:
                parameters["http_async_client"] = await stack.enter_async_context(httpx.AsyncClient(timeout=timeout))
            if "http_client" not in parameters:
                parameters["http_client"] = stack.enter_context(httpx.Client(timeout=timeout))
        copied = profile.model_copy(update={"parameters": parameters})
        bound = prepared.model_copy(update={"models": {**prepared.models, selected: copied}})
        yield factory(bound, selected)


def response_usage(message) -> dict[str, int] | None:
    raw = message.usage_metadata or {}
    result = {key: raw[key] for key in ("input_tokens", "output_tokens", "total_tokens")
              if type(raw.get(key)) is int and raw[key] >= 0}
    return result or None
