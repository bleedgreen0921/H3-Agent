"""Shared model-client ownership without rewriting profile request parameters."""

from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from langchain_core.language_models import BaseChatModel

from ai_assistant.config import AppConfig, ConfigError
from ai_assistant.llm.providers import uses_openai_client


@asynccontextmanager
async def model_session(
    prepared: AppConfig, selected: str, timeout: float | None,
    *, factory: Callable[..., BaseChatModel],
):
    """Close owned clients on every exit; caller-supplied clients stay open."""
    profile = prepared.models[selected]
    parameters = dict(profile.parameters)
    if timeout is not None:
        parameters.setdefault("timeout", timeout)
    async with AsyncExitStack() as stack:
        if uses_openai_client(profile.provider):
            try:
                from openai import DEFAULT_TIMEOUT
            except ImportError:
                raise ConfigError("Install the selected model provider dependencies.") from None
            client_timeout = parameters.get("timeout", DEFAULT_TIMEOUT)
            if "http_async_client" not in parameters:
                parameters["http_async_client"] = await stack.enter_async_context(httpx.AsyncClient(timeout=client_timeout))
            if "http_client" not in parameters:
                parameters["http_client"] = stack.enter_context(httpx.Client(timeout=client_timeout))
        copied = profile.model_copy(update={"parameters": parameters})
        bound = prepared.model_copy(update={"models": {**prepared.models, selected: copied}})
        yield factory(bound, selected)


def response_usage(message) -> dict[str, int] | None:
    raw = message.usage_metadata or {}
    result = {key: raw[key] for key in ("input_tokens", "output_tokens", "total_tokens")
              if type(raw.get(key)) is int and raw[key] >= 0}
    return result or None
