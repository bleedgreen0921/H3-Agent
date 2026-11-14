"""Bounded diagnostic probes using native interfaces and profile SDK retries."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
from time import monotonic
from typing import Literal

import httpx
from langchain_core.exceptions import (
    ModelError, ModelAuthenticationError, ModelPermissionDeniedError,
    ModelNotFoundError, ModelRateLimitError, ModelConnectionError, ModelTimeoutError,
    OutputParserException,
)
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ai_assistant.config import AppConfig, ConfigError
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.requests import model_session, response_usage as _usage
from ai_assistant.llm.completion import validate_completion, IncompleteResponseError

CAPABILITIES = ("chat", "tools")


@dataclass(frozen=True)
class CheckResult:
    """单项真实模型检查的状态、耗时、用量及安全错误摘要。"""
    capability: str
    status: Literal["passed", "failed", "skipped"]
    elapsed_seconds: float = 0
    usage: dict[str, int] | None = None
    error: str | None = None
    http_status: int | None = None


class ProbeObject(BaseModel):
    """工具调用探针期望返回的严格参数结构。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    ok: bool

    @field_validator("ok")
    @classmethod
    def require_true(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("Expected true")
        return value


SCHEMA = {
    "type": "object", "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"], "additionalProperties": False,
}
TOOL = {"type": "function", "function": {
    "name": "connection_check", "description": "Return the requested connection check flag.",
    "parameters": SCHEMA,
}}


class ProbeFailure(Exception):
    """Carries a fixed diagnostic code, never model output."""


def _error(exc: Exception) -> tuple[str, int | None] | None:
    """Map known invocation failures for display; never stringify SDK errors."""
    if isinstance(exc, IncompleteResponseError):
        return "incomplete_response", None
    if isinstance(exc, ProbeFailure):
        return str(exc), None
    if isinstance(exc, (ValidationError, OutputParserException)):
        return "invalid_output_or_parameters", None
    if isinstance(exc, NotImplementedError):
        return "unsupported_interface", None
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, ModelTimeoutError)):
        return "timeout", None
    if isinstance(exc, (httpx.TransportError, ModelConnectionError)):
        return "connection", None
    # Optional SDK types cover errors raised before native message conversion.
    try:
        from openai.types.responses import ResponseError
        from openai import (
            OpenAIError, APIConnectionError, APITimeoutError,
            ContentFilterFinishReasonError, LengthFinishReasonError,
        )
    except ImportError:
        sdk_error = False
    else:
        if isinstance(exc, APITimeoutError):
            return "timeout", None
        if isinstance(exc, APIConnectionError):
            return "connection", None
        if isinstance(exc, LengthFinishReasonError):
            return "truncated", None
        if isinstance(exc, ContentFilterFinishReasonError):
            return "refused", None
        if isinstance(exc, ValueError) and exc.args and isinstance(exc.args[0], ResponseError):
            return "request_failed", None
        sdk_error = isinstance(exc, OpenAIError)
    if isinstance(exc, (ModelError, httpx.HTTPStatusError)) or sdk_error:
        status = getattr(exc, "status_code", None)
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
        if type(status) is int and 400 <= status <= 599:
            return {401: "authentication", 402: "balance", 403: "permission", 404: "not_found",
                    429: "rate_limit"}.get(status, "http_error"), status
        for error_type, code in (
            (ModelAuthenticationError, "authentication"),
            (ModelPermissionDeniedError, "permission"),
            (ModelNotFoundError, "not_found"),
            (ModelRateLimitError, "rate_limit"),
        ):
            if isinstance(exc, error_type):
                return code, None
        return "request_failed", None
    return None


async def _probe(model, capability: str, record: dict):
    """分别检查普通文本响应或一次指定工具调用。"""
    if capability == "tools":
        message = await model.bind_tools([TOOL], tool_choice="required").ainvoke(
            'Call connection_check exactly once with {"ok":true}. Do not answer with text.'
        )
    else:
        message = await model.ainvoke("Reply with the word OK.")
    record["usage"] = _usage(message)
    validate_completion(message, responses=getattr(model, "use_responses_api", False) is True)

    # 工具检查必须返回指定调用及合法参数；聊天检查要求非空文本。
    if capability == "tools":
        calls = message.tool_calls
        if len(calls) != 1 or calls[0]["name"] != "connection_check":
            raise ProbeFailure("invalid_tool_call")
        ProbeObject.model_validate(calls[0]["args"])
    else:
        if message.tool_calls:
            raise ProbeFailure("unexpected_tool_call")
        if not message.text.strip():
            raise ProbeFailure("empty_output")


async def check_model(
    config: AppConfig, name: str | None = None, capability: str = "all", timeout: float = 60,
    *, on_result: Callable[[CheckResult], None] | None = None,
) -> list[CheckResult]:
    """Each selected capability is invoked once; the profile controls SDK retries.

    ConfigError denotes a setup failure before any probes. CancelledError is
    deliberately propagated so the CLI can handle Ctrl+C and close resources.
    """
    if capability not in (*CAPABILITIES, "all"):
        raise ConfigError("Unknown capability.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError("Timeout must be a finite positive number.")
    selected = config.default_model if name is None else name
    if selected not in config.models:
        raise ConfigError("Unknown model profile.")
    results = []

    # 显式诊断会发送真实请求；每项独立限时并沿用方案的 SDK 重试设置。
    async with model_session(config, selected, None, factory=create_model) as model:
        for item in CAPABILITIES:
            if capability not in ("all", item):
                result = CheckResult(item, "skipped")
            else:
                start, record = monotonic(), {}
                try:
                    async with asyncio.timeout(timeout):
                        await _probe(model, item, record)
                    result = CheckResult(item, "passed", monotonic() - start, record.get("usage"))
                except Exception as exc:
                    # 已知错误转换为固定分类；未知异常继续传播。
                    failure = _error(exc)
                    if failure is None:
                        raise
                    category, status = failure
                    result = CheckResult(item, "failed", monotonic() - start, record.get("usage"), category, status)
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results
