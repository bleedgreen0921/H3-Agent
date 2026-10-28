"""Bounded diagnostic probes using native interfaces and profile SDK retries."""

import asyncio
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass
import math
import json
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
from ai_assistant.llm.requests import model_session, prepare_request, response_usage as _usage
from ai_assistant.llm.providers import is_deepseek

CAPABILITIES = ("chat", "stream", "tools", "json_mode", "json_schema")


@dataclass(frozen=True)
class CheckResult:
    capability: str
    status: Literal["passed", "failed", "skipped"]
    elapsed_seconds: float = 0
    usage: dict[str, int] | None = None
    error: str | None = None
    http_status: int | None = None


class ProbeObject(BaseModel):
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


def _validate_response(message, *, deepseek: bool) -> None:
    if deepseek:
        extra = message.additional_kwargs
        if any(extra.get(key) for key in ("reasoning_content", "reasoning", "reasoning_details")):
            raise ProbeFailure("thinking_mismatch")
        if isinstance(message.content, list) and any(
            isinstance(block, dict) and block.get("type") in ("reasoning", "thinking")
            for block in message.content
        ):
            raise ProbeFailure("thinking_mismatch")
    if message.response_metadata.get("status") in ("incomplete", "failed"):
        raise ProbeFailure("truncated" if message.response_metadata["status"] == "incomplete" else "request_failed")
    if isinstance(message.content, list) and any(
        isinstance(block, dict) and block.get("type") == "refusal" for block in message.content
    ):
        raise ProbeFailure("refused")
    if message.response_metadata.get("finish_reason") == "length":
        raise ProbeFailure("truncated")
    if message.additional_kwargs.get("refusal") or message.response_metadata.get("finish_reason") in (
        "content_filter", "refusal",
    ):
        raise ProbeFailure("refused")


def _error(exc: Exception) -> tuple[str, int | None] | None:
    """Map known invocation failures for display; never stringify SDK errors."""
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


async def _probe(model, capability: str, deepseek: bool, record: dict):
    if capability == "stream":
        aggregate = None
        async with aclosing(model.astream("Reply with the word OK.")) as stream:
            async for chunk in stream:
                _validate_response(chunk, deepseek=deepseek)
                aggregate = chunk if aggregate is None else aggregate + chunk
        if aggregate is None:
            raise ProbeFailure("empty_output")
        message = aggregate
    elif capability == "tools":
        message = await model.bind_tools([TOOL], tool_choice="required").ainvoke(
            'Call connection_check exactly once with {"ok":true}. Do not answer with text.'
        )
    elif capability in ("json_mode", "json_schema"):
        response_format = {"type": "json_object"} if capability == "json_mode" else {
            "type": "json_schema", "json_schema": {"name": "connection_check", "strict": True, "schema": SCHEMA},
        }
        # Do not use with_structured_output: ChatDeepSeek substitutes tool calling
        # for json_schema, which would make this a false positive.
        message = await model.bind(response_format=response_format).ainvoke(
            'Return only this JSON object, with no extra fields: {"ok":true}'
        )
    else:
        message = await model.ainvoke("Reply with the word OK.")
    record["response"] = {"output": message.content, "text": message.text,
                          "status": message.response_metadata.get("status"),
                          "finish_reason": message.response_metadata.get("finish_reason"),
                          "tool_calls": message.tool_calls, "usage": _usage(message)}
    record["usage"] = _usage(message)
    _validate_response(message, deepseek=deepseek)
    if getattr(model, "use_responses_api", False) is True and message.response_metadata.get("status") != "completed":
        raise ProbeFailure("incomplete_response")
    if capability == "tools":
        calls = message.tool_calls
        if message.invalid_tool_calls or len(calls) != 1 or calls[0]["name"] != "connection_check":
            raise ProbeFailure("invalid_tool_call")
        ProbeObject.model_validate(calls[0]["args"])
    else:
        if message.tool_calls or message.invalid_tool_calls:
            raise ProbeFailure("unexpected_tool_call")
        if not message.text.strip():
            raise ProbeFailure("empty_output")
        if capability in ("json_mode", "json_schema"):
            try:
                value = json.loads(message.text)
            except (ValueError, TypeError):
                raise ProbeFailure("invalid_json") from None
            if not isinstance(value, dict) or set(value) != {"ok"} or type(value["ok"]) is not bool:
                raise ProbeFailure("invalid_fields")
            if value["ok"] is not True:
                raise ProbeFailure("unexpected_value")


async def check_model(
    config: AppConfig, name: str | None = None, capability: str = "all", timeout: float = 60,
    *, on_result: Callable[[CheckResult], None] | None = None,
    on_evidence: Callable[[str, dict], None] | None = None,
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
    prepared = prepare_request(config, selected, timeout, max_tokens=256)
    profile = prepared.models[selected]
    deepseek = is_deepseek(profile.provider, profile.model)
    results = []
    async with model_session(prepared, selected, timeout, factory=create_model) as model:
        for item in CAPABILITIES:
            if capability not in ("all", item):
                result = CheckResult(item, "skipped")
            else:
                start, record = monotonic(), {}
                try:
                    async with asyncio.timeout(timeout):
                        await _probe(model, item, deepseek, record)
                    result = CheckResult(item, "passed", monotonic() - start, record.get("usage"))
                except Exception as exc:
                    failure = _error(exc)
                    if failure is None:
                        raise
                    category, status = failure
                    result = CheckResult(item, "failed", monotonic() - start, record.get("usage"), category, status)
            if on_evidence is not None and capability in ("all", item):
                on_evidence(item, record)
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results
