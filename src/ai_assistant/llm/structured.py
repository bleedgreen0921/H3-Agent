"""Provider-independent structured output verification.

Native ``with_structured_output`` is preferred, but several OpenAI-compatible
providers occasionally return the JSON as normal assistant text instead of a
tool call. Treating ``parsed=None`` as a schema mistake loses that valid
response; this layer verifies the raw message and accepts either transport
while still validating the same Pydantic schema.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from ai_assistant.llm.completion import validate_completion

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Safe summary; never includes model content or credentials."""


def _message_text(message):
    """从原生消息中提取普通文本和文本内容块。"""
    if not isinstance(message, BaseMessage):
        return ''
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content if isinstance(content, list) else ():
        if isinstance(block, dict) and block.get('type') in {'text', 'output_text'}:
            parts.append(str(block.get('text', '')))
        elif isinstance(block, str):
            parts.append(block)
    return ''.join(parts)


def _json_candidates(text):
    """Yield likely JSON objects from plain text without accepting arbitrary text."""
    value = (text or '').strip()
    if not value:
        return
    fenced = re.sub(r'^```(?:json)?\s*|\s*```$', '', value, flags=re.IGNORECASE)

    # 先尝试完整文本或代码围栏，再尝试提取其中的 JSON 对象。
    for candidate in (value, fenced):
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            yield parsed
    start, end = value.find('{'), value.rfind('}')
    if 0 <= start < end:
        try:
            parsed = json.loads(value[start:end + 1])
        except (TypeError, ValueError):
            pass
        else:
            if isinstance(parsed, dict):
                yield parsed


def parse_structured_result(schema: type[SchemaT], result: Any, *, responses=False) -> SchemaT:
    """Validate native parsed output, then raw JSON text as a transport fallback."""
    if isinstance(result, schema):
        return result
    if not isinstance(result, dict):
        raw = None
        parsed = result
        parsing_error = None
    else:
        raw, parsed, parsing_error = (
            result.get('raw'), result.get('parsed'), result.get('parsing_error'))
    if raw is not None:
        validate_completion(raw, responses=responses)

    # 优先使用原生解析结果；无效时仍允许符合相同 schema 的 JSON 文本。
    if isinstance(parsed, schema):
        return parsed
    if parsed is not None:
        try:
            return schema.model_validate(parsed)
        except ValidationError:
            pass
        if isinstance(parsed, str):
            for candidate in _json_candidates(parsed):
                try:
                    return schema.model_validate(candidate)
                except ValidationError:
                    continue
    text = _message_text(raw)
    for candidate in _json_candidates(text):
        try:
            return schema.model_validate(candidate)
        except ValidationError:
            continue
    if parsing_error is not None:
        raise StructuredOutputError(
            '模型返回了无法解析的结构化内容，请重试或缩小目标。') from None
    if not text.strip():
        raise StructuredOutputError('模型没有返回可解析的结构化内容，请重试。')
    raise StructuredOutputError(
        '模型返回了非结构化内容；请重试，或减少目标范围后重新规划。')


async def invoke_structured(model, schema: type[SchemaT], messages, *,
                            config=None, method='function_calling'):
    """Invoke native structured output and verify both transport forms."""
    # 保留原始消息，以便检查完成状态并兼容文本形式的结构化输出。
    runnable = model.with_structured_output(schema, method=method, include_raw=True)
    result = await runnable.ainvoke(messages, config=config)
    return parse_structured_result(
        schema, result,
        responses=getattr(model, 'use_responses_api', False) is True)
