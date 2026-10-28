"""Native DeepSeek structured Runnables with explicit method selection."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel


def create_structured_runnable(
    model: BaseChatModel,
    schema: type[BaseModel],
    *,
    method: Literal["json_mode", "function_calling", "json_schema"] = "json_schema",
    include_raw: bool = False,
) -> Runnable:
    """Delegate parsing, output types and exceptions to the native integration.

    Compose JSON/schema instructions with messages or ChatPromptTemplate at the
    call site. Model configuration owns retries, timeouts and client lifetime.
    """
    from langchain_openai import ChatOpenAI

    # Do not require the DeepSeek extra for an OpenAI-compatible DeepSeek model.
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError:
        native_deepseek = False
    else:
        native_deepseek = isinstance(model, ChatDeepSeek)
    compatible_deepseek = isinstance(model, ChatOpenAI) and model.model_name.lower().startswith("deepseek-")
    if not (native_deepseek or compatible_deepseek):
        raise ValueError("Structured generation currently supports DeepSeek only.")
    if method == "json_schema" and not (compatible_deepseek and model.use_responses_api):
        # ChatDeepSeek silently substitutes tool calling for this method.
        raise ValueError("DeepSeek native JSON Schema is unsupported; select another method explicitly.")
    if method not in ("json_mode", "function_calling", "json_schema"):
        raise ValueError("Unknown structured output method.")
    if method != "json_schema":
        return model.with_structured_output(schema, method=method, include_raw=include_raw)
    from langchain_core.runnables import RunnableLambda
    from ai_assistant.llm.completion import validate_completion

    # 先检查原始响应，再沿用原生解析异常与 include_raw 返回约定。
    def checked(result):
        validate_completion(result["raw"], responses=True)
        if include_raw:
            return result
        if result["parsing_error"] is not None:
            raise result["parsing_error"]
        return result["parsed"]

    return model.with_structured_output(
        schema, method=method, strict=True, include_raw=True,
    ) | RunnableLambda(checked)
