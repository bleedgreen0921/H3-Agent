"""Native DeepSeek structured Runnables with explicit method selection."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel


def create_structured_runnable(
    model: BaseChatModel,
    schema: type[BaseModel],
    *,
    method: Literal["json_mode", "function_calling", "json_schema"] = "json_mode",
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
    if method == "json_schema":
        # ChatDeepSeek silently substitutes tool calling for this method.
        raise ValueError("DeepSeek native JSON Schema is unsupported; select another method explicitly.")
    if method not in ("json_mode", "function_calling"):
        raise ValueError("Unknown structured output method.")
    return model.with_structured_output(schema, method=method, include_raw=include_raw)
