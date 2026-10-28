"""Endpoint presets; model IDs and provider-specific options stay explicit."""

# name -> (LangChain integration, default public endpoint)
# These are ordinary API endpoints, not subscription-specific coding endpoints.
PROVIDER_PRESETS = {
    "deepseek": ("deepseek", "https://api.deepseek.com"),
    "glm": ("openai", "https://open.bigmodel.cn/api/paas/v4/"),
    "qwen": ("openai", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "kimi": ("openai", "https://api.moonshot.ai/v1"),
}

API_KEY_PROVIDERS = frozenset({"openai", "anthropic", "google_genai", *PROVIDER_PRESETS})


def is_deepseek(provider: str, model: str) -> bool:
    return provider == "deepseek" or (
        provider == "openai" and model.lower().startswith("deepseek-")
    )


def validate_deepseek_parameters(parameters: dict) -> None:
    """Reject alternate or contradictory controls before SDK initialization."""
    responses = parameters.get("use_responses_api", False)
    if type(responses) is not bool:
        raise ValueError("use_responses_api must be boolean")
    if responses:
        for container in (parameters, parameters.get("extra_body", {}), parameters.get("model_kwargs", {})):
            if not isinstance(container, dict):
                raise ValueError("Parameters must be tables")
            if {"previous_response_id", "conversation"}.intersection(container):
                raise ValueError("Responses requires full history replay")
            for key in ("use_previous_response_id", "store"):
                if key in container and container[key] not in (None, False):
                    raise ValueError("Responses requires full history replay")
            if "reasoning" in container and container["reasoning"] != {"effort": "none"}:
                raise ValueError("DeepSeek reasoning must be disabled")
            if "reasoning_effort" in container and container["reasoning_effort"] != "none":
                raise ValueError("DeepSeek reasoning must be disabled")
    if "thinking" in parameters:
        raise ValueError("Configure thinking only in extra_body")
    model_kwargs = parameters.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict) or any(
        key in model_kwargs for key in ("thinking", "extra_body", "use_responses_api")
    ):
        raise ValueError("Conflicting DeepSeek model_kwargs")
    body = parameters.get("extra_body", {})
    if not isinstance(body, dict):
        raise ValueError("extra_body must be a table")
    if "use_responses_api" in body:
        raise ValueError("Protocol cannot be overridden in extra_body")
    if "thinking" in body and body["thinking"] != {"type": "disabled"}:
        raise ValueError("DeepSeek thinking must be disabled")
