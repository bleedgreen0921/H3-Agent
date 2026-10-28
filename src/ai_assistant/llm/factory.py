"""Create native LangChain chat models from application configuration."""

from copy import deepcopy

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from ai_assistant.config import AppConfig, ConfigError
from ai_assistant.llm.providers import PROVIDER_PRESETS, is_deepseek, validate_deepseek_parameters


def create_model(config: AppConfig, name: str | None = None) -> BaseChatModel:
    """Create only the selected profile; do not cache or alter process settings.

    Provider packages control their own non-key defaults and client lifecycle.
    Other providers may use native SDK authentication when api_key is omitted.
    """
    selected = config.default_model if name is None else name
    if selected not in config.models:
        raise ValueError("Unknown model profile.")
    profile = config.models[selected]
    options = dict(profile.parameters)
    if is_deepseek(profile.provider, profile.model):
        try:
            validate_deepseek_parameters(options)
        except ValueError:
            raise ConfigError("DeepSeek 参数冲突；请关闭思考并使用完整历史回放。") from None
        body = deepcopy(options.get("extra_body", {}))
        if options.get("use_responses_api"):
            # Responses 由原生 OpenAI 适配器转换消息、工具与结构化输出。
            body.pop("thinking", None)
            body.pop("reasoning", None)
            body.pop("reasoning_effort", None)
            kwargs = deepcopy(options.get("model_kwargs", {}))
            kwargs.pop("reasoning", None)
            kwargs.pop("reasoning_effort", None)
            if "model_kwargs" in options:
                options["model_kwargs"] = kwargs
            options.pop("reasoning_effort", None)
            options.update(reasoning={"effort": "none"}, store=False,
                           use_previous_response_id=False)
        else:
            body["thinking"] = {"type": "disabled"}
            options["use_responses_api"] = False
        options["extra_body"] = body
    provider = profile.provider
    preset = PROVIDER_PRESETS.get(provider)
    base_url = profile.base_url
    if preset is not None:
        provider, default_url = preset
        if base_url is None:
            base_url = default_url
        if provider == "openai":
            # Compatible providers need not implement OpenAI's Responses API or
            # optional stream usage extensions. Explicit settings still win.
            options.setdefault("use_responses_api", False)
            options.setdefault("stream_usage", False)
    if is_deepseek(profile.provider, profile.model) and options.get("use_responses_api"):
        provider = "openai"
    if base_url is not None:
        options["base_url"] = base_url
    if profile.api_key is not None:
        options["api_key"] = profile.api_key.get_secret_value()
    try:
        return init_chat_model(profile.model, model_provider=provider, **options)
    except ImportError:
        raise ConfigError("Model integration is unavailable; install the selected provider package.") from None
    except (ValueError, TypeError):
        # Provider validation exceptions can embed constructor inputs and keys.
        raise ConfigError("Unable to initialize model; check provider, model and parameters.") from None
