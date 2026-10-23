"""Explicit, single-file application configuration without environment mutation."""

from pathlib import Path
import tomllib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic import field_validator, model_validator, PrivateAttr

from ai_assistant.llm.providers import API_KEY_PROVIDERS, is_deepseek, validate_deepseek_parameters


class ConfigError(ValueError):
    """A configuration error safe to display without exposing file contents."""


class ModelConfig(BaseModel):
    """A model profile; credentials are unwrapped only when creating a model."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    base_url: str | None = Field(default=None, min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "model", "base_url")
    @classmethod
    def reject_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @field_validator("api_key")
    @classmethod
    def reject_blank_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("API key must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def reject_reserved_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = {
            "model", "model_name", "model_provider", "api_key", "openai_api_key",
            "google_api_key", "anthropic_api_key", "api_key_env", "base_url",
            "openai_api_base", "configurable_fields", "config_prefix",
        }
        if reserved.intersection(value):
            raise ValueError("Use explicit configuration fields for reserved parameters")
        return value

    @model_validator(mode="after")
    def require_key(self) -> "ModelConfig":
        if self.provider in API_KEY_PROVIDERS and self.api_key is None:
            raise ValueError("This provider requires api_key in the configuration file")
        if is_deepseek(self.provider, self.model):
            validate_deepseek_parameters(self.parameters)
        return self


class WorkspaceConfig(BaseModel):
    """Shared workspace access settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    root: str = Field(default=".", min_length=1)
    exclude_patterns: list[str] = Field(default_factory=list)


class WebConfig(BaseModel):
    """DeepSeek web credentials and request deadline."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    credential_profile: str = Field(min_length=1)
    timeout: float = Field(default=30, gt=0, allow_inf_nan=False)


class FilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    enabled: bool = False


class TerminalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    profile: Literal["project", "scratch", "maintenance"] = "project"
    timeout: float = Field(default=120, gt=0, le=1800, allow_inf_nan=False)


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    web: WebConfig | None = None
    files: FilesConfig = Field(default_factory=FilesConfig)
    terminal: TerminalConfig | None = None


from ai_assistant.knowledge.config import KnowledgeConfig


class AppConfig(BaseModel):
    """Validated application settings; no clients or global state are stored."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    knowledge: KnowledgeConfig | None = None
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    _config_path: Path | None = PrivateAttr(default=None)

    @property
    def config_path(self) -> Path | None:
        return self._config_path

    default_model: str = Field(min_length=1)
    models: dict[str, ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_names(self) -> "AppConfig":
        if any(not name.strip() for name in self.models):
            raise ValueError("Model profile names must not be blank")
        if self.default_model not in self.models:
            raise ValueError("default_model must refer to an existing model profile")
        if self.tools.web is not None:
            profile = self.models.get(self.tools.web.credential_profile)
            if profile is None or profile.provider != "deepseek" or profile.api_key is None:
                raise ValueError("Web tools require a DeepSeek credential profile")
        return self


def load_config(path: str | Path = "assistant.toml") -> AppConfig:
    """Read exactly the given path, relative to the current directory if needed.

    Errors intentionally omit parser snippets and validation inputs, which can
    contain credentials. No dotenv, interpolation or parent-directory search.
    """
    try:
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError:
        raise ConfigError("Configuration file not found; specify --config or create assistant.toml.") from None
    except PermissionError:
        raise ConfigError("Permission denied when reading the configuration file.") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise ConfigError("Invalid TOML or UTF-8 in the configuration file.") from None
    except OSError:
        raise ConfigError("Unable to read the configuration file.") from None
    try:
        config = AppConfig.model_validate(data)
        config._config_path = Path(path).absolute()
        return config
    except ValidationError as exc:
        # Field locations and raw exception text may themselves contain secrets.
        raise ConfigError(
            f"Invalid configuration ({exc.error_count()} validation errors); check profile fields, default_model and required api_key values."
        ) from None
