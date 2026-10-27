"""Explicit, single-file application configuration without environment mutation."""

from pathlib import Path
import tomllib
from typing import Any

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
    max_input_tokens: int | None = Field(default=None, gt=0, strict=True)

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
    allow_network: bool = False
    env: dict[str, SecretStr] = Field(default_factory=dict)
    read_paths: list[Path] = Field(default_factory=list)
    timeout: float = Field(default=120, gt=0, le=1800, allow_inf_nan=False)


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    web: WebConfig | None = None
    files: FilesConfig = Field(default_factory=FilesConfig)
    terminal: TerminalConfig | None = None


from ai_assistant.knowledge.config import KnowledgeConfig


class AgentSettings(BaseModel):
    """Per-turn native graph limits."""

    timezone: str = 'Asia/Shanghai'

    @field_validator('timezone')
    @classmethod
    def valid_timezone(cls, value):
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError('Use a valid IANA timezone') from None
        return value

    persona_file: Path | None = None
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, protected_namespaces=())
    model_call_limit: int = Field(default=20, gt=0, strict=True)
    tool_call_limit: int = Field(default=40, gt=0, strict=True)
    summary_call_limit: int = Field(default=10, gt=0, strict=True)
    knowledge_call_limit: int = Field(default=5, gt=0, strict=True)
    turn_timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)


class SessionsConfig(BaseModel):
    """Independent psycopg connection settings; never inferred from RAG."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    database_url: SecretStr

    @field_validator("database_url")
    @classmethod
    def postgres_url(cls, value):
        if not value.get_secret_value().startswith(("postgresql://", "postgres://")):
            raise ValueError("Use a PostgreSQL psycopg URL")
        return value


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    summary_model: str | None = Field(default=None, min_length=1)
    summary_tokens: int = Field(default=1500, ge=100, le=3000)
    tool_result_tokens: int = Field(default=4000, ge=100)
    mcp_schema_tokens: int = Field(default=6000, ge=100)


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    enabled: bool = False


class SkillsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    roots: list[str] = Field(default_factory=list)


from ai_assistant.integrations.config import MCPConfig, IntegrationsConfig, ChannelsConfig


class AppConfig(BaseModel):
    """Validated application settings; no clients or global state are stored."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    mcp: MCPConfig | None = None
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    skills: SkillsConfig | None = None
    sessions: SessionsConfig | None = None
    context: ContextConfig | None = None
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentSettings = Field(default_factory=AgentSettings)
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
        if self.knowledge and any(s.kind != 'local_directory' for s in self.knowledge.sources.values()):
            if self.integrations.feishu is None or self.mcp is None:
                raise ValueError('Cloud knowledge sources require explicit Feishu integration and token state configuration')
        if self.context and self.context.summary_model is not None and self.context.summary_model not in self.models:
            raise ValueError("Unknown summary model profile")
        if self.tools.web is not None:
            profile = self.models.get(self.tools.web.credential_profile)
            if profile is None or profile.provider != "deepseek" or profile.api_key is None:
                raise ValueError("Web tools require a DeepSeek credential profile")
        if self.mcp:
            for server in self.mcp.servers.values():
                if server.allow_write and self.sessions is None:
                    raise ValueError('External writes require persistent sessions')
                if server.transport in {'feishu', 'feishu_files'} and self.integrations.feishu is None:
                    raise ValueError('Feishu services require integrations.feishu')
        for group in self.channels.feishu.groups.values():
            for name in group.resources:
                if self.mcp is None or name not in self.mcp.servers:
                    raise ValueError('Group refers to an unknown MCP service')
                server = self.mcp.servers[name]
                if server.transport != 'feishu' or server.identity != 'bot':
                    raise ValueError('Groups may only grant official Feishu bot services')
        if self.channels.feishu.enabled and (self.sessions is None or self.integrations.feishu is None):
            raise ValueError('Feishu channel requires sessions and Feishu configuration')
        if self.memory.enabled and (self.sessions is None or self.integrations.feishu is None
                                    or self.knowledge is None or self.context is None):
            raise ValueError('Memory requires sessions, Feishu owner, knowledge embedding and context')
        if self.memory.enabled:
            from urllib.parse import urlsplit
            if urlsplit(self.knowledge.embedding.base_url).hostname not in {'localhost', '127.0.0.1', '::1'}:
                raise ValueError('Personal memory requires the configured local embedding service')
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
    def table(value):
        return value if isinstance(value, dict) else {}
    integrations = table(data.get('integrations'))
    servers = table(table(data.get('mcp')).get('servers'))
    if ('netease' in integrations
            or any(table(s).get('transport') == 'netease' for s in servers.values())
            or {'dependency_install', 'npm_project', 'dependency_timeout_seconds'} & table(data.get('skills')).keys()
            or 'profile' in table(table(data.get('tools')).get('terminal'))
            or {'embedding_model', 'reranker_model', 'device'} & table(data.get('knowledge')).keys()):
        raise ConfigError('配置包含已移除字段：请删除邮箱/Skills 安装/终端档位配置，并将知识模型改为 HTTP 服务配置。')
    try:
        config = AppConfig.model_validate(data)
        config._config_path = Path(path).absolute()
        return config
    except ValidationError as exc:
        # Field locations and raw exception text may themselves contain secrets.
        raise ConfigError(
            f"Invalid configuration ({exc.error_count()} validation errors); check profile fields, default_model and required api_key values."
        ) from None
