"""具名组合配置校验；不修改进程环境。"""

from pathlib import Path
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
    search_chars: int = Field(default=16000, ge=100, strict=True)
    fetch_chars: int = Field(default=32000, ge=100, strict=True)
    credential_profile: str = Field(min_length=1)
    timeout: float = Field(default=60, gt=0, allow_inf_nan=False)


class FilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    enabled: bool = False
    max_file_bytes: int = Field(default=8 * 1024 * 1024, gt=0, strict=True)
    result_chars: int = Field(default=48000, ge=100, strict=True)
    scan_entries: int = Field(default=20000, gt=0, strict=True)
    scan_bytes: int = Field(default=128 * 1024 * 1024, gt=0, strict=True)


class PDFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    enabled: bool = False
    timeout: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)


class ToolSelectionConfig(BaseModel):
    """按轮暴露的工具选择；权限白名单仍由 mcp.servers 控制。"""
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    mode: Literal['retrieval', 'all'] = 'retrieval'
    top_k: int = Field(default=8, ge=1, le=64, strict=True)
    max_schema_tokens: int = Field(default=25000, ge=0, le=200000, strict=True)
    core: list[str] = Field(default_factory=list)
    max_search_calls: int = Field(default=3, ge=0, le=20, strict=True)
    log_path: str | None = None


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    web: WebConfig | None = None
    files: FilesConfig = Field(default_factory=FilesConfig)
    selection: ToolSelectionConfig = Field(default_factory=ToolSelectionConfig)


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
    model_call_limit: int = Field(default=40, gt=0, strict=True)
    tool_call_limit: int = Field(default=100, gt=0, strict=True)
    summary_call_limit: int = Field(default=20, gt=0, strict=True)
    knowledge_call_limit: int = Field(default=15, gt=0, strict=True)
    turn_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)


class SessionsConfig(BaseModel):
    """Independent psycopg connection settings; never inferred from RAG."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    backend: str = Field(default="postgresql", min_length=1)
    database_url: SecretStr
    open_lock_timeout_seconds: float = Field(default=2.0, ge=0, le=30)

    @field_validator('backend')
    @classmethod
    def supported_backend(cls, value):
        from ai_assistant.storage.backends import require_backend
        require_backend('sessions', value)
        return value

    @field_validator("database_url")
    @classmethod
    def postgres_url(cls, value):
        if not value.get_secret_value().startswith(("postgresql://", "postgres://")):
            raise ValueError("Use a PostgreSQL psycopg URL")
        return value


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    memory_every_turns: int = Field(default=5, ge=0, strict=True)
    memory_tokens: int = Field(default=20000, ge=100, strict=True)
    compression_summary_ratio: float = Field(default=0.1, gt=0, lt=1, strict=True)
    compression_recent_turns: int = Field(default=30, gt=0, strict=True)
    tool_result_tokens: int = Field(default=12000, ge=100)


class SkillsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    directory_tokens: int = Field(default=6000, ge=100, strict=True)
    roots: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)


from ai_assistant.integrations.config import MCPConfig, IntegrationsConfig, ChannelsConfig
from ai_assistant.integrations.mail_config import MailConfig


from ai_assistant.observability import ObservabilityConfig


class AppConfig(BaseModel):
    """Validated application settings; no clients or global state are stored."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    mail: MailConfig = Field(default_factory=MailConfig)
    mcp: MCPConfig | None = None
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    skills: SkillsConfig | None = None
    sessions: SessionsConfig | None = None
    context: ContextConfig | None = None
    agent: AgentSettings = Field(default_factory=AgentSettings)
    knowledge: KnowledgeConfig | None = None
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    _config_path: Path | None = PrivateAttr(default=None)
    _config_files: tuple[Path, ...] = PrivateAttr(default=())
    _profile: str | None = PrivateAttr(default=None)

    @property
    def config_files(self) -> tuple[Path, ...]:
        return self._config_files

    @property
    def profile(self) -> str | None:
        return self._profile


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
        if self.tools.pdf.enabled and self.integrations.feishu is None:
            raise ValueError('PDF attachments require integrations.feishu')
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
        if self.channels.feishu.enabled and (self.sessions is None or self.integrations.feishu is None):
            raise ValueError('Feishu channel requires sessions and Feishu configuration')
        return self


def load_config(path: str | Path = "assistant.toml", *, profile: str | None = None) -> AppConfig:
    """读取显式组合；相对应用路径始终以入口目录为基准。"""
    from ai_assistant.configuration import resolve_config
    data, sources, selected_profile = resolve_config(path, profile)
    # 在通用 schema 校验前给未实现后端明确摘要，不导入任何数据库客户端。
    from ai_assistant.storage.backends import require_backend
    for section, kind in [('sessions', 'sessions'), ('knowledge', 'knowledge')]:
        settings = data.get(section)
        if isinstance(settings, dict) and isinstance(settings.get('backend'), str):
            try:
                require_backend(kind, settings['backend'])
            except ValueError:
                raise ConfigError('该存储后端尚未实现；未创建连接或回退。') from None
    if 'plan' in data:
        raise ConfigError('未知配置 [plan]：Plan 模式已移除，请删除该配置节。')
    mcp = data.get('mcp')
    servers = mcp.get('servers') if isinstance(mcp, dict) else None
    if isinstance(servers, dict):
        for server in servers.values():
            if isinstance(server, dict) and (server.get('resources') or server.get('prompts')):
                raise ConfigError('MCP resources/prompts 内容接口不再支持，请删除非空配置。')
    try:
        config = AppConfig.model_validate(data)
        config._config_path = Path(path).absolute()
        config._config_files = sources
        config._profile = selected_profile
        return config
    except ValidationError as exc:
        # Field locations and raw exception text may themselves contain secrets.
        raise ConfigError(
            f"Invalid configuration ({exc.error_count()} validation errors); check profile fields, default_model and required api_key values."
        ) from None
