"""Pure configuration declarations for explicitly enabled integrations."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)


class MCPServer(Settings):
    transport: Literal['stdio', 'http', 'feishu', 'feishu_files']
    identity: Literal['private', 'user', 'bot'] = 'private'
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, SecretStr] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    write_tools: list[str] = Field(default_factory=list)
    allow_write: bool = False
    resources: list[str] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)
    # Every group tool must declare the exact argument paths that select resources.
    resource_arguments: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_transport(self):
        if self.transport == 'stdio' and not self.command:
            raise ValueError('stdio requires an explicit command')
        if self.transport == 'http' and (not self.url or not self.url.startswith(('http://', 'https://'))):
            raise ValueError('HTTP requires an http(s) URL')
        if self.transport in {'feishu', 'feishu_files'} and self.identity not in {'user', 'bot'}:
            raise ValueError('Feishu requires explicit user or bot identity')
        if self.transport == 'feishu_files' and (set(self.tools) & {'upload_drive', 'upload_message_file'}) - set(self.write_tools):
            raise ValueError('File uploads must be declared write_tools')
        if not set(self.write_tools) <= set(self.tools):
            raise ValueError('write_tools must be explicitly allowed tools')
        return self


class MCPConfig(Settings):
    connect_timeout: float = Field(default=15, gt=0, allow_inf_nan=False)
    call_timeout: float = Field(default=30, gt=0, allow_inf_nan=False)
    state_dir: str = '~/.local/state/ai-assistant'
    servers: dict[str, MCPServer] = Field(default_factory=dict)


class FeishuConfig(Settings):
    default_task_assignee: Literal['none', 'owner'] = 'none'
    app_id: str = Field(pattern=r'^cli_[A-Za-z0-9]+$')
    app_secret: SecretStr
    owner_open_id: str = Field(min_length=1)
    redirect_uri: str = 'http://127.0.0.1:3000/callback'
    scopes: list[str] = Field(default_factory=lambda: ['offline_access'])
    node_project: str = 'deploy/feishu'


class IntegrationsConfig(Settings):
    feishu: FeishuConfig | None = None


class GroupConfig(Settings):
    resources: dict[str, list[str]] = Field(default_factory=dict)


class FeishuChannel(Settings):
    enabled: bool = False
    streaming: bool = False
    stream_interval_seconds: float = Field(default=1, ge=0.1, allow_inf_nan=False)
    card_max_bytes: int = Field(default=24576, ge=1024, le=24576)
    groups: dict[str, GroupConfig] = Field(default_factory=dict)


class ChannelsConfig(Settings):
    feishu: FeishuChannel = Field(default_factory=FeishuChannel)
