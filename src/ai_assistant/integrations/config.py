"""Pure configuration declarations for explicitly enabled integrations."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)


class MCPServer(Settings):
    transport: Literal['stdio', 'http', 'netease', 'feishu', 'feishu_files']
    identity: Literal['private', 'user', 'bot'] = 'private'
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, SecretStr] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    account: str | None = None
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
        if self.transport == 'netease' and not self.account:
            raise ValueError('netease requires an account')
        if self.transport in {'feishu', 'feishu_files'} and self.identity not in {'user', 'bot'}:
            raise ValueError('Feishu requires explicit user or bot identity')
        if self.transport == 'netease' and self.identity != 'private':
            raise ValueError('Mailboxes are private services')
        if self.transport == 'netease' and (set(self.tools) & {'send', 'draft', 'mark', 'move', 'update_draft'}) - set(self.write_tools):
            raise ValueError('Mailbox mutations must be declared write_tools')
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


class MailAccount(Settings):
    address: str
    authorization_code: SecretStr
    imap_host: str | None = None
    smtp_host: str | None = None
    imap_port: int = Field(default=993, ge=1, le=65535)
    smtp_port: int = Field(default=465, ge=1, le=65535)
    folders: dict[str, str] = Field(default_factory=dict)
    attachment_limit: int = Field(default=25 * 1024**2, gt=0, le=25 * 1024**2)
    message_limit: int = Field(default=50 * 1024**2, gt=0, le=50 * 1024**2)

    @model_validator(mode='after')
    def personal_domain(self):
        if self.address.count('@') != 1 or self.address.rsplit('@', 1)[1].lower() not in {'163.com', '126.com', 'yeah.net'}:
            raise ValueError('A personal NetEase mailbox is required')
        if not self.authorization_code.get_secret_value().strip():
            raise ValueError('Client authorization code is required')
        return self


class NetEaseConfig(Settings):
    accounts: dict[str, MailAccount] = Field(default_factory=dict)


class FeishuConfig(Settings):
    app_id: str = Field(pattern=r'^cli_[A-Za-z0-9]+$')
    app_secret: SecretStr
    owner_open_id: str = Field(min_length=1)
    redirect_uri: str = 'http://127.0.0.1:3000/callback'
    scopes: list[str] = Field(default_factory=lambda: ['offline_access'])
    node_project: str = 'deploy/feishu'


class IntegrationsConfig(Settings):
    netease: NetEaseConfig | None = None
    feishu: FeishuConfig | None = None


class GroupConfig(Settings):
    resources: dict[str, list[str]] = Field(default_factory=dict)


class FeishuChannel(Settings):
    enabled: bool = False
    groups: dict[str, GroupConfig] = Field(default_factory=dict)


class ChannelsConfig(Settings):
    feishu: FeishuChannel = Field(default_factory=FeishuChannel)
