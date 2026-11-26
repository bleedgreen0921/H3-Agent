"""Pure configuration declarations for explicitly enabled integrations."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Settings(BaseModel):
    """集成配置的共同约束：拒绝未知字段、禁止重新赋值并隐藏错误输入。"""

    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)


class MCPServer(Settings):
    """单个 MCP 服务的连接方式、身份及工具权限配置。"""

    # 只保留通用 MCP；飞书官方 MCP 已由只读文档插件替代。
    transport: Literal['stdio', 'http']
    identity: Literal['private', 'user', 'bot'] = 'private'

    # stdio 使用命令和环境；HTTP 使用地址和请求头。
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, SecretStr] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, SecretStr] = Field(default_factory=dict)

    # tools 是允许范围；write_tools 标记其中的写能力，allow_write 控制写开关。
    tools: list[str] = Field(default_factory=list)
    write_tools: list[str] = Field(default_factory=list)
    allow_write: bool = False

    # 规划任务使用的显式资源边界；不从工具名或描述推导。
    resource_tools: dict[str, Literal['document', 'file', 'record']] = Field(
        default_factory=dict)
    append_tools: list[str] = Field(default_factory=list)
    create_tools: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_transport(self):
        """在字段解析完成后校验传输参数及写工具配置的一致性。"""
        if self.transport == 'stdio' and not self.command:
            raise ValueError('stdio requires an explicit command')
        if self.transport == 'http' and (not self.url or not self.url.startswith(('http://', 'https://'))):
            raise ValueError('HTTP requires an http(s) URL')
        if not set(self.write_tools) <= set(self.tools):
            raise ValueError('write_tools must be explicitly allowed tools')
        if not set(self.resource_tools) <= set(self.tools):
            raise ValueError('resource_tools must be explicitly allowed tools')
        if not set(self.append_tools) <= set(self.write_tools):
            raise ValueError('append_tools must be explicitly declared write tools')
        if not set(self.create_tools) <= set(self.write_tools):
            raise ValueError('create_tools must be explicitly declared write tools')
        return self


class MCPConfig(Settings):
    """MCP 连接和调用超时、授权状态目录及具名服务。"""

    connect_timeout: float = Field(default=30, gt=0, allow_inf_nan=False)
    call_timeout: float = Field(default=120, gt=0, allow_inf_nan=False)
    state_dir: str = '~/.local/state/ai-assistant'
    servers: dict[str, MCPServer] = Field(default_factory=dict)


class FeishuConfig(Settings):
    """飞书只读文档连接器所需的应用凭据和用户授权。"""

    app_id: str = Field(pattern=r'^cli_[A-Za-z0-9]+$')
    app_secret: SecretStr
    redirect_uri: str = 'http://127.0.0.1:3000/callback'
    scopes: list[str] = Field(default_factory=lambda: ['offline_access'])
    state_dir: str = '~/.local/state/ai-assistant'
    default_identity: Literal['user', 'bot'] = 'user'


class IntegrationsConfig(Settings):
    """外部业务集成配置；未设置飞书时不自动启用该集成。"""
    feishu: FeishuConfig | None = None
