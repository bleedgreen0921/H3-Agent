"""Pure configuration declarations for explicitly enabled integrations."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Settings(BaseModel):
    """集成配置的共同约束：拒绝未知字段、禁止重新赋值并隐藏错误输入。"""
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)


class MCPServer(Settings):
    """单个 MCP 服务的连接方式、身份及工具权限配置。"""

    # 传输方式和身份必须显式匹配具体服务类型。
    transport: Literal['stdio', 'http', 'feishu', 'feishu_files']
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

    @model_validator(mode='after')
    def validate_transport(self):
        """在字段解析完成后校验传输参数及写工具配置的一致性。"""
        if self.transport == 'stdio' and not self.command:
            raise ValueError('stdio requires an explicit command')
        if self.transport == 'http' and (not self.url or not self.url.startswith(('http://', 'https://'))):
            raise ValueError('HTTP requires an http(s) URL')
        if self.transport in {'feishu', 'feishu_files'} and self.identity not in {'user', 'bot'}:
            raise ValueError('Feishu requires explicit user or bot identity')

        # 文件上传必须被标记为写操作，写工具也必须属于工具白名单。
        if self.transport == 'feishu_files' and (set(self.tools) & {'upload_drive', 'upload_message_file'}) - set(self.write_tools):
            raise ValueError('File uploads must be declared write_tools')
        if not set(self.write_tools) <= set(self.tools):
            raise ValueError('write_tools must be explicitly allowed tools')
        return self


class MCPConfig(Settings):
    """MCP 连接和调用超时、授权状态目录及具名服务。"""

    # 超时单位为秒，必须是有限正数。
    connect_timeout: float = Field(default=30, gt=0, allow_inf_nan=False)
    call_timeout: float = Field(default=120, gt=0, allow_inf_nan=False)
    state_dir: str = '~/.local/state/ai-assistant'
    servers: dict[str, MCPServer] = Field(default_factory=dict)


class FeishuConfig(Settings):
    """飞书应用、主人身份、用户授权和办公默认策略。"""

    # 任务负责人和成员策略；默认日历未配置时由调用流程处理。
    default_task_assignee: Literal['none', 'owner'] = 'none'
    task_members: Literal['model', 'owner_only'] = 'model'
    default_calendar_id: str | None = None

    # 应用与主人身份；SecretStr 避免密钥在常规展示中直接显示。
    app_id: str = Field(pattern=r'^cli_[A-Za-z0-9]+$')
    app_secret: SecretStr
    owner_open_id: str = Field(min_length=1)

    # 用户授权回调、请求的权限范围和 Node 接收器项目目录。
    redirect_uri: str = 'http://127.0.0.1:3000/callback'
    scopes: list[str] = Field(default_factory=lambda: ['offline_access'])
    node_project: str = 'deploy/feishu'


class IntegrationsConfig(Settings):
    """外部业务集成配置；未设置飞书时不自动启用该集成。"""
    feishu: FeishuConfig | None = None


class FeishuChannel(Settings):
    """飞书消息渠道的开关、投递限制和工作卡设置。"""

    # 消息投递的尝试次数上限、总超时和单次超时。
    delivery_max_attempts: int = Field(default=3, ge=1, le=100)
    final_delivery_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    delivery_attempt_timeout_seconds: float = Field(default=8, gt=0, allow_inf_nan=False)

    # 长任务工作卡：延迟出现、答案预览与终态保留；0 秒关闭。
    progress_card_delay_seconds: float = Field(default=1.5, ge=0, le=60, allow_inf_nan=False)
    progress_card_preview_chars: int = Field(default=4000, ge=0, le=20000, strict=True)
    enabled: bool = False


class ChannelsConfig(Settings):
    """渠道配置集合；飞书渠道默认关闭。"""
    feishu: FeishuChannel = Field(default_factory=FeishuChannel)
