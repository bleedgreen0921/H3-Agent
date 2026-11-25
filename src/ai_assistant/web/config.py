"""本地 Web 入口配置。"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class WebServerConfig(BaseModel):
    """单用户本地 Web 服务设置；默认只监听回环地址。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8765, ge=1, le=65535, strict=True)
    state_dir: str = "~/.local/state/ai-assistant"
    token_file: str = "~/.local/state/ai-assistant/web-token"
    page_size: int = Field(default=50, ge=1, le=200, strict=True)

    def state_root(self) -> Path:
        return Path(self.state_dir).expanduser().absolute()

    def token_path(self) -> Path:
        return Path(self.token_file).expanduser().absolute()
