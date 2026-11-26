"""Web 上传附件的显式配置；不读取飞书消息。"""
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_assistant.documents.formats import ATTACHMENT_EXTENSIONS


class AttachmentsConfig(BaseModel):
    """本地 Web 附件目录、类型和解析预算。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: bool = True
    state_dir: str = "~/.local/state/ai-assistant/attachments"
    max_bytes: int = Field(default=25 * 1024 * 1024, gt=0, strict=True)
    allowed_extensions: list[str] = Field(
        default_factory=lambda: sorted(ATTACHMENT_EXTENSIONS)
    )
    read_chars: int = Field(default=16000, ge=1000, le=100000, strict=True)
    parse_timeout: float = Field(default=60, gt=0, le=300, allow_inf_nan=False)
    retention_days: int = Field(default=30, ge=0, le=3650, strict=True)

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, values):
        result = []
        for value in values:
            item = value.strip().lower()
            if not item.startswith(".") or len(item) < 2:
                raise ValueError("附件扩展名必须以点开头。")
            result.append(item)
        if not result:
            raise ValueError("至少需要一个受支持的附件扩展名。")
        return list(dict.fromkeys(result))

    def root(self) -> Path:
        return Path(self.state_dir).expanduser().absolute()
