"""显式邮箱配置；不创建连接。"""
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class MailConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)
    enabled: bool = False
    address: str | None = None
    authorization_code: SecretStr | None = None
    timeout: float = Field(default=30, ge=1, le=120)
    page_size: int = Field(default=20, ge=1, le=100)
    body_chars: int = Field(default=16000, ge=100, le=16000)
    body_bytes: int = Field(default=262144, ge=1024, le=1048576)
    search_limit: int = Field(default=2000, ge=20, le=10000)

    @model_validator(mode='after')
    def credentials(self):
        if self.enabled and (not self.address or not self.address.endswith('@163.com')
                or not self.authorization_code or not self.authorization_code.get_secret_value().strip()):
            raise ValueError('启用邮箱需要 163 地址与授权码')
        return self

