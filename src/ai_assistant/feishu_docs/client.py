"""飞书应用身份 HTTP 客户端；只用于只读文档请求。"""
import asyncio
import time

from ai_assistant.config import ConfigError

BASE = "https://open.feishu.cn/open-apis"


class FeishuRejected(ConfigError):
    def __init__(self, code, *, http_status=None):
        self.code = code if isinstance(code, int) else None
        self.http_status = http_status
        super().__init__(f"飞书明确拒绝请求（code {self.code}）。")


class FeishuAPI:
    """Tenant token 的最小缓存客户端；不包含消息、任务或文件写入。"""

    def __init__(self, settings, client):
        self.settings, self.client = settings, client
        self.cached = None
        self.lock = asyncio.Lock()

    async def access(self):
        async with self.lock:
            if self.cached and self.cached[1] > time.time() + 90:
                return self.cached[0]
            response = await self.client.post(
                BASE + "/auth/v3/tenant_access_token/internal",
                json={"app_id": self.settings.app_id,
                      "app_secret": self.settings.app_secret.get_secret_value()})
            response.raise_for_status()
            data = response.json()
            if data.get("code", 0) != 0:
                raise ConfigError("Feishu application authentication failed.")
            self.cached = data["tenant_access_token"], time.time() + data["expire"]
            return self.cached[0]

    async def request(self, method, path, **kwargs):
        headers = {"Authorization": "Bearer " + await self.access()}
        response = await self.client.request(method, BASE + path, headers=headers, **kwargs)
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            return {}
        if isinstance(data, dict) and data.get("code", 0) != 0:
            raise FeishuRejected(data.get("code"), http_status=response.status_code)
        response.raise_for_status()
        return data.get("data", data) if isinstance(data, dict) else data
