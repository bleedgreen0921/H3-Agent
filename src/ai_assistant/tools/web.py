"""Small asynchronous adapter for DeepSeek's standalone Search/Fetch API."""

import ipaddress
import json
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError


class WebError(ValueError):
    """A provider failure with a safe public message."""


class SearchItem(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)
    title: str
    url: str
    snippet: str
    summary: str
    content: str
    site_name: str | None = None
    date: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchItem]


class FetchResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    title: str
    content: str


def validate_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        _ = parsed.port
        if (len(url) > 8192 or parsed.scheme not in ('http', 'https') or not host
                or parsed.username is not None or parsed.password is not None
                or any(c.isspace() or ord(c) < 32 for c in url) or '\\' in url):
            raise ValueError
        host = host.rstrip('.').lower()
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if ('.' not in host or host.endswith(('.localhost', '.local', '.internal'))
                    or host == 'localhost' or all(c.isdigit() or c == '.' for c in host)
                    or host.startswith('0x')):
                raise ValueError
        else:
            if not address.is_global:
                raise ValueError
    except ValueError:
        raise WebError('仅支持无凭据的公网 HTTP/HTTPS 网页地址。') from None
    return url


class DeepSeekWeb:
    """No retry, local target fetch, or model invocation. Client ownership is external."""

    def __init__(self, client: httpx.AsyncClient, api_key: str, timeout: float = 30):
        self._client = client
        self._api_key = api_key
        self._timeout = timeout

    async def _post(self, endpoint: str, payload: dict) -> dict:
        try:
            async with self._client.stream(
                'POST', f'https://api.deepseek.com/v1/tools/{endpoint}',
                headers={'Authorization': f'Bearer {self._api_key}'},
                json=payload, timeout=self._timeout, follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    messages = {400: '联网请求参数无效。', 401: '联网服务认证失败。',
                                402: '联网服务余额不足。', 403: '网页请求被安全策略拒绝。',
                                404: '网页内容不可获取。', 429: '联网服务限流。'}
                    raise WebError(messages.get(response.status_code, '联网服务请求失败。'))
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(chunks) + len(chunk) > 2 * 1024 * 1024:
                        raise WebError('联网响应超过 2 MiB 限制。')
                    chunks.extend(chunk)
            data = json.loads(chunks)
            if not isinstance(data, dict):
                raise ValueError
            return data
        except httpx.TimeoutException:
            raise WebError('联网服务请求超时。') from None
        except httpx.RequestError:
            raise WebError('联网服务连接失败。') from None
        except (ValueError, UnicodeError) as exc:
            if isinstance(exc, WebError):
                raise
            raise WebError('联网服务响应格式无效。') from None

    async def search(self, query: str, count=5) -> tuple[str, dict]:
        if not query.strip() or not 1 <= len(query) <= 128 or not 1 <= count <= 10:
            raise WebError('搜索词或结果数量无效。')
        data = await self._post('search', {'query': query, 'count': count, 'need_content': False})
        try:
            response = SearchResponse.model_validate(data)
        except ValidationError:
            raise WebError('联网服务响应格式无效。') from None
        content = '\n\n'.join(
            f'{item.title}\n{item.url}\n{item.summary or item.snippet}'
            + (f'\n来源：{item.site_name}' if item.site_name else '')
            + (f'\n日期：{item.date}' if item.date else '')
            for item in response.results
        ) or '没有找到匹配结果。'
        artifact = {'response': data, 'truncated': len(content) > 20000}
        return content[:20000], artifact

    async def fetch(self, url: str) -> tuple[str, dict]:
        data = await self._post('fetch', {'url': validate_url(url)})
        try:
            page = FetchResponse.model_validate(data)
            if not page.content.strip():
                raise ValueError
        except (ValidationError, ValueError):
            raise WebError('联网服务响应格式无效或网页正文为空。') from None
        content = f'{page.title}\n{url}\n\n{page.content}'
        return content[:30000], {'url': url, 'response': data, 'truncated': len(content) > 30000}
