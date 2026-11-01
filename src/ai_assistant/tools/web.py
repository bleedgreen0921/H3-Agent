"""Small asynchronous adapter for DeepSeek's standalone Search/Fetch API."""

import ipaddress
import json
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

# Keep one search + fetch exchange usable within the default 24k input budget.
# The full bounded provider response remains in the artifact, outside model text.
SEARCH_TEXT_LIMIT = 16000
FETCH_TEXT_LIMIT = 32000


def bounded_text(content, limit):
    marker = '\n[内容已截断；仅展示部分结果]'
    return content if len(content) <= limit else content[:limit - len(marker)] + marker


class WebError(ValueError):
    """A provider failure with a safe public message."""


class WebLimitError(WebError):
    """服务额度或限流要求本轮直接停止。"""


class SearchItem(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)
    title: str | None = None
    url: str
    snippet: str | None = None
    summary: str | None = None
    content: str | None = None
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


def page_is_unavailable(title, content):
    markers = ('access denied', 'not available in your region', 'app unavailable in region',
               '地区限制', '请先登录', 'sign in to continue')
    heading = content.strip().splitlines()[0].strip('# *').lower() if content.strip() else ''
    title = title.strip().lower()
    return any(title == marker or title.startswith(marker + ' |')
               or heading == marker or heading.startswith(marker + '。')
               or heading.startswith(marker + '\n') for marker in markers)


class DeepSeekWeb:
    """No retry, local target fetch, or model invocation. Client ownership is external."""

    def __init__(self, client: httpx.AsyncClient, api_key: str, timeout: float = 60, *,
                 search_chars=SEARCH_TEXT_LIMIT, fetch_chars=FETCH_TEXT_LIMIT):
        self.search_chars, self.fetch_chars = search_chars, fetch_chars
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
                    error = WebLimitError if response.status_code in {402, 429} else WebError
                    raise error(messages.get(response.status_code, '联网服务请求失败。'))
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
            f'{item.title or item.url}\n{validate_url(item.url)}\n{item.summary or item.snippet or item.content or ""}'
            + (f'\n来源：{item.site_name}' if item.site_name else '')
            + (f'\n日期：{item.date}' if item.date else '')
            for item in response.results
        ) or '没有找到匹配结果。'
        artifact = {'response': data, 'full_tool_content': content, 'evidence_kind': 'search_summary',
                    'truncated': len(content) > self.search_chars}
        return bounded_text(content, self.search_chars), artifact

    async def fetch(self, url: str) -> tuple[str, dict]:
        data = await self._post('fetch', {'url': validate_url(url)})
        try:
            page = FetchResponse.model_validate(data)
            if not page.content.strip():
                raise ValueError
        except (ValidationError, ValueError):
            raise WebError('联网服务响应格式无效或网页正文为空。') from None
        content = f'{page.title}\n{url}\n\n{page.content}'
        # 只依据页面标题和主标题识别，不把正文中的一般提及误判为限制页。
        unavailable = page_is_unavailable(page.title, page.content)
        if unavailable:
            content = f'{url}\n正文不可用：页面要求登录、限制地区或拒绝访问；不能作为正文证据。'
        return bounded_text(content, self.fetch_chars), {
            'url': url, 'response': data, 'full_tool_content': content,
            'evidence_kind': 'unavailable_page' if unavailable else 'page_content',
            'target_available': not unavailable, 'truncated': len(content) > self.fetch_chars}
