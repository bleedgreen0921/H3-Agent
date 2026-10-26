"""Small binary bridge; files stay in controlled workspace storage."""
import asyncio
import json
import os
from pathlib import PurePosixPath
import stat
import time
from urllib.parse import quote
from uuid import uuid4

import httpx
from ai_assistant.config import ConfigError
from ai_assistant.workspace.writes import WritableWorkspace

BASE = 'https://open.feishu.cn/open-apis'
MAX_BYTES = 25 * 1024**2


class FeishuRejected(ConfigError):
    def __init__(self, code):
        self.code = code
        super().__init__(f'飞书明确拒绝请求（code {code}）。')


class FeishuAPI:
    def __init__(self, settings, client, tokens=None):
        self.settings, self.client, self.tokens = settings, client, tokens
        self.cached = None
        self.lock = asyncio.Lock()

    async def access(self):
        if self.tokens:
            return await self.tokens.access()
        async with self.lock:
            if self.cached and self.cached[1] > time.time() + 90:
                return self.cached[0]
            response = await self.client.post(BASE + '/auth/v3/tenant_access_token/internal', json={
                'app_id': self.settings.app_id,
                'app_secret': self.settings.app_secret.get_secret_value()})
            response.raise_for_status()
            data = response.json()
            if data.get('code', 0) != 0:
                raise ConfigError('Feishu application authentication failed.')
            self.cached = data['tenant_access_token'], time.time() + data['expire']
            return self.cached[0]

    async def request(self, method, path, **kwargs):
        headers = {'Authorization': 'Bearer ' + await self.access()}
        response = await self.client.request(method, BASE + path, headers=headers, **kwargs)
        response.raise_for_status()
        data = response.json()
        if data.get('code', 0) != 0:
            raise FeishuRejected(data.get('code'))
        return data.get('data', data)

    async def send_text(self, chat_id, text, operation_id):
        return await self.request('POST', '/im/v1/messages', params={'receive_id_type': 'chat_id'},
            json={'receive_id': chat_id, 'msg_type': 'text',
                  'content': json.dumps({'text': text}, ensure_ascii=False), 'uuid': operation_id})

    async def send_card(self, chat_id, card, operation_id):
        return await self.request('POST', '/im/v1/messages', params={'receive_id_type': 'chat_id'},
            json={'receive_id': chat_id, 'msg_type': 'interactive',
                  'content': json.dumps(card, ensure_ascii=False), 'uuid': operation_id})

    async def update_card(self, message_id, card):
        return await self.request('PATCH', '/im/v1/messages/' + quote(message_id, safe=''),
                                  json={'content': json.dumps(card, ensure_ascii=False)})


class FileBridge:
    def __init__(self, api, workspace):
        self.api, self.workspace = api, workspace

    def read(self, path):
        with WritableWorkspace(self.workspace)._parent(path) as (parent, relative):
            fd = os.open(relative.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_BYTES:
                    raise ValueError('File must be an ordinary file no larger than 25 MiB.')
                raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('File exceeds 25 MiB.')
        return raw

    async def upload_drive(self, path: str, folder_token: str) -> dict:
        """Upload one controlled workspace file into an explicitly selected cloud folder."""
        raw = self.read(path)
        return await self.api.request('POST', '/drive/v1/files/upload_all',
            data={'file_name': PurePosixPath(path).name, 'parent_type': 'explorer',
                  'parent_node': folder_token, 'size': str(len(raw))},
            files={'file': (PurePosixPath(path).name, raw)})

    async def upload_message_file(self, path: str) -> dict:
        """Upload a workspace file for a subsequent explicit message send."""
        raw = self.read(path)
        return await self.api.request('POST', '/im/v1/files',
            data={'file_type': 'stream', 'file_name': PurePosixPath(path).name},
            files={'file': (PurePosixPath(path).name, raw)})

    async def download(self, path, filename):
        if PurePosixPath(filename).name != filename or '\\' in filename or filename in {'.', '..'}:
            raise ValueError('Unsafe filename.')
        output = f'outputs/feishu/{uuid4().hex}/{filename}'
        async with self.api.client.stream('GET', BASE + path,
            headers={'Authorization': 'Bearer ' + await self.api.access()}) as response:
            response.raise_for_status()
            json_body = None
            if 'application/json' in response.headers.get('content-type', ''):
                chunks, count = [], 0
                async for chunk in response.aiter_bytes():
                    count += len(chunk)
                    if count > MAX_BYTES:
                        raise ValueError('Download exceeds 25 MiB.')
                    chunks.append(chunk)
                json_body = b''.join(chunks)
                try:
                    envelope = json.loads(json_body)
                except (ValueError, UnicodeError):
                    envelope = None
                if isinstance(envelope, dict) and envelope.get('code', 0) != 0 and 'msg' in envelope:
                    raise ConfigError('Feishu rejected the file download.')
            size = 0
            with WritableWorkspace(self.workspace)._parent(output, create=True) as (parent, relative):
                fd = os.open(relative.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        if json_body is not None:
                            stream.write(json_body); size = len(json_body)
                        if json_body is None:
                            async for chunk in response.aiter_bytes():
                                size += len(chunk)
                                if size > MAX_BYTES:
                                    raise ValueError('Download exceeds 25 MiB.')
                                stream.write(chunk)
                except BaseException:
                    os.unlink(relative.name, dir_fd=parent)
                    raise
        return {'path': output, 'bytes': size}

    async def download_drive(self, file_token: str, filename: str) -> dict:
        """Download a cloud file into a new controlled output path, without redirects."""
        return await self.download('/drive/v1/files/' + quote(file_token, safe='') + '/download', filename)

    async def download_message_file(self, message_id: str, file_key: str, filename: str) -> dict:
        """Download an incoming message file into a controlled output path."""
        return await self.download('/im/v1/messages/' + quote(message_id, safe='') + '/resources/' +
                                   quote(file_key, safe='') + '?type=file', filename)


def create_server(api, workspace):
    from fastmcp import FastMCP
    bridge = FileBridge(api, workspace)
    server = FastMCP('feishu-files')
    for name in ['upload_drive', 'download_drive', 'upload_message_file', 'download_message_file']:
        server.tool(getattr(bridge, name), name=name)
    return server
