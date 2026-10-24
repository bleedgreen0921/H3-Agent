"""Local-only OAuth with private atomic storage and cross-process refresh locking."""
import asyncio
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import time
from urllib.parse import urlencode, urlsplit, parse_qs

import httpx
from ai_assistant.config import ConfigError

BASE = 'https://open.feishu.cn'


class TokenStore:
    def __init__(self, config, workspace):
        if config.mcp is None or config.integrations.feishu is None:
            raise ConfigError('Configure [mcp] and [integrations.feishu] first.')
        self.settings = config.integrations.feishu
        self.root = Path(config.mcp.state_dir).expanduser().absolute()
        if self.root.resolve().is_relative_to(workspace.resolve()):
            raise ConfigError('OAuth state directory must be outside the workspace.')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.stat().st_uid != os.getuid():
            raise ConfigError('OAuth directory must be owned by the current user and not a symlink.')
        os.chmod(self.root, 0o700)
        self.path = self.root / ('feishu-' + self.settings.app_id + '.json')
        self.async_lock = asyncio.Lock()

    @contextmanager
    def lock(self):
        fd = os.open(self.root / 'oauth.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            # Blocking flock runs only in the worker thread, including HTTP refresh.
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def read(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise ConfigError('Run mcp auth feishu login on the local CLI.') from None
        with os.fdopen(fd) as stream:
            if os.fstat(stream.fileno()).st_mode & 0o077:
                raise ConfigError('OAuth token file must have mode 0600.')
            return json.load(stream)

    def save(self, value):
        path = self.root / ('.token-' + secrets.token_hex(16))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(value, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(path, self.path)
        finally:
            path.unlink(missing_ok=True)

    def request_token(self, grant):
        payload = dict(client_id=self.settings.app_id,
            client_secret=self.settings.app_secret.get_secret_value(), **grant)
        with httpx.Client(timeout=30, trust_env=False) as client:
            response = client.post(BASE + '/open-apis/authen/v2/oauth/token', json=payload)
            response.raise_for_status()
            data = response.json()
            if data.get('code', 0) != 0 or not data.get('access_token'):
                raise ConfigError('Feishu OAuth failed; check application scopes and re-login.')
            identity = client.get(BASE + '/open-apis/authen/v1/user_info',
                headers={'Authorization': 'Bearer ' + data['access_token']})
            identity.raise_for_status()
            if identity.json().get('data', {}).get('open_id') != self.settings.owner_open_id:
                raise ConfigError('OAuth user is not the configured owner.')
        return dict(access_token=data['access_token'], refresh_token=data.get('refresh_token'),
                    expires_at=time.time() + data['expires_in'],
                    refresh_expires_at=time.time() + data.get('refresh_token_expires_in', 0),
                    scope=data.get('scope', ''), owner_open_id=self.settings.owner_open_id,
                    status='ready')

    def _access(self):
        with self.lock():
            data = self.read()
            if data.get('status') != 'ready':
                raise ConfigError('OAuth refresh result is unknown; log in again.')
            if data.get('owner_open_id') != self.settings.owner_open_id:
                raise ConfigError('Token owner differs from configuration; log in again.')
            if data['expires_at'] > time.time() + 90:
                return data['access_token']
            if not data.get('refresh_token') or data.get('refresh_expires_at', 0) <= time.time():
                raise ConfigError('OAuth session expired; log in again.')
            refresh = data['refresh_token']
            # Persist uncertainty BEFORE sending a rotating refresh token.
            self.save({**data, 'status': 'refresh_unknown'})
            updated = self.request_token(dict(grant_type='refresh_token', refresh_token=refresh))
            if not updated.get('refresh_token'):
                raise ConfigError('OAuth refresh did not return a replacement refresh token; log in again.')
            self.save(updated)
            return updated['access_token']

    async def access(self):
        async with self.async_lock:
            worker = asyncio.create_task(asyncio.to_thread(self._access))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                await worker
                raise

    async def login(self):
        url = urlsplit(self.settings.redirect_uri)
        if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost'} or not url.port:
            raise ConfigError('OAuth callback must use an explicit loopback HTTP port.')
        state = secrets.token_urlsafe(32)
        future = asyncio.get_running_loop().create_future()

        async def callback(reader, writer):
            try:
                first = await asyncio.wait_for(reader.readline(), 5)
                parts = first.decode('ascii').split(' ')
                target = urlsplit(parts[1]) if len(parts) >= 2 else urlsplit('')
                args = parse_qs(target.query)
                valid = (parts[0] == 'GET' and target.path == url.path and
                         secrets.compare_digest(args.get('state', [''])[0], state) and
                         len(args.get('code', [])) == 1)
                if valid and not future.done():
                    future.set_result(args['code'][0])
                writer.write(b'HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nReturn to your local terminal.' if valid else
                             b'HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\nInvalid callback.')
                await writer.drain()
            finally:
                writer.close(); await writer.wait_closed()
        server = await asyncio.start_server(callback, '127.0.0.1', url.port)
        async with server:
            print(BASE + '/open-apis/authen/v1/authorize?' + urlencode(dict(
                client_id=self.settings.app_id, redirect_uri=self.settings.redirect_uri,
                response_type='code', scope=' '.join(self.settings.scopes), state=state)), flush=True)
            code = await asyncio.wait_for(future, 180)
        def exchange():
            with self.lock():
                data = self.request_token(dict(grant_type='authorization_code', code=code,
                                               redirect_uri=self.settings.redirect_uri))
                self.save(data)
        await asyncio.to_thread(exchange)

    def status(self):
        with self.lock():
            try:
                data = self.read()
            except ConfigError:
                return {'logged_in': False}
            return {key: data.get(key) for key in ['status', 'owner_open_id', 'expires_at', 'refresh_expires_at', 'scope']}

    def logout(self):
        with self.lock():
            self.path.unlink(missing_ok=True)
        return {'logged_in': False, 'remote_revocation': 'Revoke application authorization in Feishu to invalidate remote tokens.'}
