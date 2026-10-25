import asyncio
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from ai_assistant.config import ConfigError
from ai_assistant.integrations.oauth import TokenStore
from tests.integrations.test_mcp import config


def tokens(tmp_path, **feishu):
    workspace = tmp_path / 'workspace'; workspace.mkdir(exist_ok=True)
    c = config(workspace, mcp={'state_dir':str(tmp_path/'private')},integrations={'feishu':{
        'app_id':'cli_fixture','app_secret':'application-secret','owner_open_id':'owner',**feishu}})
    return TokenStore(c, workspace)


def data(**changes):
    return dict(access_token='old',refresh_token='refresh-old',expires_at=0,
        refresh_expires_at=time.time()+3600,status='ready',owner_open_id='owner',**changes)


def test_refresh_rotation_concurrent_and_private_modes(tmp_path):
    first = tokens(tmp_path); second = tokens(tmp_path)
    first.save(data())
    def request(grant):
        assert grant['refresh_token'] == 'refresh-old'
        return {**data(), 'access_token':'new','refresh_token':'refresh-new','expires_at':time.time()+3600}
    async def run():
        with patch.object(TokenStore,'request_token',side_effect=request) as exchange:
            assert await asyncio.gather(first.access(),second.access()) == ['new','new']
            assert exchange.call_count == 1
    asyncio.run(run())
    assert first.read()['refresh_token'] == 'refresh-new'
    assert first.root.stat().st_mode & 0o777 == 0o700
    assert first.path.stat().st_mode & 0o777 == 0o600


def test_unknown_refresh_never_reuses_old_refresh_token(tmp_path):
    store = tokens(tmp_path); store.save(data())
    async def run():
        with patch.object(store,'request_token',side_effect=httpx.ReadTimeout('unknown')) as exchange:
            with pytest.raises(httpx.ReadTimeout):
                await store.access()
            with pytest.raises(ConfigError,match='unknown'):
                await store.access()
            assert exchange.call_count == 1
    asyncio.run(run())
    assert store.status()['status'] == 'refresh_unknown'
    assert 'access_token' not in store.status()
    store.logout(); assert store.status() == {'logged_in':False}


def test_owner_validation_and_token_errors(tmp_path):
    store = tokens(tmp_path)
    def mock(request):
        if request.method == 'POST':
            return httpx.Response(200,json={'access_token':'new','expires_in':3600,'refresh_token':'rotated'})
        return httpx.Response(200,json={'data':{'open_id':'someone-else'}})
    original = httpx.Client
    with patch('ai_assistant.integrations.oauth.httpx.Client', side_effect=lambda **kw: original(transport=httpx.MockTransport(mock))):
        with pytest.raises(ConfigError,match='owner'):
            store.request_token({'grant_type':'authorization_code','code':'code'})
    assert not store.path.exists()


def test_oauth_callback_rejects_wrong_state_then_accepts_owner(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    store = tokens(tmp_path,redirect_uri=f'http://127.0.0.1:{port}/callback')
    output = io.StringIO()
    async def run():
        with redirect_stdout(output), patch.object(store,'request_token',return_value={**data(),'expires_at':time.time()+3600}):
            login = asyncio.create_task(store.login())
            for _ in range(100):
                if output.getvalue(): break
                await asyncio.sleep(.01)
            state = parse_qs(urlsplit(output.getvalue().strip()).query)['state'][0]
            async with httpx.AsyncClient(trust_env=False) as client:
                wrong = await client.get(f'http://127.0.0.1:{port}/callback',params={'state':'wrong','code':'code'})
                assert wrong.status_code == 400 and not login.done()
                right = await client.get(f'http://127.0.0.1:{port}/callback',params={'state':state,'code':'code'})
                assert right.status_code == 200
            await login
    asyncio.run(run())
    assert store.path.exists()


def test_state_cannot_be_inside_workspace(tmp_path):
    c = config(tmp_path,mcp={'state_dir':str(tmp_path/'state')},integrations={'feishu':{
        'app_id':'cli_fixture','app_secret':'secret','owner_open_id':'owner'}})
    with pytest.raises(ConfigError,match='outside'):
        TokenStore(c,tmp_path)


def test_missing_user_info_is_not_reported_as_owner_mismatch(tmp_path):
    store = tokens(tmp_path)
    def mock(request):
        if request.method == 'POST':
            return httpx.Response(200,json={'access_token':'new','expires_in':3600})
        return httpx.Response(200,json={'code':99991672,'msg':'PRIVATE'})
    original = httpx.Client
    with patch('ai_assistant.integrations.oauth.httpx.Client', side_effect=lambda **kw: original(transport=httpx.MockTransport(mock))):
        with pytest.raises(ConfigError,match='user information') as error:
            store.request_token({'grant_type':'authorization_code','code':'code'})
    assert 'PRIVATE' not in str(error.value) and not store.path.exists()
    assert not list(store.root.glob('feishu-owner-check-*'))


def test_owner_mismatch_is_private_and_never_replaces_trusted_owner(tmp_path):
    store = tokens(tmp_path)
    def mock(request):
        if request.method == 'POST':
            return httpx.Response(200,json={'access_token':'SECRET_TOKEN','expires_in':3600})
        return httpx.Response(200,json={'data':{'open_id':'different-owner'}})
    original = httpx.Client
    with patch('ai_assistant.integrations.oauth.httpx.Client', side_effect=lambda **kw: original(transport=httpx.MockTransport(mock))):
        with pytest.raises(ConfigError) as error:
            store.request_token({'grant_type':'authorization_code','code':'code'})
    assert 'different-owner' not in str(error.value)
    path = next(store.root.glob('feishu-owner-check-*'))
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())['authorized_open_id'] == 'different-owner'
    assert 'SECRET_TOKEN' not in path.read_text()
    assert store.settings.owner_open_id == 'owner' and not store.path.exists()
