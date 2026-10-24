import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from pydantic import ValidationError
from langchain_core.tools import ToolException
from ai_assistant.integrations.config import MCPServer, MailAccount
from ai_assistant.integrations.policy import Caller, authorize
from ai_assistant.integrations.mcp import Catalog, mcp_resources, stable_id
from ai_assistant.config import AppConfig
from ai_assistant.workspace.files import ReadOnlyWorkspace


def config(tmp_path, **extra):
    return AppConfig(default_model='test', models={'test': {'provider': 'ollama', 'model': 'test'}},
        workspace={'root': str(tmp_path)}, **extra)


def test_disabled_opens_nothing(tmp_path):
    async def run():
        async with mcp_resources(config(tmp_path), ReadOnlyWorkspace(tmp_path)) as directory:
            assert directory.entries == {}
    asyncio.run(run())


def test_group_permissions():
    server = MCPServer(transport='feishu', identity='bot',
        tools=['read', 'write'], write_tools=['write'], resource_arguments={'read': ['path.id']})
    caller = Caller(channel='feishu_group', resources={'office': ('allowed',)})
    authorize(caller, 'office', server, 'read', {'path': {'id': 'allowed'}})
    for tool, args in [('read', {'path': {'id': 'other'}}), ('write', {}), ('unknown', {})]:
        with pytest.raises(ToolException):
            authorize(caller, 'office', server, tool, args)
    with pytest.raises(ToolException):
        authorize(caller, 'office', server.model_copy(update={'identity': 'user'}), 'read', {})


def test_qualified_ids_do_not_collide():
    assert stable_id('a', 'user', 'read') != stable_id('a', 'bot', 'read')
    assert stable_id('a.b', 'user', 'read') != stable_id('a_b', 'user', 'read')


def test_native_fixture(tmp_path):
    pytest.importorskip('fastmcp')
    from fastmcp import FastMCP, Client
    from langchain.mcp import MCPAdapter
    from ai_assistant.integrations.mcp import decline
    async def run():
        server = FastMCP('fixture')
        @server.tool
        async def echo(value: str) -> str:
            return value
        async with MCPAdapter(Client(server, elicitation_handler=decline)) as adapter:
            tools = await adapter.list_tools()
            assert tools[0].name == 'echo'
            result = await tools[0].ainvoke({'value': 'local fixture'})
            assert 'local fixture' in str(result)
    asyncio.run(run())


def test_netease_server_schema(tmp_path):
    pytest.importorskip('fastmcp')
    from ai_assistant.integrations.netease import create_server
    from fastmcp import Client
    async def run():
        account = MailAccount(address='test@163.com', authorization_code='secret')
        async with Client(create_server('test', account, ReadOnlyWorkspace(tmp_path))) as client:
            tools = await client.list_tools()
            assert {t.name for t in tools} == {'folders','search','read','download','send','draft','update_draft','mark','move'}
    asyncio.run(run())
