import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from ai_assistant.config import ConfigError
from ai_assistant.integrations.config import MCPServer
from ai_assistant.integrations.feishu import capabilities, allowed_capabilities, OfficialConnection
from ai_assistant.integrations.feishu_files import FeishuAPI, FileBridge
from ai_assistant.workspace.files import ReadOnlyWorkspace
from tests.integrations.test_mcp import config


def test_bot_failure_message_distinguishes_context_and_hides_details():
    from ai_assistant.interfaces.feishu import failure_message
    from ai_assistant.agents.context import ContextBudgetError
    message = failure_message(ContextBudgetError('private-token'))
    assert '上下文' in message and '/new' in message
    assert 'private-token' not in message
    assert '超时' in failure_message(TimeoutError('private-token'))
    assert 'private-token' not in failure_message(RuntimeError('private-token'))


def test_versioned_manifest_and_mutation_classification():
    items = capabilities()['tools']
    assert {x['domain'] for x in items} == {'im','docx','drive','wiki','bitable','calendar','task','contact'}
    assert len({x['name'] for x in items}) == len(items)
    assert not any('.permission.' in x['name'] for x in items)
    assert not next(x for x in items if x['native_name'] == 'wiki_v2_space_getNode')['write']
    with pytest.raises(ConfigError,match='mutations'):
        allowed_capabilities(MCPServer(transport='feishu',identity='bot',tools=['im_v1_message_create']))
    with pytest.raises(ConfigError):
        allowed_capabilities(MCPServer(transport='feishu',identity='bot',tools=['im_v1_message_delete']))


def test_official_tool_schemas_without_network_or_credentials(tmp_path):
    project = Path(__file__).resolve().parents[2] / 'deploy/feishu'
    if not (project/'node_modules/@larksuiteoapi/lark-mcp/dist/cli.js').is_file():
        pytest.skip('Run npm ci --prefix deploy/feishu for official schema verification')
    pytest.importorskip('fastmcp')
    async def run():
        items = [i for i in capabilities()['tools'] if 'bot' in i['identities']]
        settings = MCPServer(transport='feishu',identity='bot',tools=[i['native_name'] for i in items],
                             write_tools=[i['native_name'] for i in items if i['write']])
        c = config(tmp_path,mcp={'connect_timeout':30},integrations={'feishu':{
            'app_id':'cli_fixture','app_secret':'not-real','owner_open_id':'owner','node_project':str(project)}})
        tools = await OfficialConnection(c,settings,ReadOnlyWorkspace(tmp_path)).discover()
        assert {tool.name for tool in tools} == {i['native_name'] for i in items}
    asyncio.run(run())


def test_binary_bridge_and_explicit_identity(tmp_path):
    async def run():
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path.endswith('/tenant_access_token/internal'):
                return httpx.Response(200,json={'tenant_access_token':'bot-token','expire':7200})
            assert request.headers['authorization'] == 'Bearer bot-token'
            if request.method == 'GET':
                return httpx.Response(200,content=b'binary\x00fixture',headers={'content-type':'application/octet-stream'})
            return httpx.Response(200,json={'code':0,'data':{'file_token':'file-fixture'}})
        c = config(tmp_path,integrations={'feishu':{'app_id':'cli_fixture','app_secret':'not-real','owner_open_id':'owner'}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            api = FeishuAPI(c.integrations.feishu,client)
            bridge = FileBridge(api,ReadOnlyWorkspace(tmp_path))
            (tmp_path/'source.bin').write_bytes(b'file')
            assert (await bridge.upload_drive('source.bin','folder'))['file_token'] == 'file-fixture'
            result = await bridge.download_drive('file-fixture','result.bin')
            assert (tmp_path/result['path']).read_bytes() == b'binary\x00fixture'
            with pytest.raises(ValueError,match='Unsafe'):
                await bridge.download_drive('file-fixture','../escape')
        assert sum(r.url.path.endswith('/tenant_access_token/internal') for r in requests) == 1
    asyncio.run(run())
