import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
import os
import socket
import sys
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from ai_assistant.agents import create_assistant_agent
from ai_assistant.config import AgentSettings
from ai_assistant.integrations.config import MCPServer
from ai_assistant.integrations.mcp import Catalog, Entry, mcp_resources, stable_id
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import AssistantSession
from ai_assistant.tools.context import ToolContext
from ai_assistant.tools.mcp import discovery_tools
from ai_assistant.workspace.files import ReadOnlyWorkspace
from tests.integrations.test_mcp import config
from tests.runtime.test_session import ScriptedModel

FIXTURE = str(Path(__file__).with_name('fixture_server.py'))


def message(name, args=None):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args or {}, 'id': str(uuid4())}])


def test_stdio_environment_elicitation_and_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv('ASSISTANT_TEST_MODEL_SECRET', 'never-inherit')
    async def run():
        c = config(tmp_path, mcp={'servers': {'fixture': {'transport': 'stdio', 'command': sys.executable,
            'args': [FIXTURE], 'tools': ['echo', 'environment', 'wait', 'ask']}}})
        async with mcp_resources(c, ReadOnlyWorkspace(tmp_path)) as directory:
            assert len(directory.entries) == 4
            entries = {e.original_name: e for e in directory.entries.values()}
            assert 'False' in str(await entries['environment'].tool.ainvoke({})) or 'false' in str(await entries['environment'].tool.ainvoke({}))
            elicitation = str(await entries['ask'].tool.ainvoke({}))
            assert 'decline' in elicitation or 'unavailable' in elicitation
            from ai_assistant.integrations.mcp import decline
            from fastmcp.client.elicitation import ElicitResult
            declined = await decline()
            assert isinstance(declined, ElicitResult) and declined.action == 'decline'
            task = asyncio.create_task(entries['wait'].tool.ainvoke({}))
            await asyncio.sleep(.05); task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert 'after cancel' in str(await entries['echo'].tool.ainvoke({'value': 'after cancel'}))
    asyncio.run(run())


def test_streamable_http(tmp_path):
    async def run():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        proc = await asyncio.create_subprocess_exec(sys.executable, FIXTURE, str(port),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    reader, writer = await asyncio.open_connection('127.0.0.1', port)
                    writer.close(); await writer.wait_closed(); break
                except OSError:
                    await asyncio.sleep(.05)
            c = config(tmp_path, mcp={'servers': {'http': {'transport': 'http',
                'url': f'http://127.0.0.1:{port}/mcp', 'tools': ['echo']}}})
            async with mcp_resources(c, ReadOnlyWorkspace(tmp_path)) as directory:
                result = await next(iter(directory.entries.values())).tool.ainvoke({'value': 'HTTP fixture'})
                assert 'HTTP fixture' in str(result)
        finally:
            proc.terminate(); await asyncio.wait_for(proc.wait(), 10)
    asyncio.run(run())


def test_native_load_enforcement_and_stale_schema(tmp_path):
    async def run():
        calls = []
        @tool
        async def read(value: str = 'read') -> str:
            """Record a harmless local fixture read."""
            calls.append(value); return value
        settings = MCPServer(transport='http', url='http://fixture', tools=['read'])
        identity = stable_id('fixture', 'private', 'read')
        native = read.model_copy(update={'name': identity})
        entry = Entry(identity, 'fixture', 'read', 'digest-v1', native, settings)
        directory = Catalog([entry])
        model = ScriptedModel(responses=[
            message(identity),  # Model cannot bypass activation even with a forged call.
            message('load_mcp_tools', {'tool_ids': [identity]}), message(identity), AIMessage(content='done'),
            message(identity), AIMessage(content='stale refused')])
        tools = discovery_tools() + [native]
        agent = create_assistant_agent(model, tools, checkpointer=InMemorySaver(), settings=AgentSettings(), mcp_enabled=True)
        context = ToolContext(ReadOnlyWorkspace(tmp_path), mcp=directory, caller=Caller())
        session = AssistantSession(agent, context, AgentSettings())
        first = await session.run_turn('read')
        errors = [m for m in first.messages if isinstance(m, ToolMessage) and m.status == 'error']
        assert len(errors) == 1 and calls == ['read']
        session.context = replace(context, mcp=Catalog([replace(entry, digest='digest-v2')]))
        second = await session.run_turn('read again')
        assert calls == ['read']
        assert any(isinstance(m, ToolMessage) and 'current schema' in str(m.content) for m in second.messages)
    asyncio.run(run())
