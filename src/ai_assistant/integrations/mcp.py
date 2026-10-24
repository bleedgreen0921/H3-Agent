"""Native MCP adapter ownership and a fixed, identity-qualified tool directory."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

from ai_assistant.config import ConfigError
from ai_assistant.integrations.policy import Caller


@dataclass(frozen=True)
class Entry:
    id: str
    server: str
    original_name: str
    digest: str
    tool: object
    settings: object

    def info(self):
        return dict(id=self.id, server=self.server, identity=self.settings.identity,
                    name=self.original_name, description=self.tool.description,
                    schema_digest=self.digest, write=self.original_name in self.settings.write_tools)


class Catalog:
    def __init__(self, entries=(), *, timeout=30, adapters=None, servers=None):
        self.entries = {e.id: e for e in entries}
        self.timeout = timeout
        self.adapters = adapters or {}
        self.servers = servers or {}

    def search(self, query='', offset=0, limit=20):
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError('Invalid pagination')
        matches = [e.info() for e in self.entries.values()
                   if query.casefold() in (e.id + ' ' + e.tool.description).casefold()]
        return dict(items=matches[offset:offset+limit], total=len(matches),
                    next_offset=offset+limit if offset+limit < len(matches) else None)

    def for_caller(self, caller):
        if caller.channel != 'feishu_group':
            return self
        return Catalog([e for e in self.entries.values() if e.settings.identity == 'bot' and e.settings.transport == 'feishu'
                        and e.server in caller.resources], timeout=self.timeout)

    def active(self, state):
        return [e for e in self.entries.values() if state.get(e.id) == e.digest]


async def decline(*args, **kwargs):
    from fastmcp.client.elicitation import ElicitResult
    return ElicitResult(action='decline')


def stable_id(server, identity, name):
    # Provider-safe identifier, digest protects against normalization collisions.
    import re
    prefix = re.sub('[^a-zA-Z0-9_-]', '_', f'{server}_{identity}_{name}')[:43]
    digest = sha256(f'{server}\0{identity}\0{name}'.encode()).hexdigest()[:16]
    return f'mcp_{prefix}_{digest}'


@asynccontextmanager
async def mcp_resources(config, workspace, *, caller=None):
    if config.mcp is None or not config.mcp.servers:
        yield Catalog()
        return
    try:
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
        from langchain.mcp import MCPAdapter
    except ImportError:
        raise ConfigError('Install the mcp extra to enable MCP services.') from None
    caller = caller or Caller()
    entries = []
    adapters = {}
    async with AsyncExitStack() as stack:
        for name, settings in config.mcp.servers.items():
            if caller.channel == 'feishu_group' and (settings.identity != 'bot' or settings.transport != 'feishu' or name not in caller.resources):
                continue
            if not settings.tools and not settings.resources and not settings.prompts:
                continue
            tools = None
            if settings.transport == 'feishu':
                from ai_assistant.integrations.feishu import OfficialConnection
                tools = await OfficialConnection(config, settings, workspace).discover()
            elif settings.transport == 'stdio':
                env = {'PATH': os.defpath, 'LANG': 'C.UTF-8'}
                env.update({k: v.get_secret_value() for k, v in settings.env.items()})
                target = StdioTransport(command=settings.command, args=settings.args,
                                        env=env, keep_alive=True)
            elif settings.transport == 'feishu_files':
                import httpx
                from ai_assistant.integrations.feishu_files import FeishuAPI, create_server
                from ai_assistant.integrations.oauth import TokenStore
                if config.integrations.feishu is None:
                    raise ConfigError('Configure [integrations.feishu].')
                http = await stack.enter_async_context(httpx.AsyncClient(timeout=config.mcp.call_timeout,
                    trust_env=False, follow_redirects=False))
                tokens = TokenStore(config, workspace.root) if settings.identity == 'user' else None
                target = create_server(FeishuAPI(config.integrations.feishu, http, tokens), workspace)
            elif settings.transport == 'http':
                target = StreamableHttpTransport(settings.url,
                    headers={k: v.get_secret_value() for k, v in settings.headers.items()})
            else:
                from ai_assistant.integrations.netease import create_server
                target = create_server(settings.account,
                    config.integrations.netease.accounts[settings.account], workspace)
            if tools is None:
                client = Client(target, elicitation_handler=decline,
                                timeout=config.mcp.call_timeout,
                                init_timeout=config.mcp.connect_timeout)
                adapter = MCPAdapter(client)
                async with asyncio.timeout(config.mcp.connect_timeout):
                    await stack.enter_async_context(adapter)
                    tools = await adapter.list_tools()
                adapters[name] = adapter
            found = {tool.name for tool in tools}
            if set(settings.tools) - found:
                raise ConfigError(f'MCP server {name}: configured tools are missing.')
            for tool in tools:
                if tool.name not in settings.tools:
                    continue
                original = tool.name
                identity = stable_id(name, settings.identity, original)
                schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
                digest = sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
                native = tool.model_copy(update={'name': identity, 'metadata': {
                    **(tool.metadata or {}), 'office': {'server': name, 'identity': settings.identity,
                    'original_name': original, 'schema_digest': digest}}})
                entries.append(Entry(identity, name, original, digest, native, settings))
        yield Catalog(entries, timeout=config.mcp.call_timeout, adapters=adapters, servers=config.mcp.servers)
