"""Native MCP adapter ownership and a fixed, identity-qualified tool set."""
import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256

from ai_assistant.config import ConfigError
from ai_assistant.executables import executable, executable_env
from ai_assistant.integrations.policy import Caller


class MCPCleanupError(RuntimeError):
    """连接清理失败不能作为可选能力降级。"""


@asynccontextmanager
async def owned_adapter(adapter):
    import sys
    native = await adapter.__aenter__()
    try:
        yield native
    except BaseException:
        try:
            suppressed = await adapter.__aexit__(*sys.exc_info())
        except BaseException:
            raise MCPCleanupError('MCP connection cleanup failed.') from None
        if not suppressed:
            raise
    else:
        try:
            await adapter.__aexit__(None, None, None)
        except BaseException:
            raise MCPCleanupError('MCP connection cleanup failed.') from None


def schema_tokens(tools):
    """CLI 使用框架原生工具 schema 估算，不设置独立上限。"""
    from langchain_core.messages.utils import count_tokens_approximately
    return count_tokens_approximately([], tools=tools)


@dataclass(frozen=True)
class Entry:
    id: str
    server: str
    original_name: str
    tool: object
    settings: object

    def info(self):
        return dict(id=self.id, server=self.server, identity=self.settings.identity,
                    name=self.original_name, description=(self.tool.description or '')[:80],
                    schema_tokens=schema_tokens([self.tool]),
                    write=self.original_name in self.settings.write_tools)


class Catalog:
    def __init__(self, entries=(), *, timeout=60, unavailable=None):
        self.entries = {e.id: e for e in entries}
        self.timeout = timeout
        self.unavailable = unavailable or {}

    def for_caller(self, caller):
        if not caller.owner:
            return Catalog(timeout=self.timeout)
        return self



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
    caller = caller or Caller()
    entries = []
    unavailable = {}
    async with AsyncExitStack() as stack:
        for name, settings in config.mcp.servers.items():
            if not caller.owner:
                continue
            if not settings.tools:
                continue
            scope = AsyncExitStack()
            try:
                from fastmcp import Client
                from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
                from langchain.mcp import MCPAdapter
                pending = []
                tools = None
                if settings.transport == 'feishu':
                    from ai_assistant.integrations.feishu import OfficialConnection
                    tools = await OfficialConnection(config, settings, workspace).discover()
                elif settings.transport == 'stdio':
                    overrides = {k: v.get_secret_value() for k, v in settings.env.items()}
                    binary = executable(settings.command, path=overrides.get('PATH'))
                    env = {**executable_env(binary), **overrides}
                    target = StdioTransport(command=binary, args=settings.args,
                                            env=env, keep_alive=True)
                elif settings.transport == 'feishu_files':
                    import httpx
                    from ai_assistant.integrations.feishu_files import FeishuAPI, create_server
                    from ai_assistant.integrations.oauth import TokenStore
                    if config.integrations.feishu is None:
                        raise ConfigError('Configure [integrations.feishu].')
                    http = await scope.enter_async_context(httpx.AsyncClient(timeout=config.mcp.call_timeout,
                        trust_env=False, follow_redirects=False))
                    tokens = TokenStore(config, workspace.root) if settings.identity == 'user' else None
                    target = create_server(FeishuAPI(config.integrations.feishu, http, tokens), workspace)
                elif settings.transport == 'http':
                    target = StreamableHttpTransport(settings.url,
                        headers={k: v.get_secret_value() for k, v in settings.headers.items()})
                if tools is None:
                    client = Client(target, elicitation_handler=decline,
                                    timeout=config.mcp.call_timeout,
                                    init_timeout=config.mcp.connect_timeout)
                    adapter = MCPAdapter(client)
                    async with asyncio.timeout(config.mcp.connect_timeout):
                        await scope.enter_async_context(owned_adapter(adapter))
                        tools = await adapter.list_tools()
                found = {tool.name for tool in tools}
                if set(settings.tools) - found:
                    raise ConfigError(f'MCP server {name}: configured tools are missing.')
                for tool in tools:
                    if tool.name not in settings.tools:
                        continue
                    # 发布前使用原生接口确认配置工具具备可序列化的对象 schema。
                    schema = tool.args_schema
                    if schema is not None and not isinstance(schema, dict):
                        schema = schema.model_json_schema()
                    if not isinstance(schema, dict) or schema.get('type') != 'object':
                        raise ConfigError('MCP tool input schema is unavailable.')
                    json.dumps(schema, allow_nan=False)
                    original = tool.name
                    identity = stable_id(name, settings.identity, original)
                    native = tool.model_copy(update={'name': identity, 'metadata': {
                        **(tool.metadata or {}), 'office': {'server': name, 'identity': settings.identity,
                        'original_name': original}}})
                    pending.append(Entry(identity, name, original, native, settings))
            except BaseException as exc:
                # 清理异常和取消继续传播；只隔离服务初始化异常。
                await scope.aclose()
                from langgraph.errors import GraphBubbleUp
                from ai_assistant.storage.sessions import SessionStorageError
                if isinstance(exc, (GraphBubbleUp, SessionStorageError, MCPCleanupError)) or not isinstance(exc, Exception):
                    raise
                unavailable[name] = {'settings': settings, 'reason':
                    'dependency_unavailable' if isinstance(exc, ImportError) else
                    'timeout' if isinstance(exc, TimeoutError) else 'initialization_failed'}
                continue
            await stack.enter_async_context(scope)
            entries.extend(pending)
        yield Catalog(entries, timeout=config.mcp.call_timeout, unavailable=unavailable)
