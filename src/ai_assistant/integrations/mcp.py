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


def _nested_paths(schema, key, prefix='', depth=0):
    """在 schema 的嵌套对象中定位同名字段，用于提示参数应放到的层级。"""
    if depth > 6:
        return []
    properties = schema.get('properties') if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return []
    found = []
    for name, child in properties.items():
        if not isinstance(child, dict):
            continue
        if name == key:
            found.append(prefix + name)
        found.extend(_nested_paths(child, key, prefix + name + '.', depth + 1))
    return found


def _parameter_problems(schema, arguments, path=''):
    """收集未知参数和缺失参数；一次返回全部问题，避免模型逐轮试错。"""
    where = path or '顶层'
    properties = schema.get('properties')
    problems = []
    if schema.get('additionalProperties') is not True:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            hints = []
            for key in unknown:
                candidates = [item for item in _nested_paths(schema, key) if item != key]
                hints.append(f'{key}（应位于 {candidates[0]}）' if candidates else key)
            problems.append(f'{where}包含未知参数：' + '、'.join(hints))
    missing = [key for key in schema.get('required', ()) if key not in arguments]
    if missing:
        details = []
        for key in missing:
            child = properties.get(key) if isinstance(properties.get(key), dict) else {}
            required = child.get('required') if isinstance(child, dict) else None
            details.append(f'{key}（包含 {"、".join(required)}）' if required else key)
        problems.append(f'{where}缺少必填参数：' + '、'.join(details))
    return problems


def validate_arguments(schema, arguments, path=''):
    """按 MCP JSON schema 校验参数层级，拒绝会被服务端静默忽略的未知字段。

    一次报告“未知参数”和“缺少必填参数”，并在嵌套字段同名时指出正确层级。
    这样本地预检失败可被模型修正，而不是只看到第一条错误后反复试探。
    """
    if not isinstance(schema, dict) or schema.get('type') != 'object':
        return
    properties = schema.get('properties')
    if not isinstance(properties, dict):
        return
    if not isinstance(arguments, dict):
        raise ValueError(f'{path or "参数"}必须是对象')
    problems = _parameter_problems(schema, arguments, path)
    if problems:
        raise ValueError('；'.join(problems))
    for key, value in arguments.items():
        child = properties.get(key)
        if isinstance(child, dict) and child.get('type') == 'object':
            validate_arguments(child, value, f'{path}.{key}' if path else key)


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
                    from ai_assistant.integrations.feishu_docs import (
                        SEMANTIC_TOOLS, create_semantic_tools)
                    native_tools = [name for name in settings.tools if name not in SEMANTIC_TOOLS]
                    native_writes = [name for name in settings.write_tools if name not in SEMANTIC_TOOLS]
                    native_settings = settings.model_copy(update={
                        'tools': native_tools, 'write_tools': native_writes})
                    tools = (await OfficialConnection(
                        config, native_settings, workspace).discover()) if native_tools else []
                    if set(settings.tools) & SEMANTIC_TOOLS:
                        import httpx
                        from ai_assistant.integrations.feishu_files import FeishuAPI
                        from ai_assistant.integrations.oauth import TokenStore
                        http = await scope.enter_async_context(httpx.AsyncClient(
                            timeout=config.mcp.call_timeout, trust_env=False,
                            follow_redirects=False))
                        tokens = TokenStore(config, workspace.root) if settings.identity == 'user' else None
                        api = FeishuAPI(config.integrations.feishu, http, tokens)
                        convert_api = FeishuAPI(config.integrations.feishu, http)
                        semantic = {tool.name: tool for tool in create_semantic_tools(
                            api, convert_api)}
                        tools.extend(tool for name, tool in semantic.items() if name in settings.tools)
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
