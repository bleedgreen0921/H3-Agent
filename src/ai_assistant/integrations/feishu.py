"""Pinned official MCP, explicit identity, and native schema-verified connections."""
import asyncio
from contextlib import asynccontextmanager
from importlib.resources import files
import json
import os
from pathlib import Path

from ai_assistant.config import ConfigError


def capabilities():
    return json.loads(files('ai_assistant.integrations').joinpath('feishu-capabilities.json').read_text())


def allowed_capabilities(settings):
    manifest = {item['native_name']: item for item in capabilities()['tools']}
    selected = []
    for name in settings.tools:
        item = manifest.get(name)
        if item is None or settings.identity not in item['identities']:
            raise ConfigError('Feishu tool or identity is outside the versioned capability manifest.')
        if item['write'] and name not in settings.write_tools:
            raise ConfigError('Feishu mutations must be declared write_tools.')
        selected.append(item)
    return selected


class OfficialConnection:
    """Per-service serialization. Native adapters own every connection and child process.

    Each invocation obtains a current token and a fresh official process, so a token
    refresh never leaves a running process holding an expired token. Calls are never
    replayed, and every new process must match the session's original tool schemas.
    """
    def __init__(self, config, settings, workspace):
        self.config, self.settings = config, settings
        self.feishu = config.integrations.feishu
        if self.feishu is None:
            raise ConfigError('Configure [integrations.feishu].')
        base = config.config_path.parent if config.config_path else Path.cwd()
        self.project = (base / self.feishu.node_project).resolve()
        self.entrypoint = self.project / 'node_modules/@larksuiteoapi/lark-mcp/dist/cli.js'
        try:
            package = json.loads((self.entrypoint.parent.parent / 'package.json').read_text())
        except OSError:
            raise ConfigError('Run npm ci --prefix deploy/feishu before using Feishu.') from None
        if package['version'] != '0.5.1':
            raise ConfigError('Official Feishu MCP version must be 0.5.1.')
        self.selected = allowed_capabilities(settings)
        self.tokens = None
        if settings.identity == 'user':
            from ai_assistant.integrations.oauth import TokenStore
            self.tokens = TokenStore(config, workspace.root)
        self.lock = asyncio.Lock()
        self.schemas = {}

    @asynccontextmanager
    async def connect(self):
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport
        from langchain.mcp import MCPAdapter
        from ai_assistant.integrations.mcp import decline
        env = {'PATH': os.defpath, 'LANG': 'C.UTF-8',
               'APP_ID': self.feishu.app_id,
               'APP_SECRET': self.feishu.app_secret.get_secret_value(),
               'LARK_TOOLS': ','.join(item['name'] for item in self.selected),
               'LARK_TOKEN_MODE': 'user_access_token' if self.tokens else 'tenant_access_token'}
        if self.tokens:
            env['USER_ACCESS_TOKEN'] = await self.tokens.access()
        target = StdioTransport(command='node', args=[str(self.entrypoint), 'mcp', '-c', 'snake'],
                                env=env, cwd=str(self.project), keep_alive=False, log_file=Path(os.devnull))
        client = Client(target, elicitation_handler=decline, timeout=self.config.mcp.call_timeout,
                        init_timeout=self.config.mcp.connect_timeout)
        async with MCPAdapter(client) as adapter:
            yield adapter

    async def discover(self):
        async with asyncio.timeout(self.config.mcp.connect_timeout):
            async with self.connect() as adapter:
                tools = await adapter.list_tools()
        self.schemas = {t.name: t.args_schema for t in tools}
        result = []
        def bind(name):
            async def invoke(**arguments):
                async with self.lock:
                    async with self.connect() as adapter:
                        current = {t.name: t for t in await adapter.list_tools()}
                        if {key: t.args_schema for key, t in current.items()} != self.schemas:
                            raise ConfigError('Feishu MCP schema changed; restart the session and reload tools.')
                        return await current[name].coroutine(**arguments)
            return invoke
        for tool in tools:
            result.append(tool.model_copy(update={'coroutine': bind(tool.name)}))
        return result
