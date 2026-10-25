"""Explicit offline diagnostics and opt-in read-only discovery."""
import json
from pathlib import Path
import importlib.util
import importlib.metadata
import shutil
from ai_assistant.config import ConfigError
from ai_assistant.workspace.files import ReadOnlyWorkspace


def add_commands(commands):
    root = commands.add_parser('mcp', help='Office MCP configuration and diagnostics')
    actions = root.add_subparsers(dest='action', required=True)
    for action in ['init', 'list', 'inspect', 'check', 'tools']:
        parser = actions.add_parser(action)
        parser.add_argument('--config', default='assistant.toml')
        if action in {'inspect', 'tools'}:
            parser.add_argument('server')
        if action == 'check':
            parser.add_argument('server', nargs='?')
            parser.add_argument('--network', action='store_true')
    auth = actions.add_parser('auth').add_subparsers(dest='auth_service', required=True)
    feishu = auth.add_parser('feishu').add_subparsers(dest='auth_action', required=True)
    for action in ['login', 'status', 'logout']:
        parser = feishu.add_parser(action)
        parser.add_argument('--config', default='assistant.toml')
    operations = actions.add_parser('operations').add_subparsers(dest='operation_action', required=True)
    for action in ['list', 'inspect']:
        parser = operations.add_parser(action)
        parser.add_argument('--config', default='assistant.toml')
        if action == 'inspect':
            parser.add_argument('operation_id')


async def execute(config, args):
    if args.action == 'auth':
        from ai_assistant.integrations.oauth import TokenStore
        base = config.config_path.parent if config.config_path else Path.cwd()
        tokens = TokenStore(config, base / config.workspace.root)
        if args.auth_action == 'login':
            await tokens.login()
            print('Feishu owner authorization saved.')
        else:
            print(json.dumps(tokens.status() if args.auth_action == 'status' else tokens.logout()))
        return 0
    if args.action == 'init':
        from ai_assistant.storage.sessions import session_storage
        from ai_assistant.storage.integrations import operation_storage
        if config.sessions is None:
            raise ConfigError('mcp init requires [sessions].')
        async with session_storage(config.sessions, initialize=True):
            async with operation_storage(config.sessions, initialize=True):
                pass
        print('MCP integration schema initialized.')
        return 0
    if args.action == 'operations':
        from ai_assistant.storage.integrations import operation_storage
        async with operation_storage(config.sessions) as store:
            result = {'tools': await store.list(), 'channel_replies': await store.replies()} if args.operation_action == 'list' else await store.inspect(args.operation_id)
        print(json.dumps(result, default=str, ensure_ascii=False))
        return 0
    servers = config.mcp.servers if config.mcp else {}
    name = getattr(args, 'server', None)
    if name and name not in servers:
        raise ConfigError('Unknown MCP server.')
    chosen = {name: servers[name]} if name else servers
    if args.action in {'list', 'inspect'}:
        print(json.dumps({key: dict(transport=s.transport, identity=s.identity,
            tools=s.tools, write_tools=s.write_tools, allow_write=s.allow_write)
            for key, s in chosen.items()}, ensure_ascii=False))
        return 0
    if args.action == 'check' and not args.network:
        problems = []
        if chosen and importlib.util.find_spec('fastmcp') is None:
            problems.append('Install the mcp extra.')
        base = config.config_path.parent if config.config_path else Path.cwd()
        for key, settings in chosen.items():
            if settings.transport == 'stdio' and shutil.which(settings.command, path=settings.env['PATH'].get_secret_value() if 'PATH' in settings.env else None) is None:
                problems.append(f'{key}: configured stdio executable is unavailable.')
            if settings.transport == 'feishu':
                from ai_assistant.integrations.feishu import allowed_capabilities
                allowed_capabilities(settings)
                if config.integrations.feishu is None:
                    problems.append(f'{key}: configure integrations.feishu.')
                else:
                    project = base / config.integrations.feishu.node_project
                    for package, version in [('@larksuiteoapi/lark-mcp', '0.5.1'), ('@larksuiteoapi/node-sdk', '1.76.0'), ('pg', '8.23.1')]:
                        try:
                            actual = json.loads((project / 'node_modules' / package / 'package.json').read_text())['version']
                        except (OSError, ValueError, KeyError):
                            actual = None
                        if actual != version:
                            problems.append(f'{key}: install locked Node package {package}@{version} with npm ci.')
                    if shutil.which('node') is None:
                        problems.append(f'{key}: Node >=20 is required.')
        print(json.dumps({'ok': not problems, 'problems': problems, 'network': False}, ensure_ascii=False))
        return 2 if problems else 0
    # Explicit network diagnostics authenticate using read-only endpoints.
    from ai_assistant.integrations.mcp import mcp_resources
    base = config.config_path.parent if config.config_path else Path.cwd()
    workspace = ReadOnlyWorkspace(base / config.workspace.root, config_path=config.config_path,
                                  exclude_patterns=config.workspace.exclude_patterns)
    if config.mcp is None:
        print('No MCP services configured; no connections made.')
        return 0
    narrowed = config.model_copy(update={'mcp': config.mcp.model_copy(update={'servers': chosen})})
    async with mcp_resources(narrowed, workspace) as catalog:
        if args.action == 'check' and args.network:
            import asyncio
            import httpx
            from ai_assistant.integrations.feishu_files import FeishuAPI
            from ai_assistant.integrations.oauth import TokenStore
            async with asyncio.timeout(config.mcp.call_timeout):
                async with httpx.AsyncClient(timeout=config.mcp.call_timeout, trust_env=False) as http:
                    for settings in chosen.values():
                        if settings.transport in {'feishu', 'feishu_files'}:
                            tokens = TokenStore(config, workspace.root) if settings.identity == 'user' else None
                            api = FeishuAPI(config.integrations.feishu, http, tokens)
                            await api.request('GET', '/authen/v1/user_info' if tokens else '/bot/v3/info')
        print(json.dumps([{**e.info(), **({'input_schema': e.tool.args_schema} if args.action == 'tools' else {})} for e in catalog.entries.values()], ensure_ascii=False))
    return 0
