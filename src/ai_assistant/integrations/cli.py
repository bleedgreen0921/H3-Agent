"""Explicit offline diagnostics and opt-in read-only discovery."""
import json
from pathlib import Path
import importlib.util
import importlib.metadata
import shutil
from ai_assistant.config import ConfigError
from ai_assistant.workspace.files import ReadOnlyWorkspace


def _safe_operation(row):
    """终端诊断只展示安全摘要；运行时仍保留完整业务回执用于核对。"""
    result = {key: row[key] for key in ('id', 'input_id', 'principal', 'tool', 'status', 'created_at', 'updated_at') if key in row}
    needs_verification = row.get('needs_verification') or bool((row.get('result') or {}).get('business_outcome'))
    result.update(kind='tool_operation', target=row.get('tool'), receipt=(
        ('字段核验不符，需人工核实' if (row.get('result') or {}).get('verification', row.get('verification')) == 'mismatch' else '字段未核验，需人工核实') if needs_verification else
        {'succeeded': '已有成功回执', 'failed': '操作失败，未自动重放',
         'unknown': '结果未知，需核实', 'pending': '尚未确认完成'}.get(row['status'], '请核实状态')))
    return result


def add_commands(commands):
    """注册 MCP 服务、授权和操作回执相关的终端命令。"""
    root = commands.add_parser('mcp', help='Office MCP configuration and diagnostics')
    actions = root.add_subparsers(dest='action', required=True)

    # list 查看配置，check 检查服务，tools 查看指定服务的工具及 schema。
    for action in ['list', 'check', 'tools']:
        parser = actions.add_parser(action)
        parser.add_argument('--config', default='assistant.toml')
        if action == 'tools':
            parser.add_argument('server')
        if action in {'list', 'check'}:
            parser.add_argument('server', nargs='?')

    # 飞书主人授权和外部操作回执分别使用独立子命令。
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
    """按已解析的参数执行 MCP 授权、诊断或回执查询。"""
    if args.action == 'auth':
        # 授权状态由 TokenStore 管理；login 显式发起飞书用户授权。
        from ai_assistant.integrations.oauth import TokenStore
        base = config.config_path.parent if config.config_path else Path.cwd()
        tokens = TokenStore(config, base / config.workspace.root)
        if args.auth_action == 'login':
            await tokens.login()
            print('Feishu owner authorization saved.')
        else:
            print(json.dumps(tokens.status() if args.auth_action == 'status' else tokens.logout()))
        return 0

    if args.action == 'operations':
        # 查询持久化操作及渠道记录；工具操作只输出安全摘要。
        from ai_assistant.storage.integrations import operation_storage
        async with operation_storage(config.sessions) as store:
            result = {'tools': [_safe_operation(row) for row in await store.operations.list()], 'channel_replies': await store.diagnostics.replies(), 'management_commands': await store.diagnostics.commands()} if args.operation_action == 'list' else await store.diagnostics.inspect(args.operation_id)
        if isinstance(result, dict) and result.get('kind') == 'tool_operation':
            result = _safe_operation(result)

        # 配置知识库时补充管理命令的业务回执，便于核对渠道与入库状态。
        if config.knowledge is not None:
            from ai_assistant.knowledge.sync import settings
            from ai_assistant.storage.rag import RagStore
            knowledge = RagStore(settings(config))
            try:
                ident = args.operation_id if args.operation_action == 'inspect' else None
                receipts = await knowledge.management_receipts(ident)
                by_input = {r['input_id']: r for r in receipts}
                commands = result.get('management_commands', []) if isinstance(result, dict) and args.operation_action == 'list' else (result if isinstance(result, list) else [])
                for command in commands:
                    command['business_receipt'] = by_input.get(command['input_id'])
                # inspect 命中的是单条管理命令时也附带知识库业务回执，避免只看到渠道层状态。
                if (args.operation_action == 'inspect' and isinstance(result, dict)
                        and result.get('kind') == 'management_command'):
                    result['business_receipt'] = by_input.get(result['input_id'])
                if args.operation_action == 'inspect' and not result and receipts:
                    result = [dict(r, kind='management_command', status='committed') for r in receipts]
            finally:
                await knowledge.close()
        if args.operation_action == 'inspect' and not result:
            raise ConfigError('操作回执不存在。')
        print(json.dumps(result, default=str, ensure_ascii=False))
        return 0

    # 后续服务命令可限定某个配置名；未知名称直接报错。
    servers = config.mcp.servers if config.mcp else {}
    name = getattr(args, 'server', None)
    if name and name not in servers:
        raise ConfigError('Unknown MCP server.')
    chosen = {name: servers[name]} if name else servers

    # list 只展示配置概要，不连接 MCP 服务。
    if args.action == 'list':
        print(json.dumps({key: dict(transport=s.transport, identity=s.identity,
            tools=s.tools, write_tools=s.write_tools, allow_write=s.allow_write)
            for key, s in chosen.items()}, ensure_ascii=False))
        return 0

    # check 先离线检查依赖、可执行文件和飞书工具配置。
    if args.action == 'check':
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

                    # 只检查已安装的固定版本，不在诊断时安装依赖。
                    for package, version in [('@larksuiteoapi/lark-mcp', '0.5.1'), ('@larksuiteoapi/node-sdk', '1.76.0'), ('pg', '8.23.1')]:
                        try:
                            actual = json.loads((project / 'node_modules' / package / 'package.json').read_text())['version']
                        except (OSError, ValueError, KeyError):
                            actual = None
                        if actual != version:
                            problems.append(f'{key}: install locked Node package {package}@{version} with npm ci.')
                    if shutil.which('node') is None:
                        problems.append(f'{key}: Node >=20 is required.')
        if problems:
            print(json.dumps({'ok': False, 'problems': problems}, ensure_ascii=False))
            return 2

    # Explicit network diagnostics authenticate using read-only endpoints.
    from ai_assistant.integrations.mcp import mcp_resources
    base = config.config_path.parent if config.config_path else Path.cwd()
    workspace = ReadOnlyWorkspace(base / config.workspace.root, config_path=config.config_path, config_files=config.config_files,
                                  exclude_patterns=config.workspace.exclude_patterns)
    if config.mcp is None:
        print('No MCP services configured; no connections made.')
        return 0

    # check 和 tools 只连接本次选中的服务；只读工作区保留配置文件保护。
    narrowed = config.model_copy(update={'mcp': config.mcp.model_copy(update={'servers': chosen})})
    async with mcp_resources(narrowed, workspace) as catalog:
        if catalog.unavailable:
            print(json.dumps({'ok': False, 'unavailable': {
                name: status['reason'] for name, status in catalog.unavailable.items()}}, ensure_ascii=False))
            return 2
        if args.action == 'check':
            # 飞书身份额外调用只读接口验证当前用户或机器人连接。
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

        # tools 额外返回输入 schema；check 输出已发现工具的概要。
        print(json.dumps([{**e.info(), **({'input_schema': e.tool.args_schema} if args.action == 'tools' else {})} for e in catalog.entries.values()], ensure_ascii=False))
    return 0
