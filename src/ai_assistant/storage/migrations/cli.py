"""数据库结构状态、显式迁移和备份的 CLI。"""
import json
from pathlib import Path

from ai_assistant.config import ConfigError
from ai_assistant.storage.migrations import runner
from ai_assistant.storage.migrations.backup import create_backup


def _add_config(command):
    command.add_argument('--config', default='assistant.toml')


def add_commands(commands):
    """注册 db status/migrate/backup；不构建模型或连接外部服务。"""
    parser = commands.add_parser('db', help='Database schema administration')
    actions = parser.add_subparsers(dest='action', required=True)

    status = actions.add_parser('status', help='Show schema version and pending migrations')
    _add_config(status)
    status.add_argument('--json', action='store_true')

    migrate = actions.add_parser('migrate', help='Preview or execute schema migrations')
    _add_config(migrate)
    migrate.add_argument('--execute', action='store_true')
    migrate.add_argument('--delete-legacy-sessions', action='store_true',
                         help='确认删除 binding 非空的旧渠道会话')
    migrate.add_argument('--no-backup', action='store_true',
                         help='破坏性迁移不创建备份')
    migrate.add_argument('--output', help='备份文件路径')
    migrate.add_argument('--json', action='store_true')

    backup = actions.add_parser('backup', help='Create a PostgreSQL logical backup')
    _add_config(backup)
    backup.add_argument('--output', help='备份文件路径')
    backup.add_argument('--json', action='store_true')


def _base(config):
    return config.config_path.parent if config.config_path else Path.cwd()


def _dump(data):
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _print_status(data):
    state = data.get('state')
    if state == 'uninitialized':
        print('会话结构尚未初始化；请先执行 ./assistant.sh init。')
        return
    if state == 'unsupported':
        print('会话结构版本不受支持；请检查数据库或应用版本。')
        return
    print(f"当前结构：v{data.get('current_version')}")
    print(f"目标结构：v{data.get('target_version')}")
    print(f"要求备份：{'是' if data.get('requires_backup') else '否'}")
    print(f"旧渠道会话：{data.get('legacy_sessions', 0)}")
    print(f"其他数据库连接：{data.get('other_clients', 0)}")
    issues = data.get('checksum_issues') or []
    if issues:
        print('迁移校验异常：' + '、'.join(str(item.get('version')) for item in issues))
    pending = data.get('pending') or []
    if not pending:
        print('待执行迁移：无')
    else:
        print(f"待执行迁移：{len(pending)}")
        for item in pending:
            preview = item.get('preview') or {}
            print(f"  v{item['from_version']} -> v{item['to_version']}  {item['name']}")
            if preview.get('legacy_tables'):
                print('    旧表：' + '、'.join(preview['legacy_tables']))
            if preview.get('error'):
                print('    检查失败：' + preview['error'])
    history = data.get('history') or []
    if history:
        print('迁移历史：')
        for item in history:
            print(f"  v{item['version']}  {item['name']}  {item['applied_at']}"
                  + (f"  备份：{item['backup_path']}" if item.get('backup_path') else ''))


def _print_migration(data):
    mode = data.get('mode')
    if mode == 'already-current':
        print(f"当前结构已是最新：v{data.get('current_version')}。")
        return
    _print_status(data)
    if mode == 'preview':
        print('以上为预览，未修改数据库；确认后加 --execute。')
        if data.get('requires_delete_confirmation'):
            print('存在旧渠道会话；执行时还需加 --delete-legacy-sessions。')
    elif mode == 'executed':
        print(f"迁移完成：v{data.get('current_version')}。")
        if data.get('backup_path'):
            print(f"备份：{data['backup_path']}")
        elif data.get('backup_skipped'):
            print('未创建备份：使用了 --no-backup。')
        for item in data.get('applied') or []:
            print(f"  已执行 v{item['from_version']} -> v{item['to_version']}  {item['name']}")


def execute(config, args):
    """执行 db 子命令；只使用显式配置的会话存储。"""
    if config.sessions is None:
        raise ConfigError('数据库操作需要配置 [sessions]。')
    if args.action == 'status':
        data = runner.status(config.sessions)
        _dump(data) if args.json else _print_status(data)
        return 0
    if args.action == 'migrate':
        data = runner.migrate(
            config.sessions, execute=args.execute,
            delete_legacy_sessions=args.delete_legacy_sessions,
            backup=not args.no_backup, base_dir=_base(config),
            output=args.output)
        _dump(data) if args.json else _print_migration(data)
        return 0
    if args.action == 'backup':
        path = create_backup(config.sessions, base_dir=_base(config),
                             output=args.output)
        data = {'mode': 'backup', 'path': path}
        _dump(data) if args.json else print(f"备份完成：{path}")
        return 0
    raise ConfigError('未知数据库操作。')
