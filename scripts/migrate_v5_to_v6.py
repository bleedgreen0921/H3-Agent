"""兼容入口；推荐使用 ./assistant.sh db migrate。"""
import argparse
import json

from ai_assistant.config import ConfigError, load_config
from ai_assistant.storage.migrations.backup import BackupError
from ai_assistant.storage.migrations.base import MigrationError
from ai_assistant.storage.migrations.runner import migrate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='assistant.toml')
    parser.add_argument('--profile')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--delete-legacy-sessions', action='store_true',
                        help='确认删除 binding 非空的旧渠道会话')
    parser.add_argument('--no-backup', action='store_true',
                        help='破坏性迁移不创建备份')
    parser.add_argument('--output', help='备份文件路径')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, profile=args.profile)
        result = migrate(
            config.sessions, execute=args.execute,
            delete_legacy_sessions=args.delete_legacy_sessions,
            backup=not args.no_backup,
            base_dir=config.config_path.parent if config.config_path else None,
            output=args.output)
    except (ConfigError, MigrationError, BackupError) as error:
        print(str(error))
        return 2
    except Exception:
        print('迁移未完成：请检查数据库连接、独占状态和备份工具；未自动修改。')
        return 2
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
