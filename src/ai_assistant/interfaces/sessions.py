"""Explicit persistence administration; no model clients are constructed."""
from ai_assistant.config import ConfigError
from ai_assistant.storage.sessions import session_storage


def add_commands(commands):
    parser = commands.add_parser('session', help='Persistent conversation administration')
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('init', 'list', 'delete'):
        command = actions.add_parser(action)
        command.add_argument('--config', default='assistant.toml')
        if action == 'delete':
            command.add_argument('session_id')


async def execute(config, args):
    if config.sessions is None:
        raise ConfigError('Configure [sessions] first.')
    async with session_storage(config.sessions, initialize=args.action == 'init') as repo:
        if args.action == 'init':
            print('会话 schema 已初始化。')
        elif args.action == 'list':
            for row in await repo.list():
                print(f"{row['id']} | {row['updated_at']} | {row['status']} | {row['workspace']}")
        else:
            await repo.delete(args.session_id)
            print('指定会话已删除；工作区和知识库不受影响。')
    return 0
