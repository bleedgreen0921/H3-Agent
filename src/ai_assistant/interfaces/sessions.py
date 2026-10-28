"""Explicit persistence administration; no model clients are constructed."""
from ai_assistant.config import ConfigError
from ai_assistant.storage.sessions import session_storage


def add_commands(commands):
    parser = commands.add_parser('session', help='Persistent conversation administration')
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('list', 'delete'):
        command = actions.add_parser(action)
        command.add_argument('--config', default='assistant.toml')
        if action == 'list':
            command.add_argument('--binding')
            command.add_argument('--page', type=int, default=1)
        if action == 'delete':
            command.add_argument('session_id')


async def execute(config, args):
    if config.sessions is None:
        raise ConfigError('Configure [sessions] first.')
    async with session_storage(config.sessions) as repo:
        if args.action == 'list':
            if args.page < 1:
                raise ConfigError('页码必须大于 0。')
            rows = await (await repo.execute('''SELECT s.id,s.title,s.binding,s.updated_at,s.status,
                EXISTS(SELECT 1 FROM assistant_channel_sessions c WHERE c.binding=s.binding AND c.session_id=s.id) AS current
                FROM assistant_sessions s WHERE (%s::text IS NULL OR s.binding=%s)
                ORDER BY s.updated_at DESC,s.id LIMIT 10 OFFSET %s''',
                (args.binding, args.binding, (args.page - 1) * 10))).fetchall()
            for row in rows:
                print(f"{'● ' if row['current'] else ''}{row['title'] or '新会话'} | {row['id']} | {row['binding'] or '本机'} | {row['status']} | {row['updated_at']}")
        else:
            await repo.delete(args.session_id)
            print('指定会话已删除；工作区和知识库不受影响。')
    return 0
