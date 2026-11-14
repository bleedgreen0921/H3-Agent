"""Explicit persistence administration; no model clients are constructed."""
from ai_assistant.config import ConfigError
from ai_assistant.storage.sessions import session_storage


def add_commands(commands):
    """注册会话列表和删除的终端命令。"""
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
    """直接管理已保存会话，不构建模型客户端。"""
    if config.sessions is None:
        raise ConfigError('Configure [sessions] first.')
    async with session_storage(config.sessions) as repo:
        if args.action == 'list':
            # 支持按聊天绑定筛选，每页读取 10 条会话。
            if args.page < 1:
                raise ConfigError('页码必须大于 0。')
            rows = await repo.list_admin(args.binding, (args.page - 1) * 10, 10)
            for row in rows:
                print(f"{'● ' if row['current'] else ''}{row['title'] or '新会话'} | {row['id']} | {row['binding'] or '本机'} | {row['status']} | {row['updated_at']}")
        else:
            # 删除由会话仓库执行，不清理工作区或知识库。
            await repo.delete(args.session_id)
            print('指定会话已删除；工作区和知识库不受影响。')
    return 0
