"""Explicit local administration, including native Store initialization."""
from ai_assistant.storage.memory import memory_storage


def add_commands(commands):
    parser = commands.add_parser('memory', help='Personal memory administration')
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('init', 'list', 'delete', 'clear', 'pause', 'resume'):
        command = sub.add_parser(action)
        command.add_argument('--config', default='assistant.toml')
        if action == 'delete':
            command.add_argument('memory_id')
        if action == 'list':
            command.add_argument('--offset', type=int, default=0)


async def execute(config, args):
    async with memory_storage(config, initialize=args.action == 'init') as memory:
        print('记忆存储已显式初始化。' if args.action == 'init' else
              await memory.manage(args.action, getattr(args, 'memory_id', None), getattr(args, 'offset', 0)))
    return 0
