"""Explicit persistence administration; no model clients are constructed."""
import json

from ai_assistant.config import ConfigError
from ai_assistant.storage.inspect import (
    StorageInspectError, storage_inspector,
)
from ai_assistant.storage.sessions import SessionError, session_storage


def _add_config(command):
    command.add_argument('--config', default='assistant.toml')


def _add_json(command):
    command.add_argument('--json', action='store_true', help='输出 JSON')


def add_commands(commands):
    """注册会话管理和只读持久化检查命令。"""
    parser = commands.add_parser('session', help='Persistent conversation administration')
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('list', 'delete'):
        command = actions.add_parser(action)
        _add_config(command)
        if action == 'list':
            command.add_argument('--binding')
            command.add_argument('--page', type=int, default=1)
        if action == 'delete':
            command.add_argument('session_id')

    inspect = actions.add_parser('inspect', help='查看会话摘要、轮次和最近 checkpoint')
    _add_config(inspect)
    inspect.add_argument('session_id')
    inspect.add_argument('--limit', type=int, default=20)
    _add_json(inspect)

    checkpoints = actions.add_parser('checkpoints', help='列出 checkpoint 历史')
    _add_config(checkpoints)
    checkpoints.add_argument('session_id')
    checkpoints.add_argument('--thread')
    checkpoints.add_argument('--before')
    checkpoints.add_argument('--limit', type=int, default=20)
    _add_json(checkpoints)

    checkpoint = actions.add_parser('checkpoint', help='查看单个 checkpoint')
    _add_config(checkpoint)
    checkpoint.add_argument('session_id')
    checkpoint.add_argument('checkpoint_id')
    checkpoint.add_argument('--thread')
    checkpoint.add_argument('--channels', help='逗号分隔的 channel 名')
    checkpoint.add_argument('--full', action='store_true')
    _add_json(checkpoint)

    messages = actions.add_parser('messages', help='查看 Web 展示消息')
    _add_config(messages)
    messages.add_argument('session_id')
    messages.add_argument('--limit', type=int, default=100)
    _add_json(messages)

    operations = actions.add_parser('operations', help='查看操作回执')
    _add_config(operations)
    operations.add_argument('session_id')
    operations.add_argument('--limit', type=int, default=100)
    _add_json(operations)

    usage = actions.add_parser('usage', help='查看已保存的模型用量')
    _add_config(usage)
    usage.add_argument('session_id')
    _add_json(usage)


def _dump(data):
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _text(value, limit=120):
    text = '' if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + '…'


def _print_inspect(data, checkpoint_data):
    session = data['session']
    print(f"会话：{session.get('title') or '新会话'}")
    print(f"ID：{session.get('id')}")
    print(f"状态：{session.get('status')}")
    print(f"工作区：{session.get('workspace')}")
    print(f"当前线程：{session.get('thread_id')}")
    print(f"完成点：{session.get('complete') or '无'}")
    print(f"最近输入：{session.get('last_input_id') or '无'}")
    print(f"线程数：{len(data.get('threads') or [])}；checkpoint：{data['counts']['checkpoints']}；"
          f"Web 消息：{data['counts']['web_messages']}；操作：{data['counts']['operations']}")
    print('\n轮次：')
    for item in (checkpoint_data.get('items') or [])[:20]:
        print(f"  {item.get('latest_checkpoint_ts') or '-'}  "
              f"input_id={item.get('input_id')}  status={item.get('status') or '-'}  "
              f"checkpoints={item.get('checkpoints')}  operations={item.get('operations')}")
    if not checkpoint_data.get('items'):
        print('  暂无轮次')
    try:
        checkpoints = data.get('_checkpoints') or []
    except Exception:
        checkpoints = []
    print('\n最近 checkpoint：')
    for item in checkpoints:
        print(f"  step={item.get('step')}  {item.get('checkpoint_id')}  "
              f"phase={item.get('assistant_phase') or '-'}  pending={item.get('pending_writes')}")
    if not checkpoints:
        print('  暂无 checkpoint')


def _print_checkpoints(data):
    print(f"线程：{data.get('thread_id')}")
    for item in data.get('items') or []:
        print(f"step={item.get('step')}  {item.get('checkpoint_id')}  "
              f"phase={item.get('assistant_phase') or '-'}  "
              f"status={item.get('turn_status') or '-'}  pending={item.get('pending_writes')}")
        channels = ', '.join(channel['name'] for channel in item.get('channels') or [])
        if channels:
            print(f"  channels: {channels}")


def _print_checkpoint(data):
    print(f"checkpoint：{data.get('checkpoint_id')}")
    print(f"线程：{data.get('thread_id')}  step={data.get('step')}  "
          f"phase={data.get('assistant_phase') or '-'}  source={data.get('source')}")
    print(f"input_id：{data.get('input_id') or '-'}；"
          f"execution_thread_id：{data.get('execution_thread_id') or '-'}")
    if data.get('parent_checkpoint_id'):
        print(f"parent：{data['parent_checkpoint_id']}")
    print('\nChannels：')
    for item in data.get('channels') or []:
        print(f"  {item['name']}: {item['type']} 约 {item['approx_bytes']} bytes")
    print('\n内容：')
    print(json.dumps(data.get('channel_values'), ensure_ascii=False, indent=2, default=str))


def _print_messages(data):
    for item in data.get('items') or []:
        print(f"{item.get('created_at')}  {item.get('role')}  {item.get('status')}  "
              f"{_text(item.get('text'))}")


def _print_operations(data):
    for item in data.get('items') or []:
        print(f"{item.get('created_at')}  {item.get('status')}  {item.get('tool')}  "
              f"input={item.get('input_id')}  source={item.get('source_key') or '-'}")


def _print_usage(data):
    if not data.get('available'):
        print('用量结构不可用或尚未初始化。')
        return
    for item in data.get('items') or []:
        print(f"{item.get('role')}/{item.get('profile')}: calls={item.get('calls')} "
              f"input={item.get('input_tokens')} output={item.get('output_tokens')} "
              f"total={item.get('total_tokens')} unfinished={item.get('unfinished')} "
              f"missing={item.get('missing')}")


async def _inspect(args, settings):
    async with storage_inspector(settings) as inspector:
        session = await inspector.session(args.session_id)
        turns = await inspector.turns(args.session_id, limit=args.limit)
        try:
            checkpoints = await inspector.checkpoints(
                args.session_id, limit=min(args.limit, 20))
        except StorageInspectError:
            checkpoints = {'items': []}
        data = {**session, 'turns': turns, '_checkpoints': checkpoints.get('items')}
        if args.json:
            _dump(data)
        else:
            _print_inspect(data, turns)


async def execute(config, args):
    """直接管理已保存会话，不构建模型客户端。"""
    if config.sessions is None:
        raise ConfigError('Configure [sessions] first.')
    if args.action in {'list', 'delete'}:
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

    try:
        if args.action == 'inspect':
            await _inspect(args, config.sessions)
        else:
            async with storage_inspector(config.sessions) as inspector:
                if args.action == 'checkpoints':
                    data = await inspector.checkpoints(
                        args.session_id, thread_id=args.thread,
                        before=args.before, limit=args.limit)
                    _dump(data) if args.json else _print_checkpoints(data)
                elif args.action == 'checkpoint':
                    channels = [item.strip() for item in (args.channels or '').split(',')
                                if item.strip()]
                    data = await inspector.checkpoint(
                        args.session_id, args.checkpoint_id, thread_id=args.thread,
                        channels=channels or None, full=args.full)
                    _dump(data) if args.json else _print_checkpoint(data)
                elif args.action == 'messages':
                    data = await inspector.messages(args.session_id, limit=args.limit)
                    _dump(data) if args.json else _print_messages(data)
                elif args.action == 'operations':
                    data = await inspector.operations(args.session_id, limit=args.limit)
                    _dump(data) if args.json else _print_operations(data)
                elif args.action == 'usage':
                    data = await inspector.usage(args.session_id)
                    _dump(data) if args.json else _print_usage(data)
                else:
                    raise ConfigError('未知会话检查操作。')
    except (SessionError, StorageInspectError) as error:
        print(f"会话检查失败：{getattr(error, 'safe_message', None) or str(error)}")
        return 2
    return 0
