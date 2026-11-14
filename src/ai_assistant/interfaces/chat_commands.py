"""飞书管理命令：可信归属、原生会话锁和持久回执，不创建模型。"""
import json
from uuid import UUID

from ai_assistant.storage.sessions import SessionError, session_storage
from ai_assistant.storage.commands import CommandRepository


async def reply(conn, row, text, settings):
    """调用方事务内提交回执及待投递消息，稳定 ID 防止重复投递。"""
    from ai_assistant.channels.feishu.outbox import enqueue_text
    await enqueue_text(conn, settings, row, text, purpose='diagnostic')


async def bind(conn, binding, session_id):
    """将可信聊天绑定到会话；事务由调用方管理。"""
    from ai_assistant.storage.sessions import SessionRepository
    await SessionRepository(conn, None).bind(binding, session_id)


async def prepare_session_end(repo, row, *, active_input_id=None):
    """先固定取消目标，崩溃后不能把旧命令绑定到新的会话。"""
    commands = CommandRepository(repo.conn)
    stored = await commands.get(row['id'])

    # 同一命令已登记时复用目标，避免重启后取消新启动的会话。
    if stored:
        return stored
    current = await repo.binding_session(row['binding'])
    target = {'active_input_id': active_input_id}
    if current:
        target.update(session_id=str(current['id']))

    # 先持久化目标，再由调用方执行取消和回复处理。
    await commands.begin(row['id'], row['binding'], row['body']['text'].strip(), json.dumps(target))
    return await commands.get(row['id'])
