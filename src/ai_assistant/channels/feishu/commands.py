"""飞书命令注册：所有 handler 在此分派，worker 不按命令名分支。"""
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_assistant.interfaces.commands import CommandContext, CommandRegistry, CommandResult, CommandSpec
from uuid import UUID
from ai_assistant.interfaces.chat_commands import bind, prepare_session_end, reply
from ai_assistant.storage.commands import CommandRepository
from ai_assistant.storage.queues import InboxRepository, OutboxRepository
from ai_assistant.storage.sessions import SessionError, session_storage


async def _session_end(ctx, name):
    from ai_assistant.observability import event
    async with session_storage(ctx.worker.config.sessions) as repo:
        stored = await prepare_session_end(repo, ctx.row)
        if stored['result']:
            async with repo.conn.transaction():
                await reply(repo.conn, ctx.row, stored['result'], ctx.worker.config.channels.feishu)
            return
        target = json.loads(stored['target'] or '{}')
        sid = target.get('session_id')
        if sid:
            await repo.acquire(sid)
            owned = await repo.session_binding(sid)
            if not owned or owned['binding'] != ctx.row['binding']:
                raise SessionError('取消目标会话不存在或不属于当前聊天。')
        async with repo.conn.transaction():
            active = target.get('active_input_id')
            if active:
                event('request.cancelled', command=name, target_id=active)
                await OutboxRepository(repo.conn).block_pending(active)
                await InboxRepository(repo.conn).block_running(active, ctx.row['binding'])
            if name == '/new':
                base = ctx.worker.config.config_path.parent if ctx.worker.config.config_path else Path.cwd()
                root = (base / ctx.worker.config.workspace.root).resolve()
                session = await repo.open(root)
                await bind(repo.conn, ctx.row['binding'], session['id'])
                text = f"已创建新会话。\nID：{session['id']}"
            else:
                text = '已停止当前执行；已发生的操作保留，已提交的知识库任务继续执行。'
            await CommandRepository(repo.conn).finish_existing(ctx.row['id'], text)
            await reply(repo.conn, ctx.row, text, ctx.worker.config.channels.feishu)


async def _new(ctx):
    await _session_end(ctx, '/new')
    return CommandResult()


async def _cancel(ctx):
    await _session_end(ctx, '/cancel')
    return CommandResult()


async def _usage(ctx):
    await ctx.worker.show_usage(ctx.row)
    return CommandResult()


async def _list(ctx):
    page = ctx.value
    async with session_storage(ctx.worker.config.sessions) as repo:
        conn = repo.conn
        commands = CommandRepository(conn)
        stored = await commands.get(ctx.row['id'])
        if stored and stored['result']:
            async with conn.transaction():
                await reply(conn, ctx.row, stored['result'], ctx.worker.config.channels.feishu)
            return CommandResult()
        rows = await repo.list_binding(ctx.row['binding'], (page - 1) * 10, 11)
        if not rows:
            text = '当前聊天暂无会话。' if page == 1 else '该页没有会话。'
        else:
            lines = [f'当前聊天的会话（第 {page} 页）：']
            for item in rows[:10]:
                stamp = item['updated_at'].astimezone(
                    ZoneInfo(ctx.worker.config.agent.timezone)).strftime('%Y-%m-%d %H:%M')
                lines.append(f"{'● ' if item['current'] else ''}{item['title'] or '新会话'}\nID：{item['id']}\n最近活动：{stamp}" +
                             ('｜删除未完成' if item['status'] == 'deleting' else ''))
            if len(rows) > 10:
                lines.append(f'下一页：/list {page + 1}')
            text = '\n\n'.join(lines)
        async with conn.transaction():
            await commands.upsert_result(ctx.row['id'], ctx.row['binding'], '/list', None, text)
            await reply(conn, ctx.row, text, ctx.worker.config.channels.feishu)
    return CommandResult()


async def _delete(ctx):
    target = ctx.value
    async with session_storage(ctx.worker.config.sessions) as repo:
        conn = repo.conn
        commands = CommandRepository(conn)
        stored = await commands.get(ctx.row['id'])
        if stored and stored['result']:
            async with conn.transaction():
                await reply(conn, ctx.row, stored['result'], ctx.worker.config.channels.feishu)
            return CommandResult()
        found = await repo.status_for_binding(target, ctx.row['binding'])
        if not found:
            text = '会话已删除。' if stored else '未找到可删除的会话。'
        else:
            try:
                await repo.acquire(target)
                current = await repo.status_for_binding(target, ctx.row['binding'])
                if not current or current['status'] == 'running':
                    raise SessionError('会话正在执行，请先 /cancel 并等待清理结束。')
                await commands.begin(ctx.row['id'], ctx.row['binding'], '/delete', target)
                await repo.delete(target, binding=ctx.row['binding'])
                from ai_assistant.observability import event
                event('request.session_action', command='/delete', target_id=target, status='deleted')
                text = '会话已删除（应用侧聊天、摘要及 checkpoint）。飞书消息、知识库、文件和必要业务回执保留。'
            except SessionError:
                check = await repo.status(target)
                if check and check['status'] == 'deleting':
                    raise
                text = '会话正在使用，请先 /cancel 并等待清理结束。'
        async with conn.transaction():
            await commands.upsert_result(ctx.row['id'], ctx.row['binding'], '/delete', target, text)
            await reply(conn, ctx.row, text, ctx.worker.config.channels.feishu)
    return CommandResult()


async def _skills(ctx):
    catalog = getattr(ctx.worker.resources, 'skills', None)
    text = '\n'.join(f"{s.id} | {s.metadata['name']}：{s.metadata['description']}"
                     for s in catalog.skills.values()) if catalog else '未配置 Skills。'
    text = text or '未配置可用 Skills。'
    async with session_storage(ctx.worker.config.sessions) as repo:
        commands = CommandRepository(repo.conn)
        await commands.upsert_result(ctx.row['id'], ctx.row['binding'], '/skills', None, text)
        async with repo.conn.transaction():
            await reply(repo.conn, ctx.row, text, ctx.worker.config.channels.feishu)
    return CommandResult()


async def _rag(ctx):
    from ai_assistant.knowledge.commands import handle as handle_rag
    # `/rag add` 属于管理指令：在 Active 任务中执行以便取消，但不能因此打开模型会话。
    await handle_rag(ctx.worker, ctx.row, ctx.caller, ctx.value, recover=ctx.recover)
    return CommandResult()


async def _skill(ctx):
    from ai_assistant.skills.catalog import SkillError
    identity, text = ctx.value
    if ctx.recover:
        return CommandResult()
    catalog = getattr(ctx.worker.resources, 'skills', None)
    if catalog is None:
        await ctx.worker.enqueue_reply(ctx.row, '未配置 Skills。')
        return CommandResult()
    try:
        identity = catalog.resolve(identity).id
    except SkillError:
        await ctx.worker.enqueue_reply(
            ctx.row, 'Skill 不存在、名称不唯一或未授权，请使用 /skills 中带 skill: 前缀的完整 ID。')
        return CommandResult()
    return CommandResult(start_turn=True, turn_text=text, skill_ids=[identity])


def _no_args(text):
    if text:
        raise ValueError
    return None


def _page(text):
    if not text:
        return 1
    if len(text.split()) != 1:
        raise ValueError
    page = int(text)
    if page < 1:
        raise ValueError
    return page


def _delete_parse(text):
    if len(text.split()) != 1 or str(UUID(text)) != text.lower():
        raise ValueError
    return str(UUID(text))


def _skill_parse(text):
    identity, _, request = text.partition(' ')
    if not identity or not request.strip():
        raise ValueError
    return identity, request.strip()


def _rag_parse(text):
    from ai_assistant.knowledge.commands import parse as parse_rag
    parts = ['/rag'] + (text.split(maxsplit=1) if text else [])
    return parse_rag(parts)


def default_registry():
    return CommandRegistry([
        CommandSpec('/new', '/new', _no_args, _new, preempt=True, progress=False),
        CommandSpec('/cancel', '/cancel', _no_args, _cancel, preempt=True, progress=False),
        CommandSpec('/usage', '/usage', _no_args, _usage, progress=False),
        CommandSpec('/list', '/list [页码]', _page, _list),
        CommandSpec('/delete', '/delete <会话ID>', _delete_parse, _delete),
        CommandSpec('/skills', '/skills', _no_args, _skills),
        CommandSpec('/skill', '/skill <ID> <请求>', _skill_parse, _skill, turn=True),
        CommandSpec('/rag', '/rag ...', _rag_parse, _rag, recover_rag=True,
                    turn_check=lambda value: bool(value and value[0] == 'add')),
    ])
