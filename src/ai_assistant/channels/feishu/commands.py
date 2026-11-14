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
    """处理取消或新建会话，并保存可重复读取的命令结果。"""
    from ai_assistant.observability import event
    async with session_storage(ctx.worker.config.sessions) as repo:
        # 先固定命令对应的目标；重复处理时直接使用已保存的结果。
        stored = await prepare_session_end(repo, ctx.row)
        if stored['result']:
            async with repo.conn.transaction():
                await reply(repo.conn, ctx.row, stored['result'], ctx.worker.config.channels.feishu)
            return
        target = json.loads(stored['target'] or '{}')
        sid = target.get('session_id')
        if sid:
            # 获取会话锁并校验聊天归属，避免操作其他聊天的会话。
            await repo.acquire(sid)
            owned = await repo.session_binding(sid)
            if not owned or owned['binding'] != ctx.row['binding']:
                raise SessionError('取消目标会话不存在或不属于当前聊天。')
        async with repo.conn.transaction():
            # 阻止目标输入后续处理及尚待投递的结果。
            active = target.get('active_input_id')
            if active:
                event('request.cancelled', command=name, target_id=active)
                await OutboxRepository(repo.conn).block_pending(active)
                await InboxRepository(repo.conn).block_running(active, ctx.row['binding'])

            # /new 在终止旧会话处理后创建并绑定新会话；/cancel 只返回停止回执。
            if name == '/new':
                base = ctx.worker.config.config_path.parent if ctx.worker.config.config_path else Path.cwd()
                root = (base / ctx.worker.config.workspace.root).resolve()
                session = await repo.open(root)
                await bind(repo.conn, ctx.row['binding'], session['id'])
                text = f"已创建新会话。\nID：{session['id']}"
            else:
                text = '已停止当前执行；已发生的操作保留，已提交的知识库任务继续执行。'

            # 在同一事务内保存命令结果和待投递回复。
            await CommandRepository(repo.conn).finish_existing(ctx.row['id'], text)
            await reply(repo.conn, ctx.row, text, ctx.worker.config.channels.feishu)


async def _new(ctx):
    """结束当前执行并创建新会话。"""
    await _session_end(ctx, '/new')
    return CommandResult()


async def _cancel(ctx):
    """停止当前执行并保留已发生的操作。"""
    await _session_end(ctx, '/cancel')
    return CommandResult()


async def _usage(ctx):
    """委托 worker 查询当前会话的模型用量。"""
    await ctx.worker.show_usage(ctx.row)
    return CommandResult()


async def _list(ctx):
    """分页列出当前聊天绑定下的会话。"""
    page = ctx.value
    async with session_storage(ctx.worker.config.sessions) as repo:
        conn = repo.conn
        commands = CommandRepository(conn)
        stored = await commands.get(ctx.row['id'])
        if stored and stored['result']:
            async with conn.transaction():
                await reply(conn, ctx.row, stored['result'], ctx.worker.config.channels.feishu)
            return CommandResult()

        # 每页展示 10 条，多取 1 条判断是否需要显示下一页。
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

        # 命令结果与回复一起提交，重复处理时可复用结果。
        async with conn.transaction():
            await commands.upsert_result(ctx.row['id'], ctx.row['binding'], '/list', None, text)
            await reply(conn, ctx.row, text, ctx.worker.config.channels.feishu)
    return CommandResult()


async def _delete(ctx):
    """删除当前聊天下的指定会话，并保留不属于会话的数据。"""
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
                # 加锁后重新检查状态；运行中的会话需先取消并完成清理。
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
                # 已进入删除中的异常交由上层处理，不误报为普通占用。
                if check and check['status'] == 'deleting':
                    raise
                text = '会话正在使用，请先 /cancel 并等待清理结束。'
        async with conn.transaction():
            await commands.upsert_result(ctx.row['id'], ctx.row['binding'], '/delete', target, text)
            await reply(conn, ctx.row, text, ctx.worker.config.channels.feishu)
    return CommandResult()


async def _skills(ctx):
    """列出当前已发现的 Skills，并保存命令结果。"""
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


async def _plan(ctx):
    """将规划命令转换为需要启动的模型轮次。"""
    if not getattr(getattr(ctx.worker.config, 'planning', None), 'enabled', False):
        await ctx.worker.enqueue_reply(ctx.row, '规划能力未启用。')
        return CommandResult()
    action, goal = ctx.value
    text = {'run': '执行计划', 'exit': '退出规划模式', 'status': '查看计划状态'}.get(
        action, goal or '规划任务')
    return CommandResult(start_turn=True, turn_text=text,
                         turn_kwargs={'planning_action': action, 'planning_goal': goal})


async def _skill(ctx):
    """解析用户指定的 Skill，再把请求交给模型轮次。"""
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


# 参数解析器只负责校验和转换；无效输入由通用命令注册表返回用法。
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


def _plan_parse(text):
    """解析 /plan 子命令；其余内容全部作为初始目标。"""
    text = (text or '').strip()
    if not text:
        return 'create', None
    head, _, rest = text.partition(' ')
    if not rest and head in {'run', 'exit', 'status'}:
        return head, None
    return 'create', text


def _rag_parse(text):
    # 知识库子命令的语法由知识库模块统一维护。
    from ai_assistant.knowledge.commands import parse as parse_rag
    parts = ['/rag'] + (text.split(maxsplit=1) if text else [])
    return parse_rag(parts)


def default_registry():
    """集中声明飞书命令、参数解析器及调度标记。"""
    # preempt 用于抢占当前执行；turn 标记需要进入模型轮次的命令。
    return CommandRegistry([
        CommandSpec('/new', '/new', _no_args, _new, preempt=True, progress=False),
        CommandSpec('/cancel', '/cancel', _no_args, _cancel, preempt=True, progress=False),
        CommandSpec('/usage', '/usage', _no_args, _usage, progress=False),
        CommandSpec('/list', '/list [页码]', _page, _list),
        CommandSpec('/delete', '/delete <会话ID>', _delete_parse, _delete),
        CommandSpec('/skills', '/skills', _no_args, _skills),
        CommandSpec('/plan', '/plan <目标> | /plan run | /plan status | /plan exit',
                    _plan_parse, _plan, turn=True),
        CommandSpec('/skill', '/skill <ID> <请求>', _skill_parse, _skill, turn=True),
        CommandSpec('/rag', '/rag ...', _rag_parse, _rag, recover_rag=True,
                    turn_check=lambda value: bool(value and value[0] == 'add')),
    ])
