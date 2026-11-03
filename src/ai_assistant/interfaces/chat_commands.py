"""飞书管理命令：可信归属、原生会话锁和持久回执，不创建模型。"""
import json
from pathlib import Path
from uuid import UUID

from ai_assistant.storage.sessions import SessionError, session_storage
from ai_assistant.skills.catalog import skills_for_caller


USAGE = {
    '/new': '/new', '/list': '/list [页码]', '/delete': '/delete <会话ID>',
    '/cancel': '/cancel', '/usage': '/usage', '/skills': '/skills',
    '/skill': '/skill <完整ID或唯一名称> <请求>',
    '/rag': '/rag documents [页码] | sources [页码] | status <任务ID> | sync [来源ID] | add <飞书文档链接> | remove <文档ID>',
}


def parse_command(text):
    """普通文本返回 None；错误在进入模型或入库之前返回。"""
    text = text.strip()
    if not text.startswith('/'):
        return None
    parts = text.split(maxsplit=2)
    name = parts[0]
    if name not in USAGE:
        return ('error', '未支持的指令。')
    try:
        if name == '/rag':
            from ai_assistant.interfaces.rag_commands import parse
            return (name, parse(parts))
        if name in {'/new', '/cancel', '/usage', '/skills'}:
            if len(parts) != 1:
                raise ValueError
            return (name, None)
        if name == '/list':
            page = int(parts[1]) if len(parts) == 2 else 1
            if len(parts) > 2 or page < 1:
                raise ValueError
            return (name, page)
        if name == '/delete':
            if len(parts) != 2 or str(UUID(parts[1])) != parts[1].lower():
                raise ValueError
            return (name, str(UUID(parts[1])))
        if len(parts) != 3 or not parts[2].strip():
            raise ValueError
        return (name, (parts[1], parts[2]))
    except (ValueError, AttributeError):
        return ('error', '用法：' + USAGE[name])


async def reply(conn, row, text, settings):
    """调用方事务内提交回执及待投递消息，稳定 ID 防止重复投递。"""
    from ai_assistant.interfaces.outbox import enqueue_text
    await enqueue_text(conn, settings, row, text, purpose='diagnostic')


async def bind(conn, binding, session_id):
    cursor = await conn.execute('''UPDATE assistant_sessions SET binding=%s
        WHERE id=%s AND (binding IS NULL OR binding=%s)''', (binding, session_id, binding))
    if cursor.rowcount != 1:
        raise SessionError('会话不属于当前聊天。')
    await conn.execute('''INSERT INTO assistant_channel_sessions(binding,session_id) VALUES(%s,%s)
        ON CONFLICT(binding) DO UPDATE SET session_id=EXCLUDED.session_id''', (binding, session_id))


async def prepare_session_end(repo, row, *, active_input_id=None):
    """先固定取消目标，崩溃后不能把旧命令绑定到新的会话。"""
    stored = await (await repo.execute(
        'SELECT * FROM assistant_chat_commands WHERE input_id=%s', (row['id'],))).fetchone()
    if stored:
        return stored
    current = await (await repo.execute(
        'SELECT s.id,s.thread_id FROM assistant_sessions s JOIN assistant_channel_sessions c '
        'ON c.session_id=s.id WHERE c.binding=%s', (row['binding'],))).fetchone()
    target = {'active_input_id': active_input_id}
    if current:
        target.update(session_id=str(current['id']))
    await repo.execute('INSERT INTO assistant_chat_commands(input_id,binding,action,target) '
                       'VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING',
                       (row['id'], row['binding'], row['body']['text'].strip(), json.dumps(target)))
    return await (await repo.execute(
        'SELECT * FROM assistant_chat_commands WHERE input_id=%s', (row['id'],))).fetchone()


async def handle_session_end(worker, repo, row, name):
    stored = await prepare_session_end(repo, row)
    if stored['result']:
        async with repo.conn.transaction():
            await reply(repo.conn, row, stored['result'], worker.config.channels.feishu)
        return
    target = json.loads(stored['target'] or '{}')
    sid = target.get('session_id')
    if sid:
        await repo.acquire(sid)
        owned = await (await repo.execute('SELECT binding FROM assistant_sessions WHERE id=%s', (sid,))).fetchone()
        if not owned or owned['binding'] != row['binding']:
            raise SessionError('取消目标会话不存在或不属于当前聊天。')
    async with repo.conn.transaction():
        active = target.get('active_input_id')
        if active:
            # 取消已等待工具清理；旧输入只保留回执，不再补发旧预览。
            await repo.execute("UPDATE assistant_outbox SET status='blocked' WHERE input_id=%s AND status='pending'", (active,))
            await repo.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s AND binding=%s AND status='running'", (active, row['binding']))
        if name == '/new':
            base = worker.config.config_path.parent if worker.config.config_path else Path.cwd()
            root = (base / worker.config.workspace.root).resolve()
            session = await repo.open(root)
            await bind(repo.conn, row['binding'], session['id'])
            text = f"已创建新会话。\nID：{session['id']}"
        else:
            text = '已停止当前执行；已发生的操作保留，已提交的知识库任务继续执行。'
        await repo.execute('UPDATE assistant_chat_commands SET result=%s WHERE input_id=%s', (text, row['id']))
        await reply(repo.conn, row, text, worker.config.channels.feishu)


async def handle(worker, row, caller, parsed, *, recover=False):
    """管理命令恢复也走本入口；/skill 留给业务轮次处理。"""
    name, value = parsed
    if name == '/rag':
        from ai_assistant.interfaces.rag_commands import handle as handle_rag
        await handle_rag(worker, row, caller, value, recover=recover)
        return True
    if name == '/skill':
        return False
    if name == '/usage':
        await worker.show_usage(row)
        return True
    async with session_storage(worker.config.sessions) as repo:
        conn = repo.conn
        stored = await (await conn.execute('SELECT * FROM assistant_chat_commands WHERE input_id=%s', (row['id'],))).fetchone()
        if stored and stored['result']:
            async with conn.transaction():
                await reply(conn, row, stored['result'], worker.config.channels.feishu)
            return True
        if name in {'/new', '/cancel'}:
            await handle_session_end(worker, repo, row, name)
            return True
        if name == '/delete':
            # 删除原生 Saver 可能分步提交：先持久目标，再删除，最后提交回执。
            target = stored['target'] if stored else value
            found = await (await conn.execute('SELECT status FROM assistant_sessions WHERE id=%s AND binding=%s', (target, row['binding']))).fetchone()
            if not found:
                text = '会话已删除。' if stored else '未找到可删除的会话。'
            else:
                try:
                    await repo.acquire(target)
                    current = await (await conn.execute('SELECT status FROM assistant_sessions WHERE id=%s AND binding=%s', (target, row['binding']))).fetchone()
                    if not current or current['status'] == 'running':
                        raise SessionError('会话正在执行，请先 /cancel 并等待清理结束。')
                    await conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,target)
                        VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING''', (row['id'], row['binding'], name, target))
                    await repo.delete(target, binding=row['binding'])
                    text = '会话已删除（应用侧聊天、摘要及 checkpoint）。飞书消息、知识库、文件和必要业务回执保留。'
                except SessionError:
                    check = await (await conn.execute('SELECT status FROM assistant_sessions WHERE id=%s', (target,))).fetchone()
                    if check and check['status'] == 'deleting':
                        # 保留未完成记录，恢复/新指令可继续清理。
                        raise
                    text = '会话正在使用，请先 /cancel 并等待清理结束。'
            async with conn.transaction():
                await conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,target,result)
                    VALUES(%s,%s,%s,%s,%s) ON CONFLICT(input_id) DO UPDATE SET result=EXCLUDED.result''', (row['id'], row['binding'], name, target, text))
                await reply(conn, row, text, worker.config.channels.feishu)
            return True
        async with conn.transaction():
            if name == '/list':
                rows = await (await conn.execute('''SELECT s.id,s.title,s.updated_at,s.status,
                    (c.session_id=s.id) AS current FROM assistant_sessions s
                    LEFT JOIN assistant_channel_sessions c ON c.binding=s.binding
                    WHERE s.binding=%s ORDER BY s.updated_at DESC,s.id LIMIT 11 OFFSET %s''', (row['binding'], (value - 1) * 10))).fetchall()
                if not rows:
                    text = '当前聊天暂无会话。' if value == 1 else '该页没有会话。'
                else:
                    from zoneinfo import ZoneInfo
                    lines = [f'当前聊天的会话（第 {value} 页）：']
                    for item in rows[:10]:
                        stamp = item['updated_at'].astimezone(ZoneInfo(worker.config.agent.timezone)).strftime('%Y-%m-%d %H:%M')
                        lines.append(f"{'● ' if item['current'] else ''}{item['title'] or '新会话'}\nID：{item['id']}\n最近活动：{stamp}" + ('｜删除未完成' if item['status'] == 'deleting' else ''))
                    if len(rows) > 10:
                        lines.append(f'下一页：/list {value + 1}')
                    text = '\n\n'.join(lines)
            elif name == '/skills':
                catalog = skills_for_caller(worker.config, getattr(worker.resources, 'skills', None), caller)
                text = '\n'.join(f"{s.id} | {s.metadata['name']}：{s.metadata['description']}" for s in catalog.skills.values()) if catalog else ('本群未配置可用 Skills。' if caller.channel == 'feishu_group' else '未配置 Skills。')
                text = text or '未配置可用 Skills。'
            else:
                text = value
            await conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,result)
                VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING''', (row['id'], row['binding'], name, text))
            await reply(conn, row, text, worker.config.channels.feishu)
    return True
