"""飞书私人知识库指令，不创建会话或调用模型。"""
import asyncio
from zoneinfo import ZoneInfo

from ai_assistant.knowledge import KnowledgeError

USAGE = '/rag documents [页码] | sources [页码] | status <任务ID> | sync [来源ID] | add <飞书文档链接> | remove <文档ID>'


def parse(parts):
    if len(parts) < 2:
        raise ValueError
    action = parts[1]
    values = parts[2].split() if len(parts) == 3 else []
    if action in {'documents', 'sources'}:
        if len(values) > 1:
            raise ValueError
        value = int(values[0]) if values else 1
        if value < 1:
            raise ValueError
    elif action in {'status', 'add', 'remove'}:
        if len(values) != 1:
            raise ValueError
        value = values[0]
        if action == 'add':
            # 使用现有解析器验证 HTTPS 飞书链接；身份固定为主人 user。
            # 解析器还需要配置，实际处理时再次验证。
            from urllib.parse import urlsplit
            url = urlsplit(value)
            if url.scheme != 'https' or url.username or url.password or not url.hostname:
                raise ValueError
            if not any(url.hostname.endswith('.' + domain) for domain in ('feishu.cn', 'larksuite.com')):
                raise ValueError
            path = url.path.strip('/').split('/')
            if len(path) != 2 or path[0] not in {'wiki', 'docx', 'sheets', 'base'} or not path[1].isalnum():
                raise ValueError
    elif action == 'sync':
        if len(values) > 1:
            raise ValueError
        value = values[0] if values else None
    else:
        raise ValueError
    return action, value


def stamp(config, value):
    return value.astimezone(ZoneInfo(config.agent.timezone)).strftime('%Y-%m-%d %H:%M') if value else '未同步'


def render_result(action, result):
    if action == 'add':
        return f"已登记来源：{result['source_id']}\n同步任务：{result['sync_id']}\n已提交后台处理，可用 /rag status 查询。"
    if action == 'sync':
        return f"同步任务：{result['sync_id']}\n已提交后台处理，可用 /rag status 查询。"
    return f"文档：{result['document_id']}\n" + ('已移出检索范围；原始资料和历史快照保留。' if result['removed'] else '未找到该文档。')


async def handle(worker, row, caller, value, *, recover=False):
    from ai_assistant.interfaces.chat_commands import reply
    from ai_assistant.storage.integrations import operation_storage
    from ai_assistant.storage.rag import RagStore
    from ai_assistant.knowledge.sync import settings, resolve_submission
    from ai_assistant.knowledge.management import sources, status, receipt, apply

    action, target = value
    async with operation_storage(worker.config.sessions) as channel:
        stored = await (await channel.conn.execute('SELECT result FROM assistant_chat_commands WHERE input_id=%s', (row['id'],))).fetchone()
        if stored and stored['result']:
            async with channel.conn.transaction():
                await reply(channel.conn, row, stored['result'])
            return
        await channel.conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,target)
            VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
            (row['id'], row['binding'], '/rag ' + action, str(target) if target is not None else None))
        if caller.channel != 'feishu_private' or not caller.owner:
            message = '群聊不支持私人知识库管理。请使用主人私聊。'
        elif worker.config.knowledge is None:
            message = '当前未启用私人知识库。'
        else:
            store = RagStore(settings(worker.config))
            try:
                principal = 'feishu_private:' + worker.config.integrations.feishu.app_id + ':' + worker.config.integrations.feishu.owner_open_id + ':' + row['binding']
                if action in {'add', 'sync', 'remove'}:
                    result = await receipt(store, principal, row['id'])
                    if result is None and recover:
                        message = '管理操作未完成，未自动重做；已提交的知识库任务继续执行。请核实状态后再发起新指令。'
                    else:
                        if result is None:
                            submission = None
                            if action == 'add':
                                # 云端解析可能较慢，由 BotWorker 受管理任务执行，可取消并等待清理。
                                async with asyncio.timeout(worker.config.agent.turn_timeout_seconds):
                                    submission = await resolve_submission(worker.config, target, identity='user')
                            elif action == 'sync' and target:
                                if not any(s['id'] == target and s['enabled'] for s in await sources(worker.config, store)):
                                    raise KnowledgeError('来源不存在或未启用。')
                            result = await apply(store, principal, row['id'], action, target, submission=submission)
                        message = render_result(action, result)
                elif action in {'documents', 'sources'}:
                    rows = await store.document_list() if action == 'documents' else await sources(worker.config, store)
                    page = rows[(target - 1) * 10:target * 10]
                    lines = [f'知识库{ "文档" if action == "documents" else "来源"}（第 {target} 页）：']
                    for item in page:
                        if action == 'documents':
                            metadata = item.get('metadata') or {}
                            indexing = {'queued': '排队中', 'running': '索引中', 'failed': '索引失败',
                                        'cancelled': '索引已取消', 'succeeded': '索引完成', 'superseded': '已被新任务替代'}.get(item.get('index_state'), '待索引')
                            state = '已移除' if item['deleted'] else (('可检索｜' if item['active_build'] else '') + indexing)
                            # 只展示文档名及来源 ID，绝不展示本机 source_uri/path。
                            name = metadata.get('source_name') or '未命名文档'
                            source_ids = item.get('source_ids') or []
                            lines.append(f"{name}\nID：{item['id']}\n来源：{', '.join(source_ids) or '历史登记'}｜{state}")
                        else:
                            lines.append(f"{item['id']}｜{item['config']['kind']}\n最近同步：{stamp(worker.config, item['checked_at'])}｜{'启用' if item['enabled'] else '停用'}" + ('｜同步失败' if item.get('error') else ''))
                    if not page:
                        lines.append('该页没有记录。')
                    if len(rows) > target * 10:
                        lines.append(f'下一页：/rag {action} {target + 1}')
                    message = '\n\n'.join(lines)
                else:
                    result = await status(store, target)
                    message = f"任务：{result['id']}\n类型：{result['kind']}｜阶段：{result['state']}"
                    if result['kind'] == 'sync':
                        errors = result['errors']
                        message += f"\n创建：{stamp(worker.config, result.get('created_at'))}\n关联索引任务：" + (', '.join(j['job_id'] for j in result['jobs']) or '无')
                        message += f'\n覆盖不完整：{len(errors)} 项未核实。' if errors else ('\n同步完成。' if result['state'] == 'ready' else '\n后台处理中。')
                    elif result.get('error') or result.get('error_code'):
                        message += '\n索引失败，请在终端查看诊断。'
                    if result.get('document_deleted'):
                        message += '\n文档已移除，历史索引状态不代表当前可检索。'
            except KnowledgeError:
                message = '知识库请求无效或目标不存在，请检查参数、来源权限与任务 ID。'
            finally:
                await store.close()
        # 渠道与知识库不共享事务。崩溃恢复只查询业务回执并补交回复。
        async with channel.conn.transaction():
            await channel.conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,target,result)
                VALUES(%s,%s,%s,%s,%s) ON CONFLICT(input_id) DO UPDATE SET result=EXCLUDED.result''',
                (row['id'], row['binding'], '/rag ' + action, str(target) if target is not None else None, message))
            await reply(channel.conn, row, message)
