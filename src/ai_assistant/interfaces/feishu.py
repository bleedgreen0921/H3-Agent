"""One native Agent, durable inbox/outbox, SDK WebSocket reception."""
import asyncio
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import httpx
from ai_assistant.config import ConfigError
from ai_assistant.executables import executable, executable_env
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import assistant_session, final_text, WaitingTurn
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.tools.context import tool_context

logger = logging.getLogger(__name__)


def failure_message(error):
    """仅暴露错误类别，避免异常正文包含凭据或文档内容。"""
    from ai_assistant.agents.context import ContextBudgetError
    from ai_assistant.runtime import TurnIncompleteError
    if isinstance(error, TurnIncompleteError):
        return str(error)
    if isinstance(error, ContextBudgetError):
        reason = '上下文超出预算：当前请求、必需回执或已激活提示与 schema 无法容纳。请分页读取、减少工具或 Skills、缩小范围，或将文档入库。'
    elif isinstance(error, TimeoutError):
        reason = '本轮执行超时。'
    else:
        reason = f'本轮执行失败（{type(error).__name__}）。'
    return reason + '本轮未自动重试；如涉及写入，请先查询操作回执。'


@dataclass(frozen=True)
class ProcessResources:
    catalog: object
    skills: object = None


def add_commands(commands):
    parser = commands.add_parser('feishu', help='Feishu owner bot')
    sub = parser.add_subparsers(dest='action', required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--config', default='assistant.toml')


def caller_for(config, row):
    settings = config.integrations.feishu
    body = row['body']
    if body.get('owner_open_id') != settings.owner_open_id:
        raise ConfigError('Untrusted inbox owner.')
    if body['group']:
        grant = config.channels.feishu.groups.get(body['chat_id'])
        if grant is None:
            raise ConfigError('Group is no longer authorized.')
        return Caller('feishu_group', True, row['binding'],
                      {key: tuple(value) for key, value in grant.resources.items()})
    return Caller('feishu_private', True, row['binding'])


class BotWorker:
    def __init__(self, config, store, resources, api):
        self.config, self.store, self.resources, self.api = config, store, resources, api
        self.active = None
        self.active_row = None

    async def enqueue_reply(self, row, text, *, progress=False):
        from psycopg.types.json import Jsonb
        # Store replies and completion atomically. Received text is cleared after delivery.
        async with operation_storage(self.config.sessions) as reply_store:
            async with reply_store.conn.transaction():
                for index in range(0, len(text), 3000):
                    await reply_store.conn.execute('''INSERT INTO assistant_outbox(id,input_id,body,part)
                        VALUES(%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING''', (
                            str(uuid5(NAMESPACE_URL, row['id'] + (':progress' if progress else ':reply:' + str(index // 3000)))), row['id'], Jsonb({
                            'chat_id': row['body']['chat_id'], 'text': text[index:index+3000]}), -1 if progress else index // 3000))
                if not progress:
                    await reply_store.conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (row['id'],))

    async def fail_reply(self, row, text, *, cancelled=False):
        if self.config.channels.feishu.streaming:
            from ai_assistant.interfaces.streaming import ReplyStream
            stream = ReplyStream(self.config, row)
            stream.emit({'kind': 'cancelled' if cancelled else 'failed', 'text': text})
            await stream.persist(self.store.conn)
            existing = await (await self.store.conn.execute("SELECT status FROM assistant_outbox WHERE input_id=%s AND kind='card'", (row['id'],))).fetchone()
            if existing and existing['status'] == 'rejected':
                await self.enqueue_reply(row, text)
            else:
                await self.store.conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (row['id'],))
        else:
            await self.enqueue_reply(row, text)

    async def restore_reply(self, row, text, *, incomplete=False):
        if self.config.channels.feishu.streaming:
            from ai_assistant.interfaces.streaming import ReplyStream
            stream = ReplyStream(self.config, row)
            stream.emit({'kind': 'failed' if incomplete else 'complete', 'text': text})
            await stream.persist(self.store.conn)
        await self.finish_reply(row, text)

    async def finish_reply(self, row, text):
        from ai_assistant.interfaces.streaming import card
        existing = await (await self.store.conn.execute(
            "SELECT * FROM assistant_outbox WHERE input_id=%s AND kind='card'", (row['id'],))).fetchone()
        if existing and existing['status'] != 'rejected':
            _, fits = card(text, 'complete', self.config.channels.feishu.card_max_bytes)
            if fits:
                await self.store.conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (row['id'],))
                return
        await self.enqueue_reply(row, text)

    async def binding_session(self, row):
        result = await (await self.store.conn.execute('SELECT session_id FROM assistant_channel_sessions WHERE binding=%s',
            (row['binding'],))).fetchone()
        return str(result['session_id']) if result else None

    async def show_usage(self, row):
        """管理命令直接读取可信绑定；不打开 runtime 或竞争会话锁。"""
        caller_for(self.config, row)
        from ai_assistant.storage.usage import UsageStore
        sid = await self.binding_session(row)
        text = await UsageStore(self.store.conn).report(sid) if sid else '当前会话尚未开始。'
        await self.enqueue_reply(row, text)

    async def save_binding(self, row, session_id):
        await self.store.conn.execute('''INSERT INTO assistant_channel_sessions VALUES(%s,%s)
            ON CONFLICT(binding) DO UPDATE SET session_id=EXCLUDED.session_id''', (row['binding'], session_id))

    async def execute(self, row, *, recover=False, resume=False):
        caller = caller_for(self.config, row)
        session_id = await self.binding_session(row)
        text = row['body']['text'].strip()
        async with assistant_session(self.config, session_id=session_id, caller=caller,
                                     resources=self.resources) as session:
            await self.save_binding(row, session.session_id)
            if recover:
                if session._pending:
                    await self.store.conn.execute("UPDATE assistant_inbox SET status='waiting' WHERE id=%s", (row['id'],))
                    if not self.config.channels.feishu.streaming:
                        await self.enqueue_reply(row, '正在同步知识来源，完成后将自动回答。', progress=True)
                    return
                current = await (await session.repository.execute(
                    'SELECT last_input_id FROM assistant_sessions WHERE id=%s', (session.session_id,))).fetchone()
                if current['last_input_id'] == row['id'] and session._complete:
                    state = await session.agent.aget_state(session._complete)
                    await self.restore_reply(row, final_text(state.values['messages']), incomplete=bool(state.values.get('turn_outcome')))
                    return
                await self.fail_reply(row, '上一轮未完成，可能已产生外部操作。请查询 mcp operations 核实；该请求不会自动重跑。')
                return
            if text == '/new':
                waiting_rows = await (await self.store.conn.execute("SELECT * FROM assistant_inbox WHERE binding=%s AND status='waiting'", (row['binding'],))).fetchall()
                for waiting in waiting_rows:
                    await self.fail_reply(waiting, '旧会话等待已取消，未完成；已入库资料仍保留。', cancelled=True)
                await session.new_conversation()
                await self.save_binding(row, session.session_id)
                await self.enqueue_reply(row, '已创建新会话。')
                return
            if text == '/help':
                await self.enqueue_reply(row, '/help /new /status /cancel /usage；发送文本即可使用助手。主人私聊可用 /skills、/skill <ID> <请求> 与 /memory。')
                return
            if text.startswith('/memory'):
                if session.context.memory is None:
                    await self.enqueue_reply(row, '当前入口未启用个人记忆。')
                else:
                    parts = text.split()
                    action = parts[1] if len(parts) > 1 else 'list'
                    argument = parts[2] if len(parts) > 2 else None
                    try:
                        answer = await session.context.memory.manage(action,
                            argument if action == 'delete' else None,
                            int(argument) if action == 'list' and argument else 0)
                    except (ValueError, TypeError):
                        answer = '用法：/memory [list OFFSET|delete ID|clear|pause|resume]'
                    await self.enqueue_reply(row, answer)
                return
            skill_ids = None
            if caller.channel == 'feishu_private' and text == '/skills':
                catalog = session.context.skills
                await self.enqueue_reply(row, '\n'.join(f'{s.id}: {s.metadata["name"]}' for s in catalog.skills.values()) if catalog else '未配置 Skills。')
                return
            if caller.channel == 'feishu_private' and text.startswith('/skill '):
                parts = text.split(maxsplit=2)
                if len(parts) != 3:
                    await self.enqueue_reply(row, '用法：/skill <ID> <请求>')
                    return
                skill_ids, text = [parts[1]], parts[2]
            user_source = text
            if not resume and self.config.knowledge is not None and caller.channel == 'feishu_private':
                from ai_assistant.knowledge.admission import admit_message
                admitted = await admit_message(self.config, text, attachment=text.startswith('File reference: '))
                if admitted:
                    text += '\n应用状态：以下为文档长期入库登记结果，failed 表示失败，其余尚不代表索引已完成；成功项请用 search_knowledge 并按 source_id 检索。失败时可检查已配置 MCP 的读取权限，不能声称未配置集成：' + json.dumps(admitted, ensure_ascii=False)
            stream = None
            if self.config.channels.feishu.streaming:
                from ai_assistant.interfaces.streaming import ReplyStream
                stream = await ReplyStream(self.config, row).start()
            try:
                result = await session.run_turn(text, skill_ids=skill_ids, input_id=row['id'], resume=resume,
                    user_source=user_source, on_output=stream.emit if stream else None)
            finally:
                if stream:
                    await stream.close()
            if isinstance(result, WaitingTurn):
                await self.store.conn.execute("UPDATE assistant_inbox SET status='waiting' WHERE id=%s", (row['id'],))
                if not stream:
                    await self.enqueue_reply(row, '正在同步知识来源，完成后将自动回答。', progress=True)
                return
            await self.finish_reply(row, final_text(result.messages))

    async def send_pending(self):
        rows = await (await self.store.conn.execute("SELECT o.* FROM assistant_outbox o WHERE o.status='pending' AND o.next_attempt_at<=now() AND NOT EXISTS(SELECT 1 FROM assistant_outbox p WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status!='delivered') ORDER BY o.created_at,o.input_id,o.part,o.id")).fetchall()
        for row in rows:
            # Persist sending before the remote effect. A crash is never treated as unsent.
            await self.store.conn.execute("UPDATE assistant_outbox SET status='sending' WHERE id=%s", (row['id'],))
            is_card = row.get('kind') == 'card'
            try:
                if is_card:
                    data = await self.api.update_card(row['remote_id'], row['body']['card']) if row['remote_id'] else await self.api.send_card(row['body']['chat_id'], row['body']['card'], row['id'])
                else:
                    data = await self.api.send_text(row['body']['chat_id'], row['body']['text'], row['id'])
                if is_card and not row['remote_id'] and not isinstance(data.get('message_id'), str):
                    raise RuntimeError('Card creation has no confirmed message ID.')
                if is_card and not row['remote_id'] and not data.get('message_id'):
                    raise RuntimeError('Card creation has no confirmed message ID.')
            except Exception as error:
                from ai_assistant.integrations.feishu_files import FeishuRejected
                rejected = isinstance(error, FeishuRejected) and error.code in {230001, 230006}
                throttled = (isinstance(error, FeishuRejected) and error.code == 99991400) or (
                    isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429)
                retry = is_card and (bool(row['remote_id']) or throttled) and not rejected
                await self.store.conn.execute("UPDATE assistant_outbox SET status=%s,next_attempt_at=now()+interval '1 second' WHERE id=%s",
                    ('rejected' if is_card and rejected else 'pending' if retry else 'unknown', row['id']))
                if is_card and rejected and row['final_state'] in {'complete', 'failed', 'cancelled'}:
                    inbox = await (await self.store.conn.execute('SELECT * FROM assistant_inbox WHERE id=%s', (row['input_id'],))).fetchone()
                    if inbox and inbox['body'] and row['body'].get('final_text'):
                        await self.enqueue_reply(inbox, row['body']['final_text'])
                continue
            if is_card:
                await self.store.conn.execute("""UPDATE assistant_outbox SET
                    status=CASE WHEN display_version=%s THEN 'delivered' ELSE 'pending' END,
                    remote_id=COALESCE(remote_id,%s),delivered_version=%s WHERE id=%s""",
                    (row['display_version'], data.get('message_id'), row['display_version'], row['id']))
            else:
                await self.store.conn.execute("UPDATE assistant_outbox SET status='delivered',remote_id=%s,body=NULL WHERE id=%s",
                                              (data.get('message_id'), row['id']))
            await self.store.conn.execute('''UPDATE assistant_inbox SET body=NULL,status='delivered'
                WHERE id=%s AND status='completed' AND NOT EXISTS(SELECT 1 FROM assistant_outbox WHERE input_id=%s AND (part>=0 OR kind='card') AND status NOT IN ('delivered','rejected'))''',
                (row['input_id'], row['input_id']))

        await self.store.conn.execute("""UPDATE assistant_inbox i SET body=NULL,status='delivered'
            WHERE i.status='completed' AND EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id)
            AND NOT EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id
                AND (o.part>=0 OR o.kind='card') AND o.status NOT IN ('delivered','rejected'))""")

    async def recover(self):
        await self.store.conn.execute("UPDATE assistant_outbox SET status=CASE WHEN kind='card' AND remote_id IS NOT NULL THEN 'pending' ELSE 'unknown' END WHERE status='sending'")
        rows = await (await self.store.conn.execute("SELECT * FROM assistant_inbox WHERE status IN ('running','waiting') ORDER BY received_at,id")).fetchall()
        for row in rows:
            await self.execute(row, recover=True)
        await self.send_pending()

    async def run(self):
        await self.recover()
        try:
            while True:
                if self.active and self.active.done():
                    try:
                        self.active.result()
                    except asyncio.CancelledError:
                        await self.fail_reply(self.active_row, '执行已取消，未完成；已发生的远端操作不会撤销，请核实操作回执。', cancelled=True)
                    except Exception as exc:
                        logger.error('Feishu turn failed: input_id=%s error_type=%s',
                                     self.active_row['id'], type(exc).__name__)
                        await self.fail_reply(self.active_row, failure_message(exc))
                    self.active = self.active_row = None
                rows = await (await self.store.conn.execute("SELECT * FROM assistant_inbox WHERE status='received' ORDER BY received_at,id LIMIT 100")).fetchall()
                for row in rows:
                    command = row['body']['text'].strip()
                    if command == '/usage':
                        await self.show_usage(row)
                        continue
                    if command in {'/status', '/cancel'}:
                        caller_for(self.config, row)
                        same = self.active_row and self.active_row['binding'] == row['binding']
                        waiting = await (await self.store.conn.execute("SELECT * FROM assistant_inbox WHERE binding=%s AND status='waiting' LIMIT 1", (row['binding'],))).fetchone()
                        if waiting and command == '/cancel':
                            async with assistant_session(self.config, session_id=await self.binding_session(row),
                                                         caller=caller_for(self.config,row), resources=self.resources) as session:
                                await session.cancel_waiting()
                            await self.fail_reply(waiting, '已取消本次知识回答等待，未完成；已入库资料仍保留。', cancelled=True)
                        same = same or waiting
                        if command == '/cancel' and self.active and self.active_row['binding'] == row['binding']:
                            self.active.cancel()
                        await self.enqueue_reply(row, ('已请求取消。' if same else '当前会话没有正在执行的请求。') if command == '/cancel'
                            else ('当前会话正在执行。' if same else '当前会话空闲或排队中。'))
                        continue
                    if self.active is None:
                        waiting = await (await self.store.conn.execute("SELECT 1 FROM assistant_inbox WHERE binding=%s AND status='waiting' LIMIT 1", (row['binding'],))).fetchone()
                        if waiting and command != '/new':
                            continue
                        await self.store.conn.execute("UPDATE assistant_inbox SET status='running' WHERE id=%s", (row['id'],))
                        self.active_row = row
                        self.active = asyncio.create_task(self.execute(row))
                if self.active is None and self.config.knowledge is not None:
                    from ai_assistant.knowledge.sync import settings
                    from ai_assistant.storage.rag import RagStore
                    knowledge = RagStore(settings(self.config))
                    try:
                        waiting_rows = await (await self.store.conn.execute("SELECT * FROM assistant_inbox WHERE status='waiting' ORDER BY received_at,id")).fetchall()
                        for row in waiting_rows:
                            sid = await self.binding_session(row)
                            stored = await (await self.store.conn.execute('SELECT pending FROM assistant_sessions WHERE id=%s', (sid,))).fetchone()
                            if not stored or not stored['pending']:
                                await self.execute(row, recover=True)
                                continue
                            status = await knowledge.sync_status(stored['pending']['sync_id'])
                            if status and status['state'] == 'ready':
                                self.active_row = row
                                self.active = asyncio.create_task(self.execute(row, resume=True))
                                break
                    finally:
                        await knowledge.close()
                await self.send_pending()
                await asyncio.sleep(0.25)
        finally:
            if self.active:
                self.active.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await self.active


async def serve(config):
    application_logger = logging.getLogger('ai_assistant')
    application_logger.setLevel(logging.INFO)
    if not application_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        application_logger.addHandler(handler)
        application_logger.propagate = False
    if not config.channels.feishu.enabled or config.integrations.feishu is None or config.sessions is None:
        raise ConfigError('Enable [channels.feishu] and configure Feishu and sessions.')
    from ai_assistant.integrations.mcp import mcp_resources
    from ai_assistant.integrations.feishu_files import FeishuAPI
    base = config.config_path.parent if config.config_path else Path.cwd()
    project = (base / config.integrations.feishu.node_project).resolve()
    receiver = project / 'receiver.mjs'
    if not receiver.is_file() or not (project / 'node_modules/pg/package.json').is_file():
        raise ConfigError('Deploy the pinned Feishu Node project with npm ci.')
    async with AsyncExitStack() as stack:
        knowledge_worker = None
        if config.knowledge is not None:
            from ai_assistant.knowledge.sync import background_worker
            knowledge_worker = await stack.enter_async_context(background_worker(config))
        store = await stack.enter_async_context(operation_storage(config.sessions))
        leader = await (await store.conn.execute('SELECT pg_try_advisory_lock(714160004) AS locked')).fetchone()
        if not leader['locked']:
            raise ConfigError('Another Feishu executor is already running.')
        # Fail before starting reception if migration has not been explicitly run.
        await store.conn.execute('SELECT last_input_id FROM assistant_sessions LIMIT 0')
        await store.conn.execute('SELECT call_id FROM assistant_model_usage LIMIT 0')
        await store.conn.execute('SELECT enabled_at FROM assistant_usage_schema LIMIT 0')
        await store.conn.execute('SELECT id FROM assistant_inbox LIMIT 0')
        await store.conn.execute('SELECT kind,display_version,final_state FROM assistant_outbox LIMIT 0')
        # 已持有唯一飞书执行器锁，前一进程未结束的调用不能继续显示为进行中。
        await store.conn.execute("UPDATE assistant_model_usage SET status='unknown',ended_at=clock_timestamp() WHERE ended_at IS NULL AND session_id IN (SELECT session_id FROM assistant_channel_sessions)")
        memory_repo = None
        if config.memory.enabled:
            from ai_assistant.storage.memory import memory_storage
            memory_repo = await stack.enter_async_context(memory_storage(config))
        context = await stack.enter_async_context(tool_context(config))
        directory = await stack.enter_async_context(mcp_resources(config, context.workspace))
        resources = ProcessResources(directory, context.skills)
        client = await stack.enter_async_context(httpx.AsyncClient(timeout=30, trust_env=False))
        api = FeishuAPI(config.integrations.feishu, client)
        binary = executable('node')
        process = await asyncio.create_subprocess_exec(binary, str(receiver), cwd=str(project),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            env=executable_env(binary))
        payload = dict(app_id=config.integrations.feishu.app_id,
            app_secret=config.integrations.feishu.app_secret.get_secret_value(),
            owner_open_id=config.integrations.feishu.owner_open_id,
            database_url=config.sessions.database_url.get_secret_value(),
            groups=list(config.channels.feishu.groups))
        # Receiver expects a key map, not an ordered list.
        payload['groups'] = {key: True for key in payload['groups']}
        process.stdin.write((json.dumps(payload) + '\n').encode())
        await process.stdin.drain()
        worker = asyncio.create_task(BotWorker(config, store, resources, api).run())
        receiver_wait = asyncio.create_task(process.wait())
        memory_worker = None
        if memory_repo is not None:
            async def maintain_memory():
                while True:
                    await memory_repo.drain()
                    await asyncio.sleep(1)
            memory_worker = asyncio.create_task(maintain_memory())
        try:
            done, _ = await asyncio.wait([worker, receiver_wait] + ([memory_worker] if memory_worker else []) + ([knowledge_worker] if knowledge_worker else []), return_when=asyncio.FIRST_COMPLETED)
            if memory_worker and memory_worker in done:
                await memory_worker
                raise ConfigError('Memory worker stopped.')
            if knowledge_worker and knowledge_worker in done:
                await knowledge_worker
                raise ConfigError('Knowledge worker stopped.')
            if receiver_wait in done:
                raise ConfigError('Feishu receiver stopped; check Node deployment and application permissions.')
            await worker
        finally:
            if memory_worker:
                memory_worker.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await memory_worker
            worker.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await worker
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 10)
                except TimeoutError:
                    process.kill(); await process.wait()
            receiver_wait.cancel()
            with suppress(asyncio.CancelledError):
                await receiver_wait
