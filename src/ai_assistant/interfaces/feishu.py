"""One native Agent, durable inbox/outbox, SDK WebSocket reception."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import httpx
from ai_assistant.config import ConfigError
from ai_assistant.executables import executable, executable_env
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import assistant_session
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.tools.context import tool_context

logger = logging.getLogger(__name__)


def failure_message(error):
    """仅暴露错误类别，避免异常正文包含凭据或文档内容。"""
    from ai_assistant.agents.context import ContextBudgetError
    from ai_assistant.runtime import TurnIncompleteError
    from ai_assistant.storage.sessions import SessionStorageError
    if getattr(error, 'turn_message', None):
        return error.turn_message
    if isinstance(error, TurnIncompleteError):
        return str(error)
    if isinstance(error, ContextBudgetError):
        reason = '上下文超出预算：当前请求、必需回执或固定提示与工具 schema 无法容纳。请精简 mcp.servers.<服务>.tools 或 Skills、分页读取、缩小范围，或将文档入库。'
    elif isinstance(error, SessionStorageError):
        from ai_assistant.runtime.presentation import SAFE_MESSAGE
        return SAFE_MESSAGE
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
                      {key: tuple(value) for key, value in grant.resources.items()}, scope='shared',
                      skill_ids=tuple(grant.skill_ids if grant.skill_ids is not None else config.channels.feishu.default_group_skill_ids))
    return Caller('feishu_private', True, row['binding'], scope='private')


class BotWorker:
    def __init__(self, config, store, resources, api):
        self.config, self.store, self.resources, self.api = config, store, resources, api
        self.active = None
        self.active_row = None

    async def enqueue_reply(self, row, text, *, conn=None, purpose='answer', snapshot=None):
        from ai_assistant.interfaces.outbox import enqueue_text

        async def enqueue(connection):
            async with connection.transaction():
                await enqueue_text(connection, self.config.channels.feishu, row, text,
                    purpose=purpose, snapshot=snapshot)
        if conn is not None:
            await enqueue(conn)
        else:
            async with operation_storage(self.config.sessions) as reply_store:
                await enqueue(reply_store.conn)

    async def recover_reply(self, row, *, fallback=None):
        """投递恢复只读 saver 和索引，不构建 Agent 或模型。"""
        from ai_assistant.runtime.presentation import project, response_finished
        from ai_assistant.storage.sessions import session_storage

        saved = await (await self.store.conn.execute(
            'SELECT * FROM assistant_outbox WHERE input_id=%s ORDER BY part', (row['id'],))).fetchall()
        # 旧记录仅供诊断，升级不能自动补发。
        if any((item.get('body') or {}).get('delivery', {}).get('version') != 1 for item in saved):
            logger.warning('不支持的投递格式：input_id=%s；未恢复或发送。', row['id'])
            return
        # 升级前遗留的流式卡片：终态已确认投递即不重发，正文已在远端。
        cards = [item for item in saved if item['kind'] == 'card']
        if any(item['status'] == 'delivered' and response_finished(item['final_state']) for item in cards):
            await self.store.conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (row['id'],))
            return
        # 已登记的文本批次全部复用，残缺旧批次也不重新切分或补发。
        if any(item['part'] >= 0 for item in saved):
            await self.store.conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (row['id'],))
            return
        # 卡片未完成投递：放弃卡片行并降级为文本补发，优先使用已持久化的终态正文。
        if cards:
            await self.store.conn.execute("UPDATE assistant_outbox SET status='rejected' WHERE input_id=%s AND kind='card'", (row['id'],))
            for item in cards:
                snapshot = (item.get('body') or {}).get('presentation')
                if snapshot and snapshot.get('version') == 1:
                    await self.enqueue_reply(row, snapshot['text'], snapshot=snapshot)
                    return
        async with session_storage(self.config.sessions) as repo:
            session = await (await repo.execute(
                'SELECT s.* FROM assistant_sessions s JOIN assistant_channel_sessions c ON c.session_id=s.id WHERE c.binding=%s',
                (row['binding'],))).fetchone()
            if not session:
                display = project({'status': 'unknown', 'input_id': row['id']}, fallback or '未找到当前会话记录，未恢复或重新执行。')
            else:
                from ai_assistant.runtime.history import read_saved_turn
                found = await read_saved_turn(repo.saver, session, row['id'])
                if found is not None:
                    display = project(*found)
                else:
                    display = project({'status': 'unknown', 'input_id': row['id']}, fallback or '不支持的结果格式或缺少当前输入的原生正文定位；未恢复或重新执行。')
        await self.enqueue_reply(row, display.text, snapshot=display.snapshot())

    async def binding_session(self, row):
        result = await (await self.store.conn.execute('SELECT session_id FROM assistant_channel_sessions WHERE binding=%s',
            (row['binding'],))).fetchone()
        return str(result['session_id']) if result else None

    async def show_usage(self, row):
        """管理命令直接读取可信绑定；不打开 runtime 或竞争会话锁。"""
        caller_for(self.config, row)
        from ai_assistant.storage.usage import usage_storage
        sid = await self.binding_session(row)
        text = '当前会话尚未开始。'
        if sid:
            try:
                async with usage_storage(self.config.sessions) as usage:
                    text = await usage.report(sid)
            except Exception:
                text = '累计用量统计暂不可用。'
        await self.enqueue_reply(row, text)

    async def save_binding(self, row, session_id, conn):
        from ai_assistant.interfaces.chat_commands import bind
        async with conn.transaction():
            await bind(conn, row['binding'], session_id)

    async def execute(self, row, *, recover=False):
        caller = caller_for(self.config, row)
        from ai_assistant.interfaces.chat_commands import parse_command, handle
        from ai_assistant.skills.catalog import skills_for_caller
        text = row['body']['text'].strip()
        parsed = parse_command(text)
        if parsed and parsed[0] not in {'/skill'}:
            await handle(self, row, caller, parsed, recover=recover)
            return
        skill_ids = None
        if parsed and parsed[0] == '/skill':
            identity, text = parsed[1]
            if not recover:
                catalog = skills_for_caller(self.config, getattr(self.resources, 'skills', None), caller)
                if catalog is None:
                    await self.enqueue_reply(row, '本群未配置可用 Skills。' if caller.channel == 'feishu_group' else '未配置 Skills。')
                    return
                if identity not in catalog.skills:
                    await self.enqueue_reply(row, 'Skill ID 不存在或未授权，请使用 /skills 查看可用 ID。')
                    return
            skill_ids = [identity]
        if recover:
            await self.recover_reply(row)
            return
        session_id = await self.binding_session(row)
        async with assistant_session(self.config, session_id=session_id, caller=caller,
                                     resources=self.resources) as session:
            await self.save_binding(row, session.session_id, session.repository.conn)
            result = await session.run_turn(text, skill_ids=skill_ids, input_id=row['id'],
                user_source=text)
            from ai_assistant.runtime.presentation import project, saved_body
            display = project(result.result, saved_body(result.result, result.messages))
            await self.enqueue_reply(row, display.text, snapshot=display.snapshot())
            await session.after_reply()

    async def send_pending(self):
        rows = await (await self.store.conn.execute("SELECT o.* FROM assistant_outbox o WHERE o.status='pending' AND o.kind='text' AND o.next_attempt_at<=now() AND NOT EXISTS(SELECT 1 FROM assistant_outbox p WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status!='delivered') ORDER BY o.created_at,o.input_id,o.part,o.id")).fetchall()
        from psycopg.types.json import Jsonb
        from ai_assistant.interfaces.chat_commands import parse_command
        from ai_assistant.interfaces.delivery import classify, exhausted
        import time
        settings = self.config.channels.feishu
        for row in rows:
            # 慢投递期间出现可立即处理的输入（命令或空闲时的业务请求）先让位；主循环处理后下一轮续发。
            incoming = await (await self.store.conn.execute(
                "SELECT body->>'text' AS text FROM assistant_inbox WHERE status='received' ORDER BY received_at,id LIMIT 1")).fetchone()
            if incoming and (self.active is None or parse_command(incoming['text'] or '')):
                return
            body = row['body'] or {}
            delivery = body.get('delivery')
            # 旧记录仅供诊断，升级不能自动补发。
            if not delivery or delivery.get('version') != 1:
                continue
            if exhausted(delivery, settings):
                async with self.store.conn.transaction():
                    await self.store.conn.execute("UPDATE assistant_outbox SET status='unknown' WHERE id=%s", (row['id'],))
                    await self.store.conn.execute("UPDATE assistant_outbox SET status='blocked' WHERE input_id=%s AND part>%s AND status='pending'", (row['input_id'], row['part']))
                continue
            delivery['attempts'] += 1
            if delivery.get('first_terminal_at') is not None:
                delivery['terminal_attempts'] = delivery.get('terminal_attempts', 0) + 1
            delivery['last_outcome'] = 'in_flight'
            attempt_patch = {key: delivery[key] for key in ('attempts', 'terminal_attempts', 'last_outcome') if key in delivery}
            claimed = await (await self.store.conn.execute(
                "UPDATE assistant_outbox SET status='sending',body=jsonb_set(body,'{delivery}',body->'delivery' || %s) WHERE id=%s AND status='pending' RETURNING *",
                (Jsonb(attempt_patch), row['id']))).fetchone()
            if claimed is None:
                continue
            row = claimed
            body = row['body']
            delivery = body['delivery']
            started = time.monotonic()
            try:
                remaining = delivery.get('deadline')
                timeout = min(settings.delivery_attempt_timeout_seconds,
                              max(0, remaining - time.time()) if remaining else settings.delivery_attempt_timeout_seconds)
                async with asyncio.timeout(timeout):
                    data = await self.api.send_text(body['chat_id'], body['text'], row['id'])
                    if not isinstance(data.get('message_id'), str) or not data['message_id']:
                        raise RuntimeError('Missing delivery receipt')
            except Exception as error:
                category, diagnostic = classify(error)
                delivery['last_error'] = diagnostic
                delivery['last_outcome'] = category
                retry = category == 'temporary' and not exhausted(delivery, settings)
                status = 'pending' if retry else 'rejected' if category in {'rejected', 'temporary'} else 'unknown'
                delay = 1 if delivery.get('terminal_attempts', delivery['attempts']) <= 1 else 3
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
                    try:
                        requested_delay = float(error.response.headers.get('Retry-After', delay))
                    except ValueError:
                        requested_delay = delay
                    if 0 <= requested_delay <= max(0, (delivery.get('deadline') or time.time()) - time.time()):
                        delay = requested_delay
                async with self.store.conn.transaction():
                    await self.store.conn.execute("UPDATE assistant_outbox SET status=%s,body=jsonb_set(body,'{delivery}',body->'delivery' || %s),next_attempt_at=now()+%s*interval '1 second' WHERE id=%s",
                        (status, Jsonb({'last_error': diagnostic, 'last_outcome': category}), delay, row['id']))
                    logger.warning('Delivery operation=send_text elapsed=%.3f diagnostic=%s',
                                   time.monotonic()-started, diagnostic)
                    if not retry:
                        await self.store.conn.execute("UPDATE assistant_outbox SET status='blocked' WHERE input_id=%s AND part>%s AND status='pending'", (row['input_id'], row['part']))
                continue
            logger.info('Delivery operation=send_text elapsed=%.3f outcome=confirmed',
                        time.monotonic()-started)
            delivery['last_outcome'] = 'confirmed'
            await self.store.conn.execute("UPDATE assistant_outbox SET status='delivered',remote_id=%s,body=%s WHERE id=%s",
                                          (data.get('message_id'), Jsonb({'delivery': delivery}), row['id']))
            await self.store.conn.execute('''UPDATE assistant_inbox SET body=NULL,status='delivered'
                WHERE id=%s AND status='completed' AND NOT EXISTS(SELECT 1 FROM assistant_outbox WHERE input_id=%s AND (part>=0 OR kind='card') AND (status!='delivered' AND NOT (kind='card' AND status='rejected' AND EXISTS(SELECT 1 FROM assistant_outbox t WHERE t.input_id=assistant_outbox.input_id AND t.part>=0))))''',
                (row['input_id'], row['input_id']))

        await self.store.conn.execute("""UPDATE assistant_inbox i SET body=NULL,status='delivered'
            WHERE i.status='completed' AND EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id)
            AND NOT EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id
                AND (o.part>=0 OR o.kind='card') AND (o.status!='delivered' AND NOT (o.kind='card' AND o.status='rejected' AND EXISTS(SELECT 1 FROM assistant_outbox t WHERE t.input_id=o.input_id AND t.part>=0))))""")

    async def stop_for_command(self, row):
        """用户取消意图先落盘，再等待执行任务和工具清理，最后处理命令回执。"""
        from ai_assistant.storage.sessions import session_storage
        from ai_assistant.interfaces.chat_commands import prepare_session_end
        async with session_storage(self.config.sessions) as repo:
            same = self.active and self.active_row['binding'] == row['binding']
            stored = await prepare_session_end(repo, row,
                active_input_id=str(self.active_row['id']) if same else None)
        if stored['result']:
            return
        target = json.loads(stored['target'] or '{}')
        same = self.active and self.active_row['binding'] == row['binding']
        if same:
            # 单执行器保证绑定不在中途切换；旧命令不能取消后来启动的任务。
            current = await self.binding_session(row)
            if (current != target.get('session_id')
                    or target.get('active_input_id') != str(self.active_row['id'])):
                from ai_assistant.storage.sessions import SessionError
                raise SessionError('取消目标已变化，未取消当前执行。')
            self.active.cancel()
            from ai_assistant.cancellation import _drain_worker
            await _drain_worker(self.active)
            if not self.active.cancelled() and self.active.exception() is not None:
                raise self.active.exception()
            # 回执由取消命令承担，不能随后又发送一份系统中断回复。
            self.active = self.active_row = None

    async def recover(self):
        # 卡片机制已删除：本地确认中断的投递不推测远端结果，一律 unknown 且不重发。
        await self.store.conn.execute("UPDATE assistant_outbox SET status='unknown' WHERE status='sending' AND body->'delivery'->>'version'='1'")
        await self.store.conn.execute("UPDATE assistant_outbox o SET status='blocked' WHERE o.status='pending' AND o.part>=0 AND o.body->'delivery'->>'version'='1' AND EXISTS(SELECT 1 FROM assistant_outbox p WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status IN ('unknown','blocked','rejected'))")
        rows = await (await self.store.conn.execute("SELECT i.* FROM assistant_inbox i WHERE i.status='running' AND (i.body->>'delivery_version'='1' OR EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id AND o.body->'delivery'->>'version'='1')) ORDER BY i.received_at,i.id")).fetchall()
        for row in rows:
            await self.execute(row, recover=True)
        await self.send_pending()

    async def recover_rag(self, row):
        """失败后的补交只读业务回执；渠道仍不可用时留待重启恢复。"""
        try:
            await self.execute(row, recover=True)
        except Exception as exc:
            logger.error('RAG receipt recovery pending: input_id=%s error_type=%s',
                         row['id'], type(exc).__name__)

    async def run(self):
        await self.recover()
        try:
            while True:
                if self.active and self.active.done():
                    try:
                        self.active.result()
                    except asyncio.CancelledError:
                        if self.active_row['body']['text'].strip().startswith('/rag '):
                            # 取消可能发生在提交之后；只读业务回执，保留已经提交的 ID。
                            await self.recover_rag(self.active_row)
                        else:
                            await self.recover_reply(self.active_row)
                    except Exception as exc:
                        logger.error('Feishu turn failed: input_id=%s error_type=%s',
                                     self.active_row['id'], type(exc).__name__)
                        if self.active_row['body']['text'].strip().startswith('/rag '):
                            await self.recover_rag(self.active_row)
                        else:
                            # checkpoint 提交也可能已成功但确认失败，先只读检查，不能覆盖原答案。
                            await self.recover_reply(self.active_row, fallback=failure_message(exc))
                    self.active = self.active_row = None
                rows = await (await self.store.conn.execute("SELECT * FROM assistant_inbox WHERE status='received' ORDER BY received_at,id LIMIT 100")).fetchall()
                for row in rows:
                    command = row['body']['text'].strip()
                    if command == '/usage':
                        await self.show_usage(row)
                        continue
                    if command in {'/cancel', '/new'}:
                        try:
                            caller_for(self.config, row)
                            await self.stop_for_command(row)
                            await self.execute(row)
                        except Exception as exc:
                            await self.enqueue_reply(row, failure_message(exc))
                        continue
                    from ai_assistant.interfaces.chat_commands import parse_command
                    parsed = parse_command(command)
                    rag_add = bool(parsed and parsed[0] == '/rag' and parsed[1][0] == 'add')
                    if parsed and parsed[0] not in {'/new', '/skill'} and not rag_add:
                        try:
                            await self.store.conn.execute("UPDATE assistant_inbox SET status='running',body=jsonb_set(body,'{delivery_version}','1'::jsonb) WHERE id=%s", (row['id'],))
                            await self.execute(row)
                        except Exception as exc:
                            if parsed[0] == '/rag':
                                await self.recover_rag(row)
                            else:
                                await self.enqueue_reply(row, failure_message(exc))
                        continue
                    if self.active is None:
                        await self.store.conn.execute("UPDATE assistant_inbox SET status='running',body=jsonb_set(body,'{delivery_version}','1'::jsonb) WHERE id=%s", (row['id'],))
                        self.active_row = row
                        self.active = asyncio.create_task(self.execute(row))
                await self.send_pending()
                await asyncio.sleep(0.25)
        finally:
            if self.active:
                self.active.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await self.active


@asynccontextmanager
async def optional_knowledge_worker(config):
    """同进程启动一次；知识故障不参与聊天服务退出判定。"""
    async with AsyncExitStack() as stack:
        try:
            from ai_assistant.knowledge.sync import background_worker
            task = await stack.enter_async_context(background_worker(config))
        except Exception as error:
            # 会话/清理错误仍必须传播；知识数据库启动错误单独隔离。
            from ai_assistant.storage.sessions import SessionError
            from langgraph.errors import GraphBubbleUp
            if (isinstance(error, (SessionError, GraphBubbleUp))
                    or asyncio.current_task().cancelling()):
                raise
            logger.error('知识 Worker 启动失败；聊天继续，队列尚未确认处理，不自动重启。category=%s',
                         type(error).__name__)
            task = None

        def stopped(completed):
            error = None if completed.cancelled() else completed.exception()
            logger.error('知识 Worker 已退出；聊天继续，队列尚未确认处理，不自动重启。state=%s category=%s',
                         'cancelled' if completed.cancelled() else 'failed' if error else 'ended',
                         type(error).__name__ if error else 'none')

        if task is not None:
            task.add_done_callback(stopped)
        try:
            yield task
        finally:
            if task is not None:
                # 正常关闭由 background_worker 等待任务清理并关闭连接，不记作意外退出。
                task.remove_done_callback(stopped)


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
        if config.knowledge is not None:
            await stack.enter_async_context(optional_knowledge_worker(config))
        store = await stack.enter_async_context(operation_storage(config.sessions))
        leader = await (await store.conn.execute('SELECT pg_try_advisory_lock(714160004) AS locked')).fetchone()
        if not leader['locked']:
            raise ConfigError('Another Feishu executor is already running.')
        # Fail before starting reception if migration has not been explicitly run.
        await store.conn.execute('SELECT last_input_id FROM assistant_sessions LIMIT 0')
        await store.conn.execute('SELECT binding,title FROM assistant_sessions LIMIT 0')
        await store.conn.execute('SELECT input_id FROM assistant_chat_commands LIMIT 0')
        await store.conn.execute('SELECT id FROM assistant_inbox LIMIT 0')
        await store.conn.execute('SELECT kind,display_version,final_state FROM assistant_outbox LIMIT 0')
        context = await stack.enter_async_context(tool_context(config))
        directory = await stack.enter_async_context(mcp_resources(config, context.workspace))
        from ai_assistant.skills.catalog import validate_group_skills
        validate_group_skills(config, context.skills)
        resources = ProcessResources(directory, context.skills)
        client = await stack.enter_async_context(httpx.AsyncClient(timeout=30, trust_env=False))
        api = FeishuAPI(config.integrations.feishu, client)
        binary = executable('node')
        process = await asyncio.create_subprocess_exec(binary, str(receiver), cwd=str(project),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            env=executable_env(binary))
        payload = dict(final_delivery_timeout_seconds=config.channels.feishu.final_delivery_timeout_seconds, app_id=config.integrations.feishu.app_id,
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
        try:
            done, _ = await asyncio.wait([worker, receiver_wait], return_when=asyncio.FIRST_COMPLETED)
            if receiver_wait in done:
                raise ConfigError('Feishu receiver stopped; check Node deployment and application permissions.')
            await worker
        finally:
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
