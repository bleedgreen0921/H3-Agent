"""One native Agent, durable inbox/outbox, SDK WebSocket reception."""
import asyncio
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from uuid import uuid4

import httpx
from ai_assistant.config import ConfigError
from ai_assistant.executables import executable, executable_env
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import assistant_session, final_text
from ai_assistant.storage.integrations import operation_storage
from ai_assistant.tools.context import tool_context

logger = logging.getLogger(__name__)


def failure_message(error):
    """仅暴露错误类别，避免异常正文包含凭据或文档内容。"""
    from ai_assistant.agents.context import ContextBudgetError
    if isinstance(error, ContextBudgetError):
        reason = '上下文超出预算。可发送 /new 开始新会话，或缩小本次请求范围。'
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

    async def enqueue_reply(self, row, text):
        from psycopg.types.json import Jsonb
        # Store replies and completion atomically. Received text is cleared after delivery.
        async with operation_storage(self.config.sessions) as reply_store:
            async with reply_store.conn.transaction():
                for index in range(0, len(text), 3000):
                    await reply_store.conn.execute('''INSERT INTO assistant_outbox(id,input_id,body,part)
                        VALUES(%s,%s,%s,%s)''', (str(uuid4()), row['id'], Jsonb({
                            'chat_id': row['body']['chat_id'], 'text': text[index:index+3000]}), index // 3000))
                await reply_store.conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (row['id'],))

    async def binding_session(self, row):
        result = await (await self.store.conn.execute('SELECT session_id FROM assistant_channel_sessions WHERE binding=%s',
            (row['binding'],))).fetchone()
        return str(result['session_id']) if result else None

    async def save_binding(self, row, session_id):
        await self.store.conn.execute('''INSERT INTO assistant_channel_sessions VALUES(%s,%s)
            ON CONFLICT(binding) DO UPDATE SET session_id=EXCLUDED.session_id''', (row['binding'], session_id))

    async def execute(self, row, *, recover=False):
        caller = caller_for(self.config, row)
        session_id = await self.binding_session(row)
        text = row['body']['text'].strip()
        async with assistant_session(self.config, session_id=session_id, caller=caller,
                                     resources=self.resources) as session:
            await self.save_binding(row, session.session_id)
            if recover:
                current = await (await session.repository.execute(
                    'SELECT last_input_id FROM assistant_sessions WHERE id=%s', (session.session_id,))).fetchone()
                if current['last_input_id'] == row['id'] and session._complete:
                    state = await session.agent.aget_state(session._complete)
                    await self.enqueue_reply(row, final_text(state.values['messages']))
                    return
                await self.enqueue_reply(row, '上一轮执行被中断，可能已产生外部操作。请查询 mcp operations 核实；该请求不会自动重跑。')
                return
            if text == '/new':
                await session.new_conversation()
                await self.save_binding(row, session.session_id)
                await self.enqueue_reply(row, '已创建新会话。')
                return
            if text == '/help':
                await self.enqueue_reply(row, '/help /new /status /cancel；发送文本即可使用助手。主人私聊可用 /skills 与 /skill <ID> <请求>。')
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
            result = await session.run_turn(text, skill_ids=skill_ids, input_id=row['id'])
            await self.enqueue_reply(row, final_text(result.messages))

    async def send_pending(self):
        rows = await (await self.store.conn.execute("SELECT o.* FROM assistant_outbox o WHERE o.status='pending' AND NOT EXISTS(SELECT 1 FROM assistant_outbox p WHERE p.input_id=o.input_id AND p.part<o.part AND p.status!='delivered') ORDER BY o.created_at,o.input_id,o.part,o.id")).fetchall()
        for row in rows:
            # Persist sending before the remote effect. A crash is never treated as unsent.
            await self.store.conn.execute("UPDATE assistant_outbox SET status='sending' WHERE id=%s", (row['id'],))
            try:
                data = await self.api.send_text(row['body']['chat_id'], row['body']['text'], row['id'])
            except Exception:
                await self.store.conn.execute("UPDATE assistant_outbox SET status='unknown' WHERE id=%s", (row['id'],))
                continue
            await self.store.conn.execute("UPDATE assistant_outbox SET status='delivered',remote_id=%s,body=NULL WHERE id=%s",
                                          (data.get('message_id'), row['id']))
            await self.store.conn.execute('''UPDATE assistant_inbox SET body=NULL,status='delivered'
                WHERE id=%s AND NOT EXISTS(SELECT 1 FROM assistant_outbox WHERE input_id=%s AND status!='delivered')''',
                (row['input_id'], row['input_id']))

    async def recover(self):
        await self.store.conn.execute("UPDATE assistant_outbox SET status='unknown' WHERE status='sending'")
        rows = await (await self.store.conn.execute("SELECT * FROM assistant_inbox WHERE status='running' ORDER BY received_at,id")).fetchall()
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
                        await self.enqueue_reply(self.active_row, '执行已取消；已发生的远端操作不会撤销，请核实操作回执。')
                    except Exception as exc:
                        logger.error('Feishu turn failed: input_id=%s error_type=%s',
                                     self.active_row['id'], type(exc).__name__)
                        await self.enqueue_reply(self.active_row, failure_message(exc))
                    self.active = self.active_row = None
                rows = await (await self.store.conn.execute("SELECT * FROM assistant_inbox WHERE status='received' ORDER BY received_at,id LIMIT 100")).fetchall()
                for row in rows:
                    command = row['body']['text'].strip()
                    if command in {'/status', '/cancel'}:
                        caller_for(self.config, row)
                        same = self.active_row and self.active_row['binding'] == row['binding']
                        if command == '/cancel' and same:
                            self.active.cancel()
                        await self.enqueue_reply(row, ('已请求取消。' if same else '当前会话没有正在执行的请求。') if command == '/cancel'
                            else ('当前会话正在执行。' if same else '当前会话空闲或排队中。'))
                        continue
                    if self.active is None:
                        await self.store.conn.execute("UPDATE assistant_inbox SET status='running' WHERE id=%s", (row['id'],))
                        self.active_row = row
                        self.active = asyncio.create_task(self.execute(row))
                await self.send_pending()
                await asyncio.sleep(0.25)
        finally:
            if self.active:
                self.active.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await self.active


async def serve(config):
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
        store = await stack.enter_async_context(operation_storage(config.sessions))
        leader = await (await store.conn.execute('SELECT pg_try_advisory_lock(714160004) AS locked')).fetchone()
        if not leader['locked']:
            raise ConfigError('Another Feishu executor is already running.')
        # Fail before starting reception if migration has not been explicitly run.
        await store.conn.execute('SELECT last_input_id FROM assistant_sessions LIMIT 0')
        await store.conn.execute('SELECT id FROM assistant_inbox LIMIT 0')
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
