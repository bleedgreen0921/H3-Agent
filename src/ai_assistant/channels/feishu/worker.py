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

from ai_assistant.observability import event, span, ObservabilityConfig
from ai_assistant.channels.feishu.progress import ProgressCard, close_stale_card
from ai_assistant.interfaces.commands import CommandContext, CommandRegistry
from ai_assistant.channels.feishu.commands import default_registry

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
    capability_cache: dict | None = None
    commands: object = None




def caller_for(config, row):
    settings = config.integrations.feishu
    body = row['body']
    if body.get('owner_open_id') != settings.owner_open_id:
        raise ConfigError('Untrusted inbox owner.')
    if body.get('group') is not False:
        raise ConfigError('仅支持主人私聊，群消息不再处理。')
    return Caller('feishu_private', True, row['binding'], scope='private')


class BotWorker:
    def __init__(self, config, store, resources, api=None):
        self.config, self.store, self.resources, self.api = config, store, resources, api
        self.commands = getattr(resources, 'commands', None) or default_registry()
        if store is not None and getattr(store, 'conn', None) is not None:
            from ai_assistant.storage.queues import InboxRepository, OutboxRepository
            self.inbox = getattr(store, 'inbox', None) or InboxRepository(store.conn)
            self.outbox = getattr(store, 'outbox', None) or OutboxRepository(store.conn)
        else:
            self.inbox = self.outbox = None
        self.active = None
        self.active_row = None

    async def enqueue_reply(self, row, text, *, conn=None, purpose='answer', snapshot=None,
                            presentation_ref=None):
        from ai_assistant.channels.feishu.outbox import enqueue_text

        async def enqueue(connection):
            async with connection.transaction():
                await enqueue_text(connection, self.config.channels.feishu, row, text,
                    purpose=purpose, snapshot=snapshot, presentation_ref=presentation_ref)
        if conn is not None:
            await enqueue(conn)
        else:
            async with operation_storage(self.config.sessions) as reply_store:
                await enqueue(reply_store.conn)

    async def _remember_progress_card(self, input_id, card_id):
        """卡片 ID 随输入持久化，服务重启后才能关闭陈旧工作卡。"""
        async with operation_storage(self.config.sessions) as store:
            await store.inbox.set_progress_card(input_id, card_id)

    async def _finish_progress(self, progress, state, *, recall, phase=None):
        """辅助卡片的终态清理；取消期间也等待网络调用超时结束。"""
        if progress is None:
            return
        task = asyncio.create_task(progress.finish(state, recall=recall, phase=phase))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            with suppress(Exception):
                await task
            raise
        except Exception as error:
            # 卡片是辅助展示，任何清理错误都不能改写业务结果。
            logger.warning('Progress card cleanup failed: input_id=%s category=%s',
                           getattr(progress, 'input_id', '-'), type(error).__name__)

    async def recover_reply(self, row, *, fallback=None):
        """投递恢复只读 saver 和索引，不构建 Agent 或模型。"""
        from ai_assistant.runtime.presentation import project
        from ai_assistant.storage.sessions import session_storage

        saved = await self.outbox.parts(row['id'])
        # 旧记录仅供诊断，升级不能自动补发。
        if any((item.get('body') or {}).get('delivery', {}).get('version') != 1 for item in saved):
            logger.warning('不支持的投递格式：input_id=%s；未恢复或发送。', row['id'])
            return
        # 当前已登记批次复用，不重切或重发；未知格式已在上方拒绝。
        if saved:
            await self.inbox.mark_completed(row['id'])
            return
        async with session_storage(self.config.sessions) as repo:
            session = await repo.binding_session(row['binding'])
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
        from ai_assistant.storage.sessions import SessionRepository
        result = await SessionRepository(self.store.conn, None).binding_session_id(row['binding'])
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
        # 回复已登记后结束执行 trace；摘要独立记录，资源和会话锁仍由同一作用域清理。
        async with AsyncExitStack() as stack:
            with span("request.execute", input_id=row["id"]):
                session = await self._execute(row, stack, recover=recover)
            if session is not None:
                await session.after_reply()

    async def _execute(self, row, stack, *, recover=False):
        from datetime import datetime, timezone
        from time import monotonic
        preparation_started = monotonic()
        if row.get("received_at"):
            event("request.dequeued", queue_wait_seconds=max(0, (datetime.now(timezone.utc) - row["received_at"]).total_seconds()))
        caller = caller_for(self.config, row)
        text = row['body']['text'].strip()
        command, value = self.commands.parse(text)
        if command is None and value and value[0] == 'error':
            await self.enqueue_reply(row, value[1])
            return None
        result = None
        # 延迟工作卡只服务真实等待；即时指令和关闭状态不建卡。
        progress = None
        channel = getattr(getattr(self.config, 'channels', None), 'feishu', None)
        show_progress = command is None or command.progress
        if (not recover and show_progress and channel is not None and self.api is not None
                and channel.progress_card_delay_seconds > 0):
            progress = ProgressCard(self.api, channel,
                chat_id=row['body']['chat_id'], input_id=row['id'],
                on_created=self._remember_progress_card)
            progress.start()
        if command is not None:
            result = await command.handler(CommandContext(
                worker=self, row=row, caller=caller, text=text,
                value=value, recover=recover))
            if result.reply:
                await self.enqueue_reply(row, result.reply)
            if not result.start_turn:
                await self._finish_progress(progress, 'complete', recall=True)
                return None
            text = result.turn_text or text
            skill_ids = result.skill_ids or None
        else:
            skill_ids = None
        try:
            if recover:
                await self.recover_reply(row)
                return
            session_id = await self.binding_session(row)
            session = await stack.enter_async_context(assistant_session(
                self.config, session_id=session_id, caller=caller, resources=self.resources))
            await self.save_binding(row, session.session_id, session.repository.conn)
            event("request.prepared", session_id=session.session_id, preparation_seconds=monotonic() - preparation_started)
            turn_kwargs = dict(skill_ids=skill_ids, input_id=row['id'], user_source=text)
            if progress is not None:
                turn_kwargs['on_event'] = progress.on_event
            result = await session.run_turn(text, **turn_kwargs)
            from ai_assistant.runtime.presentation import project, saved_body
            display = project(result.result, saved_body(result.result, result.messages))
            await self.enqueue_reply(row, display.text, snapshot=display.snapshot(),
                presentation_ref=progress.card_id if progress is not None else None)
            # 成功时保留卡片到最终文本送达，再由发送器撤回。
            await self._finish_progress(progress, 'complete', recall=False)
            return session
        except BaseException as error:
            state = 'cancelled' if isinstance(error, asyncio.CancelledError) else 'failed'
            with suppress(Exception):
                await self._finish_progress(progress, state, recall=True)
            raise

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
        """仅恢复已保存回复；启动发送循环前由 serve 顺序调用。"""
        rows = await self.inbox.recoverable()
        for row in rows:
            # 旧群记录保留，但不恢复执行或补发回复。
            if (row.get('body') or {}).get('group') is not False:
                continue
            # 服务重启后不再有进程更新工作卡，先标记待核实再进入只读恢复。
            await close_stale_card(self.api, (row.get('body') or {}).get('presentation_ref'), row['id'])
            await self.execute(row, recover=True)

    async def recover_rag(self, row):
        """失败后的补交只读业务回执；渠道仍不可用时留待重启恢复。"""
        try:
            await self.execute(row, recover=True)
        except Exception as exc:
            logger.error('RAG receipt recovery pending: input_id=%s error_type=%s',
                         row['id'], type(exc).__name__)

    async def run(self):
        try:
            while True:
                if self.active and self.active.done():
                    try:
                        self.active.result()
                    except asyncio.CancelledError:
                        spec, _ = self.commands.parse(self.active_row['body']['text'])
                        if spec is not None and spec.recover_rag:
                            # 取消可能发生在提交之后；只读业务回执，保留已经提交的 ID。
                            await self.recover_rag(self.active_row)
                        else:
                            await self.recover_reply(self.active_row)
                    except Exception as exc:
                        logger.error('Feishu turn failed: input_id=%s error_type=%s',
                                     self.active_row['id'], type(exc).__name__)
                        spec, _ = self.commands.parse(self.active_row['body']['text'])
                        if spec is not None and spec.recover_rag:
                            await self.recover_rag(self.active_row)
                        else:
                            # checkpoint 提交也可能已成功但确认失败，先只读检查，不能覆盖原答案。
                            await self.recover_reply(self.active_row, fallback=failure_message(exc))
                    self.active = self.active_row = None
                rows = await self.inbox.received()
                for row in rows:
                    if (row.get('body') or {}).get('group') is not False:
                        continue
                    text = row['body']['text'].strip()
                    spec, value = self.commands.parse(text)
                    turn_command = bool(spec and (spec.turn or (
                        spec.turn_check is not None and spec.turn_check(value))))
                    if spec is not None and spec.preempt:
                        try:
                            caller_for(self.config, row)
                            await self.stop_for_command(row)
                            await self.execute(row)
                        except Exception as exc:
                            await self.enqueue_reply(row, failure_message(exc))
                        continue
                    if not turn_command:
                        try:
                            await self.inbox.claim(row['id'])
                            await self.execute(row)
                        except Exception as exc:
                            if spec is not None and spec.recover_rag:
                                await self.recover_rag(row)
                            else:
                                await self.enqueue_reply(row, failure_message(exc))
                        continue
                    if self.active is None:
                        await self.inbox.claim(row['id'])
                        self.active_row = row
                        self.active = asyncio.create_task(self.execute(row))
                await asyncio.sleep(0.25)
        finally:
            if self.active:
                self.active.cancel()
                from ai_assistant.cancellation import _drain_worker
                await _drain_worker(self.active)
