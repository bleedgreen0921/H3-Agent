"""飞书 outbox 单消费者；只发送已登记回复，不执行会话或业务工具。"""
import asyncio
import logging
import time

import httpx

from ai_assistant.observability import observed, event
from ai_assistant.integrations.feishu_files import FeishuRejected

logger = logging.getLogger(__name__)


class FeishuOutboxSender:
    """使用独立存储连接消费飞书回复队列。"""

    def __init__(self, settings, store, api):
        self.settings, self.store, self.api = settings, store, api
        from ai_assistant.storage.queues import InboxRepository, OutboxRepository

        # 优先复用存储对象提供的仓库，否则基于其连接创建。
        self.inbox = getattr(store, 'inbox', None) or InboxRepository(store.conn)
        self.outbox = getattr(store, 'outbox', None) or OutboxRepository(store.conn)

    async def recover(self):
        """启动时处理未确认投递；必须在发送循环之前调用。"""
        # 本地确认中断的投递不推测远端结果，一律 unknown 且不重发。
        await self.outbox.recover_sending()
        await self.outbox.block_unknown_dependents()

    async def send_pending(self):
        """按既有顺序投递已提交消息；发送时不持有事务。"""
        rows = await self.outbox.pending_for_delivery()
        for row in rows:
            await self.send_one(row)

        # 同步更新已满足投递完成条件的输入记录。
        await self.inbox.deliver_completed()

    # 为每次投递记录输入、分段和尝试次数，便于关联观测事件。
    @observed("reply.delivery", lambda self, row: {
        "input_id": row["input_id"], "outbox_id": row["id"], "part": row["part"],
        "attempt": (row.get("body") or {}).get("delivery", {}).get("attempts", 0) + 1}, root=True)
    async def send_one(self, row):
        """领取并投递一个回复分段，依据远端结果更新状态。"""
        settings = self.settings
        from psycopg.types.json import Jsonb
        from ai_assistant.interfaces.delivery import classify, exhausted
        body = row['body'] or {}
        # 新回复显式标记私聊；旧记录从原输入确认，群记录只保留不投递。
        if body.get('group') is not False:
            event('delivery.skipped', status='unsupported')
            return
        delivery = body.get('delivery')
        # 旧记录仅供诊断，升级不能自动补发。
        if not delivery or delivery.get('version') != 1:
            event('delivery.skipped', status='unsupported')
            return

        # 超过尝试次数或截止时间后停止发送，并阻止同一回复的后续分段。
        if exhausted(delivery, settings):
            event("delivery.exhausted", status="unknown")
            async with self.store.conn.transaction():
                await self.outbox.mark_unknown(row['id'])
                await self.outbox.block_pending_after(row['input_id'], row['part'])
            return

        # 先登记本次尝试并原子领取消息；领取失败说明本次不应发送。
        delivery['attempts'] += 1
        if delivery.get('first_terminal_at') is not None:
            delivery['terminal_attempts'] = delivery.get('terminal_attempts', 0) + 1
        delivery['last_outcome'] = 'in_flight'
        attempt_patch = {key: delivery[key] for key in ('attempts', 'terminal_attempts', 'last_outcome') if key in delivery}
        claimed = await self.outbox.claim_for_send(row['id'], attempt_patch)
        if claimed is None:
            return
        row = claimed
        body = row['body']
        delivery = body['delivery']

        # 成功投递全部分段后把本轮工作卡更新为完成终态；卡片是辅助展示，不影响投递状态。
        card_id = body.get('presentation_ref')
        started = time.monotonic()

        # 取消异常直接传播，保留 sending，由下次启动转为 unknown。
        try:
            # 单次超时不能超过回复总截止时间剩余的时长。
            remaining = delivery.get('deadline')
            timeout = min(settings.delivery_attempt_timeout_seconds,
                          max(0, remaining - time.time()) if remaining else settings.delivery_attempt_timeout_seconds)
            async with asyncio.timeout(timeout):
                from ai_assistant.channels.feishu.message import validate_message
                validate_message(body['message'])
                data = await self.api.send_message(body['chat_id'], body['message'], row['id'])
                if not isinstance(data.get('message_id'), str) or not data['message_id']:
                    raise RuntimeError('Missing delivery receipt')
        except Exception as error:
            # 只有可确认的临时错误且未耗尽预算时才重试；未知效果不重发。
            category, diagnostic = classify(error, _classify_feishu)
            delivery['last_error'] = diagnostic
            delivery['last_outcome'] = category
            retry = category == 'temporary' and not exhausted(delivery, settings)
            status = 'pending' if retry else 'rejected' if category in {'rejected', 'temporary'} else 'unknown'
            delay = 1 if delivery.get('terminal_attempts', delivery['attempts']) <= 1 else 3

            # 对 HTTP 429，在剩余截止时间允许时采用服务端建议的等待秒数。
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
                try:
                    requested_delay = float(error.response.headers.get('Retry-After', delay))
                except ValueError:
                    requested_delay = delay
                if 0 <= requested_delay <= max(0, (delivery.get('deadline') or time.time()) - time.time()):
                    delay = requested_delay

            # 保存失败状态；不可重试时阻止后续分段并更新辅助工作卡。
            async with self.store.conn.transaction():
                await self.outbox.mark_retry(row['id'], status,
                    {'last_error': diagnostic, 'last_outcome': category}, delay)
                event('delivery.failed', elapsed_seconds=time.monotonic()-started, status=status, category=category)
                if not retry:
                    await self.outbox.block_pending_after(row['input_id'], row['part'])
            if not retry and card_id:
                from ai_assistant.channels.feishu.progress import mark_delivery_failed
                await mark_delivery_failed(self.api, card_id, row['input_id'])
            return

        # 收到有效远端回执后保存 message_id，再检查整条回复是否已经送达。
        event('delivery.remote_confirmed', remote_id=data.get('message_id'), elapsed_seconds=time.monotonic()-started)
        delivery['last_outcome'] = 'confirmed'
        await self.outbox.mark_delivered(row['id'], data.get('message_id'), delivery)
        event('delivery.persisted', status='delivered')
        completed = await self.inbox.mark_delivered_if_drained(row['input_id'])

        # 最后一个分段完成时记录耗时，并将工作卡更新为终态。
        if completed:
            from datetime import datetime, timezone
            event("reply.delivered", status="delivered", reply_delivery_seconds=max(0,
                (datetime.now(timezone.utc) - completed["received_at"]).total_seconds()))
            if card_id:
                from ai_assistant.channels.feishu.progress import mark_complete_card
                await mark_complete_card(self.api, card_id, row['input_id'],
                    state=(body.get('presentation') or {}).get('state', 'unknown'))

    async def run(self):
        """独占投递连接；未处理异常交给服务监督器关闭服务。"""
        while True:
            await self.send_pending()
            # 短暂等待后再次检查，避免空队列时持续忙轮询。
            await asyncio.sleep(0.25)


def _classify_feishu(error):
    """将已识别的飞书错误码映射为临时、拒绝或未知效果。"""
    if isinstance(error, FeishuRejected):
        return ('temporary' if error.code == 99991400 else
                'rejected' if error.code in {230001, 230006} else 'unknown',
                {'category': 'feishu', 'code': error.code, 'status': error.http_status})
    # 非飞书业务错误交给通用投递错误分类器处理。
    return None
