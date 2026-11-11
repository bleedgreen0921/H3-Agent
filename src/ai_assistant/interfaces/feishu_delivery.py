"""飞书 outbox 单消费者；只发送已登记回复，不执行会话或业务工具。"""
import asyncio
import logging
import time

import httpx

from ai_assistant.observability import observed, event

logger = logging.getLogger(__name__)


class FeishuOutboxSender:
    def __init__(self, settings, store, api):
        self.settings, self.store, self.api = settings, store, api

    async def recover(self):
        """启动时处理未确认投递；必须在发送循环之前调用。"""
        # 本地确认中断的投递不推测远端结果，一律 unknown 且不重发。
        await self.store.conn.execute("UPDATE assistant_outbox SET status='unknown' WHERE status='sending' AND body->'delivery'->>'version'='1'")
        await self.store.conn.execute("UPDATE assistant_outbox o SET status='blocked' WHERE o.status='pending' AND o.part>=0 AND o.body->'delivery'->>'version'='1' AND EXISTS(SELECT 1 FROM assistant_outbox p WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status IN ('unknown','blocked','rejected'))")

    async def send_pending(self):
        """按既有顺序投递已提交消息；发送时不持有事务。"""
        rows = await (await self.store.conn.execute("SELECT o.* FROM assistant_outbox o WHERE o.status='pending' AND COALESCE(o.body->>'group','false')!='true' AND NOT EXISTS(SELECT 1 FROM assistant_inbox i WHERE i.id=o.input_id AND i.body->>'group'='true') AND o.next_attempt_at<=now() AND NOT EXISTS(SELECT 1 FROM assistant_outbox p WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status!='delivered') ORDER BY o.created_at,o.input_id,o.part,o.id")).fetchall()
        for row in rows:
            await self.send_one(row)

        await self.store.conn.execute("""UPDATE assistant_inbox i SET body=NULL,status='delivered'
            WHERE i.status='completed' AND EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id)
            AND NOT EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id
                AND o.status!='delivered')""")

    @observed("reply.delivery", lambda self, row: {
        "input_id": row["input_id"], "outbox_id": row["id"], "part": row["part"],
        "attempt": (row.get("body") or {}).get("delivery", {}).get("attempts", 0) + 1}, root=True)
    async def send_one(self, row):
        settings = self.settings
        from psycopg.types.json import Jsonb
        from ai_assistant.interfaces.delivery import classify, exhausted
        body = row['body'] or {}
        # 新回复显式标记私聊；旧记录从原输入确认，群记录只保留不投递。
        if body.get('group') is True:
            event('delivery.skipped', status='group_unsupported')
            return
        if body.get('group') is not False:
            source = await (await self.store.conn.execute(
                "SELECT EXISTS(SELECT 1 FROM assistant_inbox WHERE id=%s AND body->>'group'='true') AS is_group",
                (row['input_id'],))).fetchone()
            if source and source.get('is_group') is True:
                event('delivery.skipped', status='group_unsupported')
                return
        delivery = body.get('delivery')
        # 旧记录仅供诊断，升级不能自动补发。
        if not delivery or delivery.get('version') != 1:
            event('delivery.skipped', status='unsupported')
            return
        if exhausted(delivery, settings):
            event("delivery.exhausted", status="unknown")
            async with self.store.conn.transaction():
                await self.store.conn.execute("UPDATE assistant_outbox SET status='unknown' WHERE id=%s", (row['id'],))
                await self.store.conn.execute("UPDATE assistant_outbox SET status='blocked' WHERE input_id=%s AND part>%s AND status='pending'", (row['input_id'], row['part']))
            return
        delivery['attempts'] += 1
        if delivery.get('first_terminal_at') is not None:
            delivery['terminal_attempts'] = delivery.get('terminal_attempts', 0) + 1
        delivery['last_outcome'] = 'in_flight'
        attempt_patch = {key: delivery[key] for key in ('attempts', 'terminal_attempts', 'last_outcome') if key in delivery}
        claimed = await (await self.store.conn.execute(
            "UPDATE assistant_outbox SET status='sending',body=jsonb_set(body,'{delivery}',body->'delivery' || %s) WHERE id=%s AND status='pending' RETURNING *",
            (Jsonb(attempt_patch), row['id']))).fetchone()
        if claimed is None:
            return
        row = claimed
        body = row['body']
        delivery = body['delivery']
        # 成功投递全部分段后撤回本轮工作卡；卡片是辅助展示，不影响投递状态。
        card_id = body.get('progress_card_id')
        started = time.monotonic()
        # 取消异常直接传播，保留 sending，由下次启动转为 unknown。
        try:
            remaining = delivery.get('deadline')
            timeout = min(settings.delivery_attempt_timeout_seconds,
                          max(0, remaining - time.time()) if remaining else settings.delivery_attempt_timeout_seconds)
            async with asyncio.timeout(timeout):
                from ai_assistant.interfaces.feishu_message import validate_message
                validate_message(body['message'])
                data = await self.api.send_message(body['chat_id'], body['message'], row['id'])
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
                event('delivery.failed', elapsed_seconds=time.monotonic()-started, status=status, category=category)
                if not retry:
                    await self.store.conn.execute("UPDATE assistant_outbox SET status='blocked' WHERE input_id=%s AND part>%s AND status='pending'", (row['input_id'], row['part']))
            if not retry and card_id:
                from ai_assistant.interfaces.feishu_progress import mark_delivery_failed
                await mark_delivery_failed(self.api, card_id, row['input_id'])
            return
        event('delivery.remote_confirmed', remote_id=data.get('message_id'), elapsed_seconds=time.monotonic()-started)
        delivery['last_outcome'] = 'confirmed'
        await self.store.conn.execute("UPDATE assistant_outbox SET status='delivered',remote_id=%s,body=%s WHERE id=%s",
                                      (data.get('message_id'), Jsonb({'delivery': delivery}), row['id']))
        event('delivery.persisted', status='delivered')
        completed = await (await self.store.conn.execute('''UPDATE assistant_inbox SET body=NULL,status='delivered'
            WHERE id=%s AND status='completed' AND NOT EXISTS(SELECT 1 FROM assistant_outbox WHERE input_id=%s AND status!='delivered') RETURNING received_at''',
            (row['input_id'], row['input_id']))).fetchone()

        if completed:
            from datetime import datetime, timezone
            event("reply.delivered", status="delivered", reply_delivery_seconds=max(0,
                (datetime.now(timezone.utc) - completed["received_at"]).total_seconds()))
            if card_id and settings.progress_card_recall:
                from ai_assistant.interfaces.feishu_progress import recall_card
                await recall_card(self.api, card_id, row['input_id'])

    async def run(self):
        """独占投递连接；未处理异常交给服务监督器关闭服务。"""
        while True:
            await self.send_pending()
            await asyncio.sleep(0.25)
