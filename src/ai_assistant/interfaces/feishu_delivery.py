"""飞书 outbox 单消费者；只发送已登记回复，不执行会话或业务工具。"""
import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)


class FeishuOutboxSender:
    def __init__(self, settings, store, api):
        self.settings, self.store, self.api = settings, store, api

    async def recover(self):
        """启动时处理未确认投递；必须在发送循环之前调用。"""
        # 卡片机制已删除：本地确认中断的投递不推测远端结果，一律 unknown 且不重发。
        await self.store.conn.execute("UPDATE assistant_outbox SET status='unknown' WHERE status='sending' AND body->'delivery'->>'version'='1'")
        await self.store.conn.execute("UPDATE assistant_outbox o SET status='blocked' WHERE o.status='pending' AND o.part>=0 AND o.body->'delivery'->>'version'='1' AND EXISTS(SELECT 1 FROM assistant_outbox p WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status IN ('unknown','blocked','rejected'))")

    async def send_pending(self):
        """按既有顺序投递已提交消息；发送时不持有事务。"""
        rows = await (await self.store.conn.execute("SELECT o.* FROM assistant_outbox o WHERE o.status='pending' AND o.kind='text' AND o.next_attempt_at<=now() AND NOT EXISTS(SELECT 1 FROM assistant_outbox p WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status!='delivered') ORDER BY o.created_at,o.input_id,o.part,o.id")).fetchall()
        from psycopg.types.json import Jsonb
        from ai_assistant.interfaces.delivery import classify, exhausted
        settings = self.settings
        for row in rows:
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
            # 取消异常直接传播，保留 sending，由下次启动转为 unknown。
            try:
                remaining = delivery.get('deadline')
                timeout = min(settings.delivery_attempt_timeout_seconds,
                              max(0, remaining - time.time()) if remaining else settings.delivery_attempt_timeout_seconds)
                async with asyncio.timeout(timeout):
                    if 'message' in body:
                        from ai_assistant.interfaces.feishu_message import validate_message
                        validate_message(body['message'])
                        data = await self.api.send_message(body['chat_id'], body['message'], row['id'])
                    else:
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
                    logger.warning('Delivery operation=send_message elapsed=%.3f diagnostic=%s',
                                   time.monotonic()-started, diagnostic)
                    if not retry:
                        await self.store.conn.execute("UPDATE assistant_outbox SET status='blocked' WHERE input_id=%s AND part>%s AND status='pending'", (row['input_id'], row['part']))
                continue
            logger.info('Delivery operation=send_message elapsed=%.3f outcome=confirmed',
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

    async def run(self):
        """独占投递连接；未处理异常交给服务监督器关闭服务。"""
        while True:
            await self.send_pending()
            await asyncio.sleep(0.25)
