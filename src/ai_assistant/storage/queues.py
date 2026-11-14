"""Inbox/outbox 仓储：所有队列 SQL 与 JSON body 操作集中在此。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class InboxRepository:
    """管理渠道输入的领取、处理完成和投递完成状态。"""
    conn: object

    async def received(self, limit=100):
        return await (await self.conn.execute(
            "SELECT * FROM assistant_inbox WHERE status='received' ORDER BY received_at,id LIMIT %s",
            (limit,))).fetchall()

    async def recoverable(self):
        # 只恢复当前支持格式的处理中输入，不自动重跑业务。
        return await (await self.conn.execute(
            "SELECT i.* FROM assistant_inbox i WHERE i.status='running' AND "
            "(i.body->>'delivery_version'='1' OR EXISTS(SELECT 1 FROM assistant_outbox o "
            "WHERE o.input_id=i.id AND o.body->'delivery'->>'version'='1')) "
            "ORDER BY i.received_at,i.id")).fetchall()

    async def get(self, input_id):
        return await (await self.conn.execute(
            'SELECT * FROM assistant_inbox WHERE id=%s', (input_id,))).fetchone()

    async def claim(self, input_id):
        # 标记当前投递格式及运行状态；调度并发由上层控制。
        await self.conn.execute(
            "UPDATE assistant_inbox SET status='running',body=jsonb_set(body,'{delivery_version}','1'::jsonb) WHERE id=%s",
            (input_id,))

    async def mark_completed(self, input_id):
        await self.conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (input_id,))

    async def set_progress_card(self, input_id, card_id):
        await self.conn.execute(
            "UPDATE assistant_inbox SET body=jsonb_set(body,'{presentation_ref}',to_jsonb(%s::text)) "
            "WHERE id=%s AND body IS NOT NULL", (card_id, input_id))

    async def block_running(self, input_id, binding):
        await self.conn.execute(
            "UPDATE assistant_inbox SET status='completed' WHERE id=%s AND binding=%s AND status='running'",
            (input_id, binding))

    async def deliver_completed(self):
        # 已登记回复全部送达后清除输入正文，标记投递完成。
        await self.conn.execute('''UPDATE assistant_inbox i SET body=NULL,status='delivered'
            WHERE i.status='completed' AND EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id)
            AND NOT EXISTS(SELECT 1 FROM assistant_outbox o WHERE o.input_id=i.id AND o.status!='delivered')''')

    async def mark_delivered_if_drained(self, input_id):
        return await (await self.conn.execute('''UPDATE assistant_inbox SET body=NULL,status='delivered'
            WHERE id=%s AND status='completed' AND NOT EXISTS(
                SELECT 1 FROM assistant_outbox WHERE input_id=%s AND status!='delivered')
            RETURNING received_at''', (input_id, input_id))).fetchone()


@dataclass(frozen=True)
class OutboxRepository:
    """管理有序回复分段和单段投递状态。"""
    conn: object

    async def parts(self, input_id):
        return await (await self.conn.execute(
            'SELECT * FROM assistant_outbox WHERE input_id=%s ORDER BY part', (input_id,))).fetchall()

    async def first(self, input_id):
        return await (await self.conn.execute(
            'SELECT id,body FROM assistant_outbox WHERE input_id=%s AND part>=0 LIMIT 1',
            (input_id,))).fetchone()

    async def lock_inbox(self, input_id):
        # 调用方事务持有输入行锁，协调同一输入的批次登记。
        await self.conn.execute('SELECT id FROM assistant_inbox WHERE id=%s FOR UPDATE', (input_id,))

    async def enqueue_parts(self, input_id, parts):
        from psycopg.types.json import Jsonb
        for part, outbox_id, body in parts:
            # psycopg 不会自动把 dict 适配为 jsonb，必须显式包装，否则登记回复时抛 ProgrammingError。
            await self.conn.execute('''INSERT INTO assistant_outbox(id,input_id,body,part)
                VALUES(%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING''',
                (outbox_id, input_id, Jsonb(body), part))

    async def pending_for_delivery(self):
        # 仅返回到达重试时间、且之前分段均已送达的私聊候选。
        return await (await self.conn.execute(
            "SELECT o.* FROM assistant_outbox o WHERE o.status='pending' "
            "AND COALESCE(o.body->>'group','false')!='true' "
            "AND NOT EXISTS(SELECT 1 FROM assistant_inbox i WHERE i.id=o.input_id AND i.body->>'group'='true') "
            "AND o.next_attempt_at<=now() AND NOT EXISTS(SELECT 1 FROM assistant_outbox p "
            "WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status!='delivered') "
            "ORDER BY o.created_at,o.input_id,o.part,o.id")).fetchall()

    async def recover_sending(self):
        # 重启后无法确认远端效果，发送中记录转 unknown 而非重发。
        await self.conn.execute(
            "UPDATE assistant_outbox SET status='unknown' WHERE status='sending' AND body->'delivery'->>'version'='1'")

    async def block_unknown_dependents(self):
        # 前序分段结果未知或失败时阻止依赖它的后续分段。
        await self.conn.execute(
            "UPDATE assistant_outbox o SET status='blocked' WHERE o.status='pending' AND o.part>=0 "
            "AND o.body->'delivery'->>'version'='1' AND EXISTS(SELECT 1 FROM assistant_outbox p "
            "WHERE p.input_id=o.input_id AND p.part>=0 AND p.part<o.part AND p.status IN ('unknown','blocked','rejected'))")

    async def block_pending(self, input_id):
        await self.conn.execute(
            "UPDATE assistant_outbox SET status='blocked' WHERE input_id=%s AND status='pending'",
            (input_id,))

    async def mark_unknown(self, outbox_id):
        await self.conn.execute("UPDATE assistant_outbox SET status='unknown' WHERE id=%s", (outbox_id,))

    async def block_pending_after(self, input_id, part):
        await self.conn.execute(
            "UPDATE assistant_outbox SET status='blocked' WHERE input_id=%s AND part>%s AND status='pending'",
            (input_id, part))

    async def claim_for_send(self, outbox_id, patch):
        # 仅从 pending 原子切换到 sending，同时记录本次尝试信息。
        from psycopg.types.json import Jsonb
        return await (await self.conn.execute(
            "UPDATE assistant_outbox SET status='sending',body=jsonb_set(body,'{delivery}',body->'delivery' || %s) "
            "WHERE id=%s AND status='pending' RETURNING *", (Jsonb(patch), outbox_id))).fetchone()

    async def mark_retry(self, outbox_id, status, patch, delay):
        from psycopg.types.json import Jsonb
        await self.conn.execute(
            "UPDATE assistant_outbox SET status=%s,body=jsonb_set(body,'{delivery}',body->'delivery' || %s),"
            "next_attempt_at=now()+%s*interval '1 second' WHERE id=%s",
            (status, Jsonb(patch), delay, outbox_id))

    async def mark_delivered(self, outbox_id, remote_id, delivery):
        # 保留远端消息 ID 和投递元数据，清除已投递正文。
        from psycopg.types.json import Jsonb
        await self.conn.execute(
            "UPDATE assistant_outbox SET status='delivered',remote_id=%s,body=%s WHERE id=%s",
            (remote_id, Jsonb({'delivery': delivery}), outbox_id))

    async def pending_parts_after(self, input_id, part):
        return await (await self.conn.execute(
            "SELECT 1 FROM assistant_outbox WHERE input_id=%s AND part>%s AND status='pending' LIMIT 1",
            (input_id, part))).fetchone()

    async def unsupported(self, input_id):
        row = await (await self.conn.execute(
            "SELECT 1 FROM assistant_outbox WHERE input_id=%s AND part>=0 "
            "AND (body->'delivery'->>'version')!='1' LIMIT 1", (input_id,))).fetchone()
        return row is not None
