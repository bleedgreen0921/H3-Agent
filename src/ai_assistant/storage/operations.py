"""外部效果仓储：幂等 claim、终态回执和只读查询。"""
import asyncio
import hashlib
import json
from uuid import uuid4

from ai_assistant.observability import observed, event


class OperationRepository:
    def __init__(self, conn):
        self.conn = conn
        # psycopg AsyncConnection 不支持无保护并发 execute；写任务本身串行，
        # 但同一任务内的多个工具调用仍可能并发进入回执。
        self.lock = asyncio.Lock()

    @observed("operation.claim",
        lambda self, input_id, principal, tool, arguments, source_key=None: {"input_id": input_id, "tool": tool})
    async def begin(self, input_id, principal, tool, arguments, source_key=None):
        digest = hashlib.sha256(json.dumps(arguments, sort_keys=True, ensure_ascii=False,
                                          separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        ident = str(uuid4())
        async with self.lock:
            row = await (await self.conn.execute('''INSERT INTO assistant_operations
                (id,input_id,principal,tool,args_digest,source_key,status) VALUES(%s,%s,%s,%s,%s,%s,'pending')
                ON CONFLICT DO NOTHING RETURNING *''',
                (ident, input_id, principal, tool, digest, source_key))).fetchone()
            if row:
                event("operation.claimed", operation_id=row["id"], input_id=input_id,
                      tool=tool, status=row["status"])
                return True, row
            if source_key:
                existing = await self._by_source(source_key)
                if existing:
                    return False, existing
            row = await (await self.conn.execute('''SELECT * FROM assistant_operations
                WHERE input_id=%s AND principal=%s AND tool=%s AND args_digest=%s''',
                (input_id, principal, tool, digest))).fetchone()
            return False, row

    async def _by_source(self, source_key):
        return await (await self.conn.execute(
            'SELECT * FROM assistant_operations WHERE source_key=%s', (source_key,))).fetchone()

    async def by_source(self, source_key):
        async with self.lock:
            return await self._by_source(source_key)

    @observed("operation.finish",
        lambda self, ident, status, result=None: {"operation_id": ident, "status": status})
    async def finish(self, ident, status, result=None):
        from psycopg.types.json import Jsonb
        payload = Jsonb(result) if result is not None else None
        async with self.lock:
            await self.conn.execute('''UPDATE assistant_operations SET status=%s,
                result=COALESCE(result || %s, %s, result), updated_at=now()
                WHERE id=%s AND (status='pending' OR status=%s)''',
                (status, payload, payload, ident, status))

    async def list(self, input_id=None):
        if input_id:
            cursor = await self.conn.execute(
                'SELECT * FROM assistant_operations WHERE input_id=%s ORDER BY created_at', (input_id,))
        else:
            cursor = await self.conn.execute(
                'SELECT * FROM assistant_operations ORDER BY created_at DESC LIMIT 100')
        return await cursor.fetchall()
