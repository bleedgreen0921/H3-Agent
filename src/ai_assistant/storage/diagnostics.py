"""只读诊断仓储：不改变队列或命令状态。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticsRepository:
    conn: object

    async def replies(self):
        return await (await self.conn.execute('''SELECT id,input_id,status,remote_id,part,created_at,
            body->'delivery' AS delivery,
            (body->'delivery'->>'version'='1') IS NOT TRUE AS unsupported_format,
            'channel_reply' AS kind FROM assistant_outbox ORDER BY created_at DESC LIMIT 100''')).fetchall()

    async def commands(self, ident=None):
        return await (await self.conn.execute("""SELECT input_id,binding,action,target,
            CASE WHEN result IS NULL THEN 'incomplete' ELSE 'completed' END AS status,
            created_at,result AS receipt,'management_command' AS kind FROM assistant_chat_commands
            WHERE (%s::text IS NULL OR input_id=%s) ORDER BY created_at DESC LIMIT 100""",
            (ident, ident))).fetchall()

    async def inspect(self, ident):
        row = await (await self.conn.execute("""SELECT id,input_id,tool,principal,status,
            created_at,updated_at,result->>'version' AS receipt_version,
            COALESCE(result->>'verification','unverified') AS verification,
            (COALESCE(result->>'verification','unverified')!='passed') AS needs_verification,
            'tool_operation' AS kind FROM assistant_operations WHERE id=%s""", (ident,))).fetchone()
        if row is not None:
            return row
        row = await (await self.conn.execute("""SELECT id,input_id,status,remote_id,part,created_at,
            body->'delivery' AS delivery,
            (body->'delivery'->>'version'='1') IS NOT TRUE AS unsupported_format,
            'channel_reply' AS kind FROM assistant_outbox WHERE id=%s""", (ident,))).fetchone()
        if row is not None:
            return row
        rows = await self.commands(ident)
        return rows[0] if rows else None
