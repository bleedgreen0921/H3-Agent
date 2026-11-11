"""管理命令回执仓储：命令行幂等和恢复。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandRepository:
    conn: object

    async def get(self, input_id):
        return await (await self.conn.execute(
            'SELECT * FROM assistant_chat_commands WHERE input_id=%s', (input_id,))).fetchone()

    async def begin(self, input_id, binding, action, target=None):
        await self.conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,target)
            VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
            (input_id, binding, action, target))

    async def finish_existing(self, input_id, result):
        await self.conn.execute(
            'UPDATE assistant_chat_commands SET result=%s WHERE input_id=%s', (result, input_id))

    async def upsert_result(self, input_id, binding, action, target, result):
        await self.conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,target,result)
            VALUES(%s,%s,%s,%s,%s) ON CONFLICT(input_id) DO UPDATE SET result=EXCLUDED.result''',
            (input_id, binding, action, target, result))

    async def list(self, ident=None):
        return await (await self.conn.execute("""SELECT input_id,binding,action,target,
            CASE WHEN result IS NULL THEN 'incomplete' ELSE 'completed' END AS status,
            created_at,result AS receipt,'management_command' AS kind FROM assistant_chat_commands
            WHERE (%s::text IS NULL OR input_id=%s) ORDER BY created_at DESC LIMIT 100""",
            (ident, ident))).fetchall()
