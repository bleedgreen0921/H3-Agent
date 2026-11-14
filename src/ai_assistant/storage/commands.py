"""管理命令回执仓储：命令行幂等和恢复。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandRepository:
    """基于调用方连接读写聊天管理命令及其结果。"""
    conn: object

    async def get(self, input_id):
        """按输入 ID 查询已登记命令及结果。"""
        return await (await self.conn.execute(
            'SELECT * FROM assistant_chat_commands WHERE input_id=%s', (input_id,))).fetchone()

    async def begin(self, input_id, binding, action, target=None):
        """登记初始命令；同一输入已有记录时保留原目标。"""
        await self.conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,target)
            VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
            (input_id, binding, action, target))

    async def finish_existing(self, input_id, result):
        """为已有命令保存完成结果。"""
        await self.conn.execute(
            'UPDATE assistant_chat_commands SET result=%s WHERE input_id=%s', (result, input_id))

    async def upsert_result(self, input_id, binding, action, target, result):
        """写入命令结果；冲突时仅更新结果，不改原动作和目标。"""
        await self.conn.execute('''INSERT INTO assistant_chat_commands(input_id,binding,action,target,result)
            VALUES(%s,%s,%s,%s,%s) ON CONFLICT(input_id) DO UPDATE SET result=EXCLUDED.result''',
            (input_id, binding, action, target, result))

    async def list(self, ident=None):
        """返回最近的管理命令回执，按结果是否存在标记状态。"""
        return await (await self.conn.execute("""SELECT input_id,binding,action,target,
            CASE WHEN result IS NULL THEN 'incomplete' ELSE 'completed' END AS status,
            created_at,result AS receipt,'management_command' AS kind FROM assistant_chat_commands
            WHERE (%s::text IS NULL OR input_id=%s) ORDER BY created_at DESC LIMIT 100""",
            (ident, ident))).fetchall()
