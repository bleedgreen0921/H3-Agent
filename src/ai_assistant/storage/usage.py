"""服务已报告的调用用量；独立连接避免随业务 checkpoint 回退。"""
from contextlib import asynccontextmanager

DDL = '''CREATE TABLE IF NOT EXISTS assistant_model_usage (
    call_id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES assistant_sessions(id) ON DELETE CASCADE,
    profile text NOT NULL, role text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    ended_at timestamptz, status text NOT NULL DEFAULT 'running',
    input_tokens bigint, output_tokens bigint, total_tokens bigint
)'''


class UsageStore:
    def __init__(self, conn):
        self.conn = conn

    async def start(self, call_id, session_id, profile, role):
        await self.conn.execute('''INSERT INTO assistant_model_usage(call_id,session_id,profile,role)
            VALUES(%s,%s,%s,%s) ON CONFLICT(call_id) DO NOTHING''',
            (call_id, session_id, profile, role))

    async def finish(self, call_id, status, usage=None):
        usage = usage or {}
        await self.conn.execute('''UPDATE assistant_model_usage SET ended_at=clock_timestamp(),
            status=%s,input_tokens=%s,output_tokens=%s,total_tokens=%s
            WHERE call_id=%s AND ended_at IS NULL''',
            (status, usage.get('input_tokens'), usage.get('output_tokens'),
             usage.get('total_tokens'), call_id))

    async def abandon(self, session_id):
        # 只在持有该会话锁后调用，其他会话正在运行的记录不受影响。
        await self.conn.execute("UPDATE assistant_model_usage SET status='unknown',ended_at=clock_timestamp() WHERE session_id=%s AND ended_at IS NULL", (session_id,))

    async def report(self, session_id):
        session = await (await self.conn.execute('''SELECT s.created_at,v.enabled_at
            FROM assistant_sessions s CROSS JOIN assistant_usage_schema v WHERE s.id=%s''', (session_id,))).fetchone()
        if not session:
            return '当前会话尚无用量统计。'
        rows = await (await self.conn.execute('''SELECT profile,role,count(*) AS calls,
            sum(input_tokens) AS input_tokens,sum(output_tokens) AS output_tokens,
            sum(total_tokens) AS total_tokens,
            count(*) FILTER(WHERE status='running') AS running,
            count(*) FILTER(WHERE input_tokens IS NULL AND output_tokens IS NULL AND total_tokens IS NULL) AS missing,
            count(*) FILTER(WHERE (input_tokens IS NULL OR output_tokens IS NULL OR total_tokens IS NULL)
                AND (input_tokens IS NOT NULL OR output_tokens IS NOT NULL OR total_tokens IS NOT NULL)) AS partial
            FROM assistant_model_usage WHERE session_id=%s GROUP BY profile,role ORDER BY role,profile''', (session_id,))).fetchall()
        lines = ['服务已报告的累计用量', f"会话开始：{session['created_at'].isoformat()}",
                 f"统计覆盖起点：{max(session['created_at'], session['enabled_at']).isoformat()}"]
        def value(v):
            return '未知' if v is None else str(v)
        # 同角色多个配置时提供服务报告值的小计；缺失字段仍保持未知。
        for role_name in ('main', 'summary'):
            selected = [row for row in rows if row['role'] == role_name]
            if len(selected) > 1:
                totals = {key: sum(row[key] for row in selected if row[key] is not None)
                          if any(row[key] is not None for row in selected) else None
                          for key in ('input_tokens', 'output_tokens', 'total_tokens')}
                label = '摘要模型' if role_name == 'summary' else '主模型'
                lines.append(f"{label}累计：输入 {value(totals['input_tokens'])}，输出 {value(totals['output_tokens'])}，合计 {value(totals['total_tokens'])} token。")
        for row in rows:
            role = '摘要模型' if row['role'] == 'summary' else '主模型'
            lines.append(f"{role}（{row['profile']}）：输入 {value(row['input_tokens'])}，输出 {value(row['output_tokens'])}，合计 {value(row['total_tokens'])} token；调用 {row['calls']}，进行中 {row['running']}，未返回用量 {row['missing']}，部分用量 {row['partial']}。")
        if not rows:
            lines.append('尚无模型调用记录。')
        lines.append('缺失用量未填零；不包含服务未报告的 SDK 内部重试消耗，不等同完整账单。')
        return '\n'.join(lines)


@asynccontextmanager
async def usage_storage(settings):
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    conn = await AsyncConnection.connect(settings.database_url.get_secret_value(),
                                         autocommit=True, row_factory=dict_row, connect_timeout=10)
    async with conn:
        yield UsageStore(conn)
