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
        self.disabled = False

    async def record(self, call_id, session_id, profile, role, started_at, status, usage):
        if self.disabled or not usage:
            return
        try:
            # 一次完成记录；相同调用 ID 重放不会重复累计。
            await self.conn.execute('''INSERT INTO assistant_model_usage
                (call_id,session_id,profile,role,started_at,ended_at,status,
                 input_tokens,output_tokens,total_tokens)
                VALUES(%s,%s,%s,%s,%s,clock_timestamp(),%s,%s,%s,%s)
                ON CONFLICT(call_id) DO NOTHING''',
                (call_id, session_id, profile, role, started_at, status,
                 usage.get('input_tokens'), usage.get('output_tokens'), usage.get('total_tokens')))
        except Exception:
            self.disabled = True
            raise

    async def report(self, session_id):
        rows = await (await self.conn.execute('''SELECT profile,role,count(*) AS calls,
            sum(input_tokens) AS input_tokens,sum(output_tokens) AS output_tokens,
            sum(total_tokens) AS total_tokens,
            count(*) FILTER(WHERE ended_at IS NULL OR status IN ('running','unknown')) AS unfinished,
            count(*) FILTER(WHERE input_tokens IS NULL AND output_tokens IS NULL AND total_tokens IS NULL) AS missing,
            count(*) FILTER(WHERE (input_tokens IS NULL OR output_tokens IS NULL OR total_tokens IS NULL)
                AND (input_tokens IS NOT NULL OR output_tokens IS NOT NULL OR total_tokens IS NOT NULL)) AS partial
            FROM assistant_model_usage WHERE session_id=%s GROUP BY profile,role ORDER BY role,profile''', (session_id,))).fetchall()
        lines = ['已保存的服务报告用量']
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
            lines.append(f"{role}（{row['profile']}）：输入 {value(row['input_tokens'])}，输出 {value(row['output_tokens'])}，合计 {value(row['total_tokens'])} token；调用 {row['calls']}，旧记录未结束 {row['unfinished']}，未返回用量 {row['missing']}，部分用量 {row['partial']}。")
        if not rows:
            lines.append('暂无已保存的用量记录。')
        lines.append('尚在执行的调用未落库。未返回用量、写入失败、响应返回后落库前崩溃及服务未报告的内部重试都可能造成遗漏；缺失值不填零，不等同完整账单。')
        return '\n'.join(lines)


@asynccontextmanager
async def postgres_usage_storage(settings):
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    conn = await AsyncConnection.connect(settings.database_url.get_secret_value(),
                                         autocommit=True, row_factory=dict_row, connect_timeout=2)
    try:
        await conn.execute("SET statement_timeout = '2000ms'")
        yield UsageStore(conn)
    finally:
        try:
            await conn.close()
        except Exception as error:
            import logging
            logging.getLogger(__name__).warning('Usage unavailable: phase=close category=%s', type(error).__name__)


def usage_storage(settings):
    from ai_assistant.storage.backends import backend_factory
    if settings is None:
        from ai_assistant.config import ConfigError
        raise ConfigError('存储需要显式配置 [sessions]。')
    return backend_factory('usage', settings.backend)(settings)
