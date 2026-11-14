"""PostgreSQL advisory lock helpers with bounded handover waiting."""
import asyncio
from time import monotonic


async def try_advisory_lock(conn, key, *, wait_seconds=0, interval=0.2):
    """尝试获取 advisory lock；只在调用方指定的时间窗口内等待。

    不开启事务，不执行任何业务操作；连接错误由调用方转换为存储错误。
    """
    deadline = monotonic() + max(0.0, float(wait_seconds))
    interval = max(0.01, float(interval))
    while True:
        # 会话级 advisory lock 绑定数据库连接；未成功时仅在窗口内轮询。
        row = await (await conn.execute(
            'SELECT pg_try_advisory_lock(%s) AS locked', (key,))).fetchone()
        locked = row.get('locked') if hasattr(row, 'get') else row['locked']
        if locked:
            return True
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(interval, remaining))
