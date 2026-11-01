"""共用文本批次登记；事务、连接及提交由调用方持有。"""
from uuid import NAMESPACE_URL, uuid5

from psycopg.types.json import Jsonb

from ai_assistant.interfaces.delivery import metadata


def split_text(text, limit=3000):
    if limit < 1:
        raise ValueError('分段长度必须大于零')
    while text:
        end = min(limit, len(text))
        if end < len(text):
            newline = text.rfind('\n', 0, end)
            if newline >= 0:
                end = newline + 1
        yield text[:end]
        text = text[end:]


async def enqueue_text(conn, settings, row, text, *, purpose='answer', snapshot=None):
    # 整批复用：旧批次即使残缺也不能追加另一版正文。
    await conn.execute('SELECT id FROM assistant_inbox WHERE id=%s FOR UPDATE', (row['id'],))
    existing = await (await conn.execute(
        'SELECT id,body FROM assistant_outbox WHERE input_id=%s AND part>=0 LIMIT 1', (row['id'],))).fetchone()
    if existing and (existing.get('body') or {}).get('delivery', {}).get('version') != 1:
        from ai_assistant.storage.sessions import SessionError
        raise SessionError('不支持的投递格式；未重新登记或发送。')
    if not existing:
        for part, value in enumerate(split_text(text)):
            identity = row['id'] + ':reply:' + str(part)
            await conn.execute('''INSERT INTO assistant_outbox(id,input_id,body,part)
                VALUES(%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING''',
                (str(uuid5(NAMESPACE_URL, identity)), row['id'], Jsonb({
                    'chat_id': row['body']['chat_id'], 'text': value,
                    'presentation': snapshot,
                    'delivery': {**metadata(settings), 'purpose': purpose}}), part))
    await conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (row['id'],))
