"""共用 Markdown 批次登记；事务、连接及提交由调用方持有。"""
from uuid import NAMESPACE_URL, uuid5

from psycopg.types.json import Jsonb

from ai_assistant.interfaces.delivery import metadata
from ai_assistant.interfaces.feishu_message import markdown_message, split_markdown


from ai_assistant.observability import observed


@observed("reply.enqueue", lambda conn, settings, row, text, **kw: {"input_id": row["id"]})
async def enqueue_text(conn, settings, row, text, *, purpose='answer', snapshot=None, progress_card_id=None):
    # 接收端之外的第二道边界：历史群请求也不能登记新的回复。
    if (row.get('body') or {}).get('group') is not False:
        from ai_assistant.config import ConfigError
        raise ConfigError('仅支持私聊回复，未登记群消息投递。')
    # 整批复用：旧批次即使残缺也不能追加另一版正文。
    await conn.execute('SELECT id FROM assistant_inbox WHERE id=%s FOR UPDATE', (row['id'],))
    existing = await (await conn.execute(
        'SELECT id,body FROM assistant_outbox WHERE input_id=%s AND part>=0 LIMIT 1', (row['id'],))).fetchone()
    if existing and (existing.get('body') or {}).get('delivery', {}).get('version') != 1:
        from ai_assistant.storage.sessions import SessionError
        raise SessionError('不支持的投递格式；未重新登记或发送。')
    if not existing:
        for part, value in enumerate(split_markdown(text, row['body']['chat_id'])):
            identity = row['id'] + ':reply:' + str(part)
            body = {'chat_id': row['body']['chat_id'], 'group': False,
                    'message': markdown_message(value),
                    'presentation': snapshot,
                    'delivery': {**metadata(settings), 'purpose': purpose}}
            # 最终答案全部分段送达后由发送器撤回工作卡；卡片 ID 随批次持久化。
            if progress_card_id:
                body['progress_card_id'] = progress_card_id
            await conn.execute('''INSERT INTO assistant_outbox(id,input_id,body,part)
                VALUES(%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING''',
                (str(uuid5(NAMESPACE_URL, identity)), row['id'], Jsonb(body), part))
    await conn.execute("UPDATE assistant_inbox SET status='completed' WHERE id=%s", (row['id'],))
