"""Coalesced output: the model only touches memory; delivery reads durable snapshots."""
import asyncio
import json
from ai_assistant.runtime.outcomes import TERMINAL_STATES
from uuid import uuid5, NAMESPACE_URL


def card(text, state, max_bytes=24576):
    title = {'complete': '已完成', 'partial': '部分完成', 'failed': '未完成', 'unknown': '状态待核实', 'cancelled': '已取消', 'needs_clarification': '需要补充信息'}.get(state, '处理中')
    def build(value):
        # 正文交给飞书原生 Markdown 渲染；plain_text 仅用于不支持富文本的标题。
        return {'schema': '2.0', 'config': {'update_multi': True},
                'header': {'title': {'tag': 'plain_text', 'content': title}},
                'body': {'elements': [{'tag': 'markdown', 'content': value or '正在处理…'}]}}
    def size(value):
        # 包括作为 JSON 字符串嵌入请求后的转义开销。
        return len(json.dumps({'content': json.dumps(build(value), ensure_ascii=False)}, ensure_ascii=False).encode())
    complete = size(text) <= max_bytes
    if complete:
        return build(text), True
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if size(text[:mid] + '\n（预览，完整正文在完成后交付）') <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    return build(text[:lo] + '\n（预览，完整正文在完成后交付）'), False


class ReplyStream:
    def __init__(self, config, row):
        self.config, self.row = config, row
        self.text, self.state, self.version = '正在处理…', 'stage', 0
        self.dirty = asyncio.Event()
        self.stopped = False
        self.task = None

    def emit(self, event):
        kind = event['kind']
        if self.state in TERMINAL_STATES:
            return
        if kind == 'delta':
            self.text += event['text']
        else:
            self.text = event.get('text', '')
        self.state = kind
        self.version += 1
        self.dirty.set()

    async def persist(self, conn):
        from psycopg.types.json import Jsonb
        payload, fits = card(self.text, self.state, self.config.channels.feishu.card_max_bytes)
        if self.state in TERMINAL_STATES and not fits:
            payload, _ = card('正文较长，完整回答见后续消息。', self.state)
        await conn.execute('''INSERT INTO assistant_outbox(id,input_id,body,part,kind,display_version,final_state)
            VALUES(%s,%s,%s,-2,'card',%s,%s) ON CONFLICT(id) DO UPDATE SET
            body=EXCLUDED.body,display_version=assistant_outbox.display_version+1,
            final_state=EXCLUDED.final_state,
            status=CASE WHEN assistant_outbox.status IN ('unknown','rejected','sending') THEN assistant_outbox.status ELSE 'pending' END
            WHERE assistant_outbox.final_state NOT IN ('complete','partial','failed','unknown','cancelled','needs_clarification')
                ''',
            (str(uuid5(NAMESPACE_URL, self.row['id'] + ':card')), self.row['id'],
             Jsonb({'chat_id': self.row['body']['chat_id'], 'card': payload,
                    'final_text': self.text if self.state in TERMINAL_STATES else None}), self.version, self.state))

    async def run(self):
        from ai_assistant.storage.integrations import operation_storage
        async with operation_storage(self.config.sessions) as store:
            while not self.stopped:
                await self.dirty.wait()
                self.dirty.clear()
                await self.persist(store.conn)
                if not self.stopped:
                    await asyncio.sleep(self.config.channels.feishu.stream_interval_seconds)
            if self.dirty.is_set():
                await self.persist(store.conn)

    async def start(self):
        self.emit({'kind': 'stage', 'text': '正在处理…'})
        self.task = asyncio.create_task(self.run())
        return self

    async def close(self):
        self.stopped = True
        self.dirty.set()
        if self.task:
            await self.task
