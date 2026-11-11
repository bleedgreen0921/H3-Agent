"""长任务工作卡：尽最大努力原地展示进度，不承载业务结果。

最终答案仍走持久化的文本 outbox；卡片的创建、更新或撤回失败只记录安全事件，
不改变本轮结果、失败回执、取消和恢复语义。卡片是“过程展示”，不是业务回执。
"""
import asyncio
import inspect
import json
from contextlib import suppress
from time import monotonic
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

from ai_assistant.observability import event

# 飞书卡片消息上限 30 KB；预留余量，避免样式和转义放大实际体积。
MAX_CARD_BYTES = 24_000
# 单条消息更新频控 5 QPS；每 0.5 秒最多一次，留出余量。
TICK_SECONDS = 0.5
MIN_UPDATE_SECONDS = 0.5
CALL_TIMEOUT = 5.0
PHASE_CHARS = 40

STATE_STYLES = {
    'working': ('blue', '处理中'),
    'complete': ('green', '已完成'),
    'failed': ('red', '未完成'),
    'cancelled': ('grey', '已取消'),
    'unknown': ('orange', '结果待核实'),
}

TERMINAL_PHASES = {
    'complete': '完整答案正在发送…',
    'failed': '本轮未完成；已保留必要回执，未自动重试。',
    'cancelled': '已停止当前执行；已提交的后台任务继续执行。',
    'unknown': '服务重启，结果未确认；正在恢复已保存的回复。',
}


def friendly_tool(name):
    text = (name or '').strip()
    if not text:
        return '未知工具'
    if len(text) > PHASE_CHARS:
        return text[:PHASE_CHARS - 1] + '…'
    return text


def _elapsed_text(seconds):
    total = max(0, int(seconds))
    return f'{total // 60:02d}:{total % 60:02d}'


def _card_json(title, template, summary, preview, phase):
    elements = []
    if preview:
        elements.append({'tag': 'markdown', 'element_id': 'preview', 'content': preview})
    elements.append({'tag': 'markdown', 'element_id': 'phase', 'content': phase})
    return {
        'schema': '2.0',
        'config': {'update_multi': True, 'summary': {'content': summary}},
        'header': {'template': template, 'title': {'tag': 'plain_text', 'content': title}},
        'body': {'elements': elements},
    }


def _encoded_size(card):
    return len(json.dumps(card, ensure_ascii=False).encode('utf-8'))


def _fit_card(card_factory, preview):
    """按序列化体积收缩预览尾部，保证卡片不超过上限。"""
    while True:
        card = card_factory(preview)
        if _encoded_size(card) <= MAX_CARD_BYTES or len(preview) <= 200:
            return card
        # 优先保留最新内容：从头部按比例裁剪。
        preview = preview[max(1, len(preview) // 8):]


def _closed_fence(text):
    """预览中的未闭合代码围栏补一个结束标记，避免整段渲染为代码。"""
    if text.count('```') % 2:
        return text + '\n```'
    return text


def _safe_markdown(text):
    """避免模型正文中的 <at> 标签被卡片解析为真实人员资源。"""
    return text.replace('<at ', '&lt;at ')


class ProgressCard:
    """一次业务轮次对应的工作卡；未创建或失败时所有方法安全空转。"""

    def __init__(self, api, settings, *, chat_id, input_id, on_created=None, started_at=None):
        self.api = api
        self.settings = settings
        self.chat_id = chat_id
        self.input_id = input_id
        self.on_created = on_created
        self.started = monotonic() if started_at is None else started_at
        self.card_id = None
        self.preview = ''
        self.phase = '正在准备上下文…'
        self.state = 'working'
        self._done = False
        self._closed = False
        self._truncated = False
        self._dirty = False
        self._last_patch = 0.0
        self._last_second = -1
        self._failures = 0
        self._updates_disabled = False
        self._done_event = asyncio.Event()
        self._task = None

    @property
    def enabled(self):
        return self.api is not None and self.settings.progress_card_delay_seconds > 0

    def start(self):
        if not self.enabled:
            self._closed = True
            return
        self._task = asyncio.create_task(self._run())

    def on_event(self, item):
        """运行时辅助回调：只更新内存状态，绝不向业务轮次抛出异常。"""
        if self._closed or not isinstance(item, dict):
            return
        try:
            kind = item.get('event')
            if kind == 'on_model_start':
                self.phase = '正在思考…'
            elif kind == 'on_tool_start':
                self.phase = '正在调用工具：' + friendly_tool(item.get('name'))
            elif kind == 'on_tool_end':
                self.phase = '整理工具结果…'
            elif kind == 'on_tool_error':
                self.phase = '工具返回错误，正在处理…'
            elif kind == 'on_text':
                text = item.get('text')
                limit = self.settings.progress_card_preview_chars
                if not isinstance(text, str) or not text:
                    return
                self.phase = '正在生成回答…'
                if limit:
                    combined = self.preview + text
                    if len(combined) > limit:
                        self._truncated = True
                        combined = combined[-limit:]
                    self.preview = combined
            else:
                return
            self._dirty = True
        except Exception:
            # 进度辅助不能影响业务轮次。
            pass

    async def finish(self, state, *, recall=False, phase=None):
        """提交终态；网络调用有超时，调用方可安全地在取消清理中等待。"""
        if self._closed:
            return
        self._closed = True
        self.state = state
        if phase is not None:
            self.phase = phase
        elif state in TERMINAL_PHASES:
            self.phase = TERMINAL_PHASES[state]
        self._done = True
        self._done_event.set()
        if self._task is not None:
            with suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(self._task, CALL_TIMEOUT + self.settings.progress_card_delay_seconds)
        if self.card_id is not None:
            await self._patch_safely(force=True)
        if self.card_id is not None and recall and self.settings.progress_card_recall:
            await self._recall()

    async def _run(self):
        try:
            if self.settings.progress_card_delay_seconds > 0:
                try:
                    await asyncio.wait_for(self._done_event.wait(),
                                           self.settings.progress_card_delay_seconds)
                    return
                except TimeoutError:
                    pass
            if self._done or not await self._create():
                return
            while not self._done:
                try:
                    await asyncio.wait_for(self._done_event.wait(), TICK_SECONDS)
                except TimeoutError:
                    pass
                if self._done:
                    break
                await self._patch_safely()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            event('progress.failed', input_id=self.input_id, status='disabled',
                  category=type(error).__name__)
            self._updates_disabled = True

    def _render_preview(self):
        if not self.preview:
            return ''
        text = _closed_fence(self.preview)
        if self._truncated:
            text = '…（前文省略）\n' + text
        return _safe_markdown(text)

    def _card(self):
        template, label = STATE_STYLES.get(self.state, STATE_STYLES['working'])
        elapsed = _elapsed_text(monotonic() - self.started)
        title = f'{label} · {elapsed}' if self.state in {'working', 'complete'} else label
        summary = '正在处理' if self.state == 'working' else label
        preview = self._render_preview()

        def factory(value):
            return _card_json(title, template, summary, value, self.phase)

        return _fit_card(factory, preview)

    async def _create(self):
        try:
            async with asyncio.timeout(CALL_TIMEOUT):
                data = await self.api.request(
                    'POST', '/im/v1/messages',
                    params={'receive_id_type': 'chat_id'},
                    json={'receive_id': self.chat_id, 'msg_type': 'interactive',
                          'content': json.dumps(self._card(), ensure_ascii=False),
                          'uuid': str(uuid5(NAMESPACE_URL, 'assistant-progress:' + self.input_id))})
        except Exception as error:
            event('progress.failed', input_id=self.input_id, status='create_failed',
                  category=type(error).__name__)
            return False
        card_id = data.get('message_id') if isinstance(data, dict) else None
        if not isinstance(card_id, str) or not card_id:
            event('progress.failed', input_id=self.input_id, status='create_failed',
                  category='MissingReceipt')
            return False
        self.card_id = card_id
        event('progress.created', input_id=self.input_id, status='working')
        if self.on_created is not None:
            try:
                returned = self.on_created(self.input_id, card_id)
                if inspect.isawaitable(returned):
                    await returned
            except Exception:
                event('progress.failed', input_id=self.input_id, status='persist_failed')
        return True

    async def _patch_safely(self, *, force=False):
        if self.card_id is None or self.api is None:
            return
        now = monotonic()
        elapsed = int(now - self.started)
        if not force:
            if self._updates_disabled or now - self._last_patch < MIN_UPDATE_SECONDS:
                return
            if not self._dirty and elapsed == self._last_second:
                return
        self._dirty = False
        self._last_patch = now
        self._last_second = elapsed
        try:
            async with asyncio.timeout(CALL_TIMEOUT):
                await self.api.request(
                    'PATCH', '/im/v1/messages/' + quote(self.card_id, safe=''),
                    json={'content': json.dumps(self._card(), ensure_ascii=False)})
            self._failures = 0
        except Exception as error:
            self._failures += 1
            if self._failures >= 3:
                self._updates_disabled = True
            event('progress.failed', input_id=self.input_id, status='update_failed',
                  category=type(error).__name__)

    async def _recall(self):
        card_id, self.card_id = self.card_id, None
        if card_id is None:
            return
        try:
            async with asyncio.timeout(CALL_TIMEOUT):
                await self.api.request('DELETE', '/im/v1/messages/' + quote(card_id, safe=''))
            event('progress.recalled', input_id=self.input_id, status='recalled')
        except Exception as error:
            event('progress.failed', input_id=self.input_id, status='recall_failed',
                  category=type(error).__name__)


def _simple_card(title, template, summary, text):
    return _card_json(title, template, summary, '', text)


async def recall_card(api, card_id, input_id):
    """最终文本全部分段送达后撤回工作卡；失败只记事件。"""
    if api is None or not card_id:
        return
    try:
        async with asyncio.timeout(CALL_TIMEOUT):
            await api.request('DELETE', '/im/v1/messages/' + quote(card_id, safe=''))
        event('progress.recalled', input_id=input_id, status='recalled')
    except Exception as error:
        event('progress.failed', input_id=input_id, status='recall_failed',
              category=type(error).__name__)


async def close_stale_card(api, card_id, input_id):
    """重启恢复时把陈旧工作卡标记为待核实并撤回；失败不影响恢复。"""
    if api is None or not card_id:
        return
    try:
        async with asyncio.timeout(CALL_TIMEOUT):
            await api.request('PATCH', '/im/v1/messages/' + quote(card_id, safe=''),
                              json={'content': json.dumps(_simple_card(
                                  '结果待核实', 'orange', '结果待核实',
                                  TERMINAL_PHASES['unknown']), ensure_ascii=False)})
    except Exception as error:
        event('progress.failed', input_id=input_id, status='stale_mark_failed',
              category=type(error).__name__)
    try:
        async with asyncio.timeout(CALL_TIMEOUT):
            await api.request('DELETE', '/im/v1/messages/' + quote(card_id, safe=''))
        event('progress.stale_closed', input_id=input_id, status='closed')
    except Exception as error:
        event('progress.failed', input_id=input_id, status='stale_close_failed',
              category=type(error).__name__)


async def mark_delivery_failed(api, card_id, input_id):
    """最终文本投递永久失败时把卡片改为失败态并保留，不重发答案。"""
    if api is None or not card_id:
        return
    try:
        async with asyncio.timeout(CALL_TIMEOUT):
            await api.request('PATCH', '/im/v1/messages/' + quote(card_id, safe=''),
                              json={'content': json.dumps(_simple_card(
                                  '结果发送失败', 'red', '结果发送失败',
                                  '答案已生成，但文本投递未确认。请查看操作回执；未自动重发。'),
                                  ensure_ascii=False)})
    except Exception as error:
        event('progress.failed', input_id=input_id, status='delivery_failed_card',
              category=type(error).__name__)
