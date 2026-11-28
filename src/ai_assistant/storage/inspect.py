"""只读持久化检查：会话、checkpoint、展示消息和操作回执。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
from langchain_core.messages import BaseMessage

from ai_assistant.config import ConfigError
from ai_assistant.storage.sessions import (
    SessionNotFoundError, session_storage, session_uuid,
)
from ai_assistant.storage.web import WebMessageStore

PREVIEW_CHARS = 500
TOOL_PREVIEW_CHARS = 800
MAX_CHANNEL_PREVIEW = 240
_SECRET_KEYS = frozenset({
    'api_key', 'apikey', 'authorization', 'password', 'secret',
    'cookie', 'set-cookie', 'access_token', 'refresh_token',
})


class StorageInspectError(RuntimeError):
    """检查失败的安全摘要；不包含数据库诊断和原始内容。"""


def _clip(text, limit):
    text = '' if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + '…'


def _is_secret_key(key):
    lowered = str(key).lower()
    return lowered in _SECRET_KEYS or any(
        lowered.endswith('_' + item) for item in _SECRET_KEYS)


def _message_text(message):
    if hasattr(message, 'text'):
        try:
            return message.text or ''
        except Exception:
            pass
    content = getattr(message, 'content', '')
    return content if isinstance(content, str) else str(content)


def message_view(message: BaseMessage, *, full=False):
    """把 LangChain 消息压缩成稳定的调试视图。"""
    limit = PREVIEW_CHARS if not full else 10**9
    tool_limit = TOOL_PREVIEW_CHARS if not full else 10**9
    text = _message_text(message)
    view = {
        'type': getattr(message, 'type', type(message).__name__),
        'id': getattr(message, 'id', None),
        'name': getattr(message, 'name', None),
        'text': _clip(text, tool_limit if getattr(message, 'type', '') == 'tool' else limit),
        'tool_call_id': getattr(message, 'tool_call_id', None),
        'status': getattr(message, 'status', None),
    }
    tool_calls = getattr(message, 'tool_calls', None) or []
    if tool_calls:
        view['tool_calls'] = jsonable(tool_calls, full=full)
    artifact = getattr(message, 'artifact', None)
    if artifact:
        view['artifact'] = jsonable(artifact, full=full)
    additional = getattr(message, 'additional_kwargs', None) or {}
    if additional:
        view['additional_kwargs'] = jsonable(
            {key: value for key, value in additional.items()
             if key in {'input_id', 'user_source', 'lc_source', 'source'}},
            full=full)
    return view


def redact(value):
    if isinstance(value, dict):
        return {
            str(key): ('[已隐藏]' if _is_secret_key(key) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def _jsonable(value, *, full=False, depth=0):
    limit = PREVIEW_CHARS if not full else 10**9
    if depth > 8:
        return '[层级过深]'
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(value, limit)
    if isinstance(value, bytes):
        return {'type': 'bytes', 'size': len(value)}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, BaseMessage):
        return message_view(value, full=full)
    if isinstance(value, dict):
        return {
            str(key): ('[已隐藏]' if _is_secret_key(key)
                       else _jsonable(item, full=full, depth=depth + 1))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, full=full, depth=depth + 1) for item in value]
    if hasattr(value, 'model_dump'):
        try:
            return _jsonable(value.model_dump(mode='json'), full=full, depth=depth + 1)
        except Exception:
            pass
    if is_dataclass(value):
        return _jsonable(asdict(value), full=full, depth=depth + 1)
    return _clip(str(value), limit)


def jsonable(value, *, full=False, depth=0):
    return redact(_jsonable(value, full=full, depth=depth))


def _approx_size(value):
    try:
        return len(json.dumps(_jsonable(value), ensure_ascii=False, default=str))
    except Exception:
        return len(str(value))


def _channel_preview(value):
    if isinstance(value, BaseMessage):
        return _clip(_message_text(value), MAX_CHANNEL_PREVIEW)
    try:
        return _clip(json.dumps(jsonable(value), ensure_ascii=False, default=str),
                     MAX_CHANNEL_PREVIEW)
    except Exception:
        return _clip(str(value), MAX_CHANNEL_PREVIEW)


def _channel_summary(name, value):
    return {
        'name': name,
        'type': type(value).__name__,
        'approx_bytes': _approx_size(value),
        'preview': _channel_preview(value),
    }


def _config(thread_id, checkpoint_id=None, checkpoint_ns=''):
    configurable = {'thread_id': str(thread_id), 'checkpoint_ns': checkpoint_ns}
    if checkpoint_id:
        configurable['checkpoint_id'] = str(checkpoint_id)
    return {'configurable': configurable}


@asynccontextmanager
async def storage_inspector(settings):
    """打开只读检查连接；所有检查方法只执行 SELECT 和原生 Saver 读取。"""
    if settings is None:
        raise ConfigError('存储检查需要显式配置 [sessions]。')
    async with session_storage(settings) as repo:
        try:
            await repo.execute("SET statement_timeout = '5000ms'")
            # 检查器只允许读取；即使代码误改也不会在本连接写业务表。
            await repo.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        except Exception:
            pass
        yield StorageInspector(repo, settings)


class StorageInspector:
    """只读检查器；不打开模型、不执行工具、不修改会话。"""

    def __init__(self, repository, settings=None):
        self.repo = repository
        self.settings = settings
        self.web = WebMessageStore(settings) if settings is not None else None

    async def _session_row(self, session_id):
        ident = session_uuid(session_id)
        row = await (await self.repo.execute(
            '''SELECT id,workspace,thread_id,thread_ids,complete,status,
                      format_version,title,last_input_id,
                      created_at,updated_at
               FROM assistant_sessions WHERE id=%s''', (ident,))).fetchone()
        if not row:
            raise SessionNotFoundError('指定会话不存在。')
        return dict(row)

    @staticmethod
    def _threads(row):
        threads = list(row.get('thread_ids') or [])
        if row.get('thread_id') and row['thread_id'] not in threads:
            threads.append(row['thread_id'])
        return threads

    async def _count(self, sql, params=()):
        try:
            row = await (await self.repo.execute(sql, params)).fetchone()
            return int(row['n']) if row else 0
        except Exception:
            return 0

    async def _input_ids(self, session_id, threads):
        result = set()
        try:
            rows = await (await self.repo.execute(
                '''SELECT DISTINCT input_id FROM assistant_web_messages
                   WHERE session_id=%s AND input_id IS NOT NULL''',
                (str(session_id),))).fetchall()
            result.update(row['input_id'] for row in rows if row.get('input_id'))
        except Exception:
            pass
        if threads:
            try:
                rows = await (await self.repo.execute(
                    '''SELECT DISTINCT metadata->>'input_id' AS input_id
                       FROM checkpoints WHERE thread_id = ANY(%s)''',
                    (threads,))).fetchall()
                result.update(row['input_id'] for row in rows if row.get('input_id'))
            except Exception:
                pass
        return sorted(result)

    async def _operations(self, session_id, threads, limit=100):
        input_ids = await self._input_ids(session_id, threads)
        if not input_ids:
            return [], 0
        try:
            rows = await (await self.repo.execute(
                '''SELECT id,input_id,principal,tool,args_digest,source_key,
                          status,result,created_at,updated_at
                   FROM assistant_operations
                   WHERE input_id = ANY(%s)
                   ORDER BY created_at DESC,id DESC LIMIT %s''',
                (input_ids, limit + 1))).fetchall()
        except Exception:
            return [], False
        return [
            {**dict(row), 'result': jsonable(row.get('result'))}
            for row in rows[:limit]
        ], len(rows) > limit

    async def list_sessions(self, limit=50):
        rows = await self.repo.list_admin(0, max(1, min(int(limit), 100)))
        return [dict(row) for row in rows]

    async def session(self, session_id):
        row = await self._session_row(session_id)
        threads = self._threads(row)
        thread_rows = []
        if threads:
            counts = await (await self.repo.execute(
                '''SELECT thread_id,count(*) AS n FROM checkpoints
                   WHERE thread_id = ANY(%s) GROUP BY thread_id''',
                (threads,))).fetchall()
            by_thread = {item['thread_id']: int(item['n']) for item in counts}
        else:
            by_thread = {}
        for thread in threads:
            thread_rows.append({
                'thread_id': thread,
                'current': thread == row.get('thread_id'),
                'checkpoint_count': by_thread.get(thread, 0),
            })
        web_count = await self._count(
            'SELECT count(*) AS n FROM assistant_web_messages WHERE session_id=%s',
            (str(session_id),))
        input_ids = await self._input_ids(session_id, threads)
        operations_count = 0
        if input_ids:
            operations_count = await self._count(
                'SELECT count(*) AS n FROM assistant_operations WHERE input_id = ANY(%s)',
                (input_ids,))
        return {
            'session': row,
            'threads': thread_rows,
            'counts': {
                'checkpoints': sum(by_thread.values()),
                'web_messages': web_count,
                'operations': operations_count,
            },
            'input_ids': input_ids,
        }

    async def messages(self, session_id, limit=100):
        row = await self._session_row(session_id)
        limit = max(1, min(int(limit), 200))
        if self.web is None:
            return {'items': [], 'truncated': False}
        rows = await self.web.list(str(row['id']), limit=limit + 1)
        items = []
        for item in rows[:limit]:
            data = dict(item)
            data['text'] = _clip(data.get('text'), PREVIEW_CHARS)
            data['attachment_ids'] = jsonable(data.get('attachment_ids'))
            items.append(data)
        return {'items': items, 'truncated': len(rows) > limit}

    async def operations(self, session_id, limit=100):
        row = await self._session_row(session_id)
        threads = self._threads(row)
        items, truncated = await self._operations(
            row['id'], threads, max(1, min(int(limit), 200)))
        return {'items': items, 'truncated': truncated}

    async def usage(self, session_id):
        await self._session_row(session_id)
        try:
            rows = await (await self.repo.execute(
                '''SELECT profile,role,count(*) AS calls,
                          sum(input_tokens) AS input_tokens,
                          sum(output_tokens) AS output_tokens,
                          sum(total_tokens) AS total_tokens,
                          count(*) FILTER (WHERE ended_at IS NULL
                              OR status IN ('running','unknown')) AS unfinished,
                          count(*) FILTER (WHERE input_tokens IS NULL
                              AND output_tokens IS NULL AND total_tokens IS NULL) AS missing
                   FROM assistant_model_usage WHERE session_id=%s
                   GROUP BY profile,role ORDER BY role,profile''',
                (str(session_id),))).fetchall()
        except Exception:
            return {'available': False, 'items': []}
        return {'available': True, 'items': [dict(row) for row in rows]}

    async def checkpoints(self, session_id, *, thread_id=None, before=None, limit=20):
        row = await self._session_row(session_id)
        threads = self._threads(row)
        thread = str(thread_id or row['thread_id'])
        if thread not in threads:
            raise StorageInspectError('指定线程不属于当前会话。')
        limit = max(1, min(int(limit), 100))
        before_config = _config(thread, before) if before else None
        items = []
        try:
            async for checkpoint in self.repo.saver.alist(
                    _config(thread), before=before_config, limit=limit):
                items.append(self._checkpoint_summary(checkpoint))
        except Exception as error:
            raise StorageInspectError('Checkpoint 历史读取失败。') from error
        return {
            'thread_id': thread,
            'items': items,
            'next_before': items[-1]['checkpoint_id'] if len(items) == limit else None,
        }

    def _checkpoint_summary(self, checkpoint):
        configurable = (checkpoint.config or {}).get('configurable', {})
        metadata = checkpoint.metadata or {}
        values = (checkpoint.checkpoint or {}).get('channel_values', {}) or {}
        turn_result = values.get('turn_result') or {}
        parent = (checkpoint.parent_config or {}).get('configurable', {})
        channels = [
            _channel_summary(name, value)
            for name, value in sorted(values.items())
            if not name.startswith('branch:to:') and name != '__pregel_tasks'
        ]
        return {
            'thread_id': configurable.get('thread_id'),
            'checkpoint_ns': configurable.get('checkpoint_ns', ''),
            'checkpoint_id': configurable.get('checkpoint_id'),
            'parent_checkpoint_id': parent.get('checkpoint_id'),
            'step': metadata.get('step'),
            'source': metadata.get('source'),
            'input_id': metadata.get('input_id'),
            'execution_thread_id': metadata.get('execution_thread_id'),
            'assistant_phase': metadata.get('assistant_phase'),
            'ts': (checkpoint.checkpoint or {}).get('ts'),
            'pending_writes': len(checkpoint.pending_writes or []),
            'turn_status': turn_result.get('status') if isinstance(turn_result, dict) else None,
            'channels': channels,
        }

    async def checkpoint(self, session_id, checkpoint_id, *, thread_id=None,
                         channels=None, full=False):
        row = await self._session_row(session_id)
        threads = self._threads(row)
        thread = str(thread_id or row['thread_id'])
        if thread not in threads:
            raise StorageInspectError('指定线程不属于当前会话。')
        config = _config(thread, checkpoint_id)
        try:
            checkpoint = await self.repo.saver.aget_tuple(config)
        except Exception as error:
            raise StorageInspectError('Checkpoint 读取失败。') from error
        if checkpoint is None:
            raise StorageInspectError('指定 checkpoint 不存在。')
        values = (checkpoint.checkpoint or {}).get('channel_values', {}) or {}
        selected = list(channels or ())
        if not selected:
            selected = [
                'messages', 'planning', 'task_results', 'turn_result',
                'session_state', 'execution', 'diagnostics', 'effect_facts',
                'active_skills', 'conversation_title', 'completed_turns',
                'last_counted_input_id', 'input_id',
            ]
        payload = {}
        for name in selected:
            if name not in values:
                continue
            value = values[name]
            if name == 'messages':
                payload[name] = [message_view(item, full=full)
                                 for item in (value or [])]
            else:
                payload[name] = jsonable(value, full=full)
        summaries = [
            _channel_summary(name, value)
            for name, value in sorted(values.items())
            if not name.startswith('branch:to:') and name != '__pregel_tasks'
        ]
        writes = [
            {
                'task_id': item[0],
                'channel': item[1],
                'value': jsonable(item[2], full=full),
            }
            for item in (checkpoint.pending_writes or [])
            if isinstance(item, tuple) and len(item) >= 3
        ]
        blob_sizes = []
        try:
            rows = await (await self.repo.execute(
                '''SELECT channel,version,type,octet_length(blob) AS bytes
                   FROM checkpoint_blobs
                   WHERE thread_id=%s AND checkpoint_ns=%s
                   ORDER BY channel,version''',
                (thread, (checkpoint.config or {}).get(
                    'configurable', {}).get('checkpoint_ns', '')))).fetchall()
            versions = (checkpoint.checkpoint or {}).get('channel_versions', {}) or {}
            blob_sizes = [
                dict(item) for item in rows
                if str(item.get('version')) == str(versions.get(item.get('channel')))
            ]
        except Exception:
            blob_sizes = []
        configurable = (checkpoint.config or {}).get('configurable', {})
        parent = (checkpoint.parent_config or {}).get('configurable', {})
        metadata = checkpoint.metadata or {}
        return {
            'thread_id': thread,
            'checkpoint_ns': configurable.get('checkpoint_ns', ''),
            'checkpoint_id': configurable.get('checkpoint_id'),
            'parent_checkpoint_id': parent.get('checkpoint_id'),
            'step': metadata.get('step'),
            'source': metadata.get('source'),
            'input_id': metadata.get('input_id'),
            'execution_thread_id': metadata.get('execution_thread_id'),
            'assistant_phase': metadata.get('assistant_phase'),
            'ts': (checkpoint.checkpoint or {}).get('ts'),
            'pending_writes': writes,
            'channels': summaries,
            'channel_values': payload,
            'blob_sizes': blob_sizes,
            'full': bool(full),
        }

    async def turns(self, session_id, limit=20):
        row = await self._session_row(session_id)
        threads = self._threads(row)
        limit = max(1, min(int(limit), 50))
        grouped = {}
        for thread in threads:
            try:
                snapshots = [
                    item async for item in self.repo.saver.alist(
                        _config(thread), limit=max(50, limit * 4))
                ]
            except Exception:
                snapshots = []
            for checkpoint in snapshots:
                metadata = checkpoint.metadata or {}
                input_id = metadata.get('input_id')
                if not input_id:
                    # update/中间 checkpoint 可能没有输入关联，不伪造成 unknown 轮次。
                    continue
                values = (checkpoint.checkpoint or {}).get('channel_values', {}) or {}
                turn_result = values.get('turn_result') or {}
                item = grouped.setdefault(input_id, {
                    'input_id': input_id,
                    'status': None,
                    'checkpoints': 0,
                    'threads': set(),
                    'latest_checkpoint_id': None,
                    'latest_checkpoint_ts': None,
                    'assistant_phase': None,
                })
                item['checkpoints'] += 1
                item['threads'].add(thread)
                ts = (checkpoint.checkpoint or {}).get('ts')
                if not item['latest_checkpoint_ts'] or str(ts or '') > str(item['latest_checkpoint_ts']):
                    item['latest_checkpoint_ts'] = ts
                    item['latest_checkpoint_id'] = (
                        checkpoint.config or {}).get('configurable', {}).get('checkpoint_id')
                if metadata.get('assistant_phase'):
                    item['assistant_phase'] = metadata.get('assistant_phase')
                if isinstance(turn_result, dict) and turn_result.get('status'):
                    item['status'] = turn_result['status']
        web_counts = {}
        if self.web is not None:
            try:
                messages = await self.web.list(str(row['id']), limit=200)
                for message in messages:
                    input_id = message.get('input_id') or 'unknown'
                    web_counts[input_id] = web_counts.get(input_id, 0) + 1
                    if input_id in grouped and not grouped[input_id]['status']:
                        grouped[input_id]['status'] = message.get('status')
            except Exception:
                pass
        input_ids = sorted(grouped)
        operation_counts = {}
        if input_ids:
            try:
                rows = await (await self.repo.execute(
                    '''SELECT input_id,count(*) AS n FROM assistant_operations
                       WHERE input_id = ANY(%s) GROUP BY input_id''',
                    (input_ids,))).fetchall()
                operation_counts = {item['input_id']: int(item['n']) for item in rows}
            except Exception:
                pass
        items = []
        for input_id, item in grouped.items():
            data = dict(item)
            data['threads'] = sorted(data['threads'])
            data['web_messages'] = web_counts.get(input_id, 0)
            data['operations'] = operation_counts.get(input_id, 0)
            items.append(data)
        items.sort(key=lambda item: str(item.get('latest_checkpoint_ts') or ''), reverse=True)
        return {'items': items[:limit], 'truncated': len(items) > limit}

    async def health(self, session_id):
        row = await self._session_row(session_id)
        threads = self._threads(row)
        warnings = []
        complete = row.get('complete')
        complete_valid = None
        if complete:
            try:
                checkpoint = await self.repo.saver.aget_tuple(complete)
                values = (checkpoint.checkpoint or {}).get('channel_values', {}) if checkpoint else {}
                from ai_assistant.runtime.history import saved_turn
                complete_valid = bool(checkpoint and saved_turn(values, threads=threads))
            except Exception:
                complete_valid = False
            if not complete_valid:
                warnings.append('完成点指向的 checkpoint 不存在、不可读或缺少可交付正文。')
            elif checkpoint and checkpoint.pending_writes:
                warnings.append(
                    f'完成点 checkpoint 仍包含 {len(checkpoint.pending_writes)} 条待合并写入。')
        latest = []
        for thread in threads:
            try:
                checkpoint = await self.repo.saver.aget_tuple(_config(thread))
            except Exception:
                checkpoint = None
            if checkpoint is None:
                warnings.append(f'线程 {thread} 没有可读 checkpoint。')
                continue
            values = (checkpoint.checkpoint or {}).get('channel_values', {}) or {}
            latest.append({
                'thread_id': thread,
                'checkpoint_id': (checkpoint.config or {}).get('configurable', {}).get('checkpoint_id'),
                'ts': (checkpoint.checkpoint or {}).get('ts'),
                'pending_writes': len(checkpoint.pending_writes or []),
                'messages': len(values.get('messages') or []),
                'turn_status': (values.get('turn_result') or {}).get('status')
                if isinstance(values.get('turn_result'), dict) else None,
            })
            if checkpoint.pending_writes:
                warnings.append(f'线程 {thread} 存在待合并写入（{len(checkpoint.pending_writes)} 条）。')
        if len(threads) > 1:
            warnings.append(f'会话包含 {len(threads)} 个执行分支（可能来自失败恢复）。')
        for item in latest:
            if item.get('turn_status') in {'failed', 'unknown', 'cancelled'}:
                warnings.append(
                    f"线程 {item['thread_id']} 最新状态为 {item['turn_status']}，"
                    "请结合操作回执核实。")
        if row.get('status') == 'running':
            warnings.append('会话状态为 running；可能正在执行，也可能是崩溃残留。')
        elif row.get('status') == 'interrupted':
            warnings.append('上一轮未完成；下一轮会创建恢复分支，不会自动重放旧操作。')
        web_count = await self._count(
            'SELECT count(*) AS n FROM assistant_web_messages WHERE session_id=%s',
            (str(session_id),))
        operations_pending = operations_unknown = 0
        input_ids = await self._input_ids(session_id, threads)
        if input_ids:
            try:
                rows = await (await self.repo.execute(
                    '''SELECT status,count(*) AS n FROM assistant_operations
                       WHERE input_id = ANY(%s) GROUP BY status''',
                    (input_ids,))).fetchall()
                by_status = {item['status']: int(item['n']) for item in rows}
                operations_pending = by_status.get('pending', 0)
                operations_unknown = by_status.get('unknown', 0)
                if operations_pending or operations_unknown:
                    warnings.append(
                        f'存在待确认操作回执（pending={operations_pending}，unknown={operations_unknown}）。')
            except Exception:
                pass
        return {
            'session_id': str(row['id']),
            'status': row.get('status'),
            'complete': complete,
            'complete_valid': complete_valid,
            'threads': threads,
            'latest_checkpoints': latest,
            'web_messages': web_count,
            'operations_pending': operations_pending,
            'operations_unknown': operations_unknown,
            'warnings': warnings,
        }
