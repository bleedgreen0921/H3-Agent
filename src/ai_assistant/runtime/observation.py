"""Per-turn native callbacks; no persistent event bus or message copy."""
import asyncio
import logging
from dataclasses import dataclass, field
from time import monotonic
from datetime import datetime, timezone

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from ai_assistant.agents.context import SummaryCallLimitError
from ai_assistant.llm.requests import response_usage


@dataclass
class TurnUsage:
    """当前轮的调用次数、服务用量和耗时指标。"""
    main_model_calls: int = 0
    summary_model_calls: int = 0
    tool_calls: int = 0
    token_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    unreported_model_calls: int = 0
    first_status_seconds: float | None = None
    first_body_seconds: float | None = None
    total_seconds: float = 0
    tool_seconds: float = 0
    max_input_tokens: int = 0


class TurnObserver(AsyncCallbackHandler):
    """从原生回调收集调用指标及工具返回的证据，不复制事件流。"""
    raise_error = True
    run_inline = True

    def __init__(self, summary_limit, model_profile='', usage_store=None, session_id=None, *, input_id=None, execution_thread_id=None):
        self.store, self.session_id = usage_store, session_id
        self.input_id = input_id
        self.execution_thread_id = execution_thread_id
        self._model_times = {}
        self._model_parents = {}
        self.summary_limit = summary_limit
        self._completed = set()
        self.model_profile = model_profile
        self.usage = TurnUsage()
        self.sources = []
        self._seen = set()
        self._models = {}
        self._tool_started = {}
        self._model_started = set()
        self._tool_seen = set()
        self._parents = {}
        self._tool_scopes = set()
        self._model_calls = {}
        self._model_call_index = 0
        self._tool_names = {}
        self._last_tool = None
        # 规划任务可并行调用模型；服务用量使用单连接，写入必须串行。
        self._record_lock = asyncio.Lock()

    async def _record(self, *args):
        """串行写入服务报告用量；统计故障不阻断业务。"""
        if self.store is None:
            return
        try:
            options = {}
            if getattr(self.store, "capabilities", None) and self.store.capabilities.linked:
                options = dict(input_id=self.input_id, execution_thread_id=self.execution_thread_id,
                               parent_run_id=self._model_parents.get(args[0]))
            async with self._record_lock:
                await self.store.record(*args, **options)
        except Exception as error:
            self.store = None
            logging.getLogger(__name__).warning('Usage unavailable: input_id=%s session_id=%s phase=%s category=%s',
                self.input_id, self.session_id, 'record', type(error).__name__)

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        # 按 run ID 去重，并区分普通模型调用与摘要调用。
        if run_id in self._model_started or run_id in self._completed:
            return
        self._model_started.add(run_id)
        self._model_times[run_id] = monotonic()
        self._model_parents[run_id] = kwargs.get("parent_run_id")
        self._event("model.start", run_id=run_id, parent_run_id=kwargs.get("parent_run_id"))
        metadata = metadata or {}
        if metadata.get('lc_source') == 'summarization':
            if self.usage.summary_model_calls >= self.summary_limit:
                raise SummaryCallLimitError('Summary call limit reached.')
            self.usage.summary_model_calls += 1
            self._event('summary.start', run_id=run_id,
                kind=metadata.get('assistant_summary_kind', 'compression'))
        else:
            self.usage.main_model_calls += 1
        from ai_assistant.agents.context import count_tokens
        self.usage.max_input_tokens = max(self.usage.max_input_tokens, *(count_tokens(m) for m in messages))
        profile = metadata.get('assistant_model_profile', self.model_profile)
        role = 'summary' if metadata.get('lc_source') == 'summarization' else 'main'
        self._models[run_id] = (profile, role, datetime.now(timezone.utc))
        self._model_call_index += 1
        self._model_calls[run_id] = (self._model_call_index, role, self._last_tool)

    async def on_llm_end(self, response, *, run_id, **kwargs):
        # 只累计服务实际报告的用量；缺失单独计数。
        if run_id in self._completed:
            return
        self._completed.add(run_id)
        self._model_finished(run_id, "succeeded")
        profile, role, started_at = self._models.pop(run_id, (self.model_profile, 'main', datetime.now(timezone.utc)))
        message = response.generations[0][0].message if response.generations and response.generations[0] else None
        usage = response_usage(message) if message is not None else None
        if usage is None:
            self.usage.unreported_model_calls += 1
        else:
            totals = self.usage.token_usage.setdefault(profile, {})
            for key, value in usage.items():
                totals[key] = totals.get(key, 0) + value

        if usage is not None:
            self._event("model.usage", profile=profile, role=role, **usage)
            await self._record(run_id, self.session_id, profile, role, started_at, 'succeeded', usage)

    async def on_llm_error(self, error, *, run_id, **kwargs):
        if run_id in self._completed:
            return
        self._completed.add(run_id)
        self._model_finished(run_id, 'failed', type(error).__name__)
        response = kwargs.get('response')
        message = (response.generations[0][0].message if response and response.generations and response.generations[0] else None)
        usage = response_usage(message) if message is not None else None
        if run_id in self._models:
            profile, role, started_at = self._models.pop(run_id)
            if usage:
                totals = self.usage.token_usage.setdefault(profile, {})
                for key, value in usage.items():
                    totals[key] = totals.get(key, 0) + value
            else:
                self.usage.unreported_model_calls += 1

            if usage is not None:
                await self._record(run_id, self.session_id, profile, role, started_at, 'failed', usage)

    def _event(self, name, **values):
        from ai_assistant.observability import event
        event(name, input_id=self.input_id, session_id=self.session_id,
              execution_thread_id=self.execution_thread_id, **values)

    def _model_finished(self, run_id, status, category=None):
        started = self._model_times.pop(run_id, None)
        elapsed = monotonic() - started if started is not None else None
        self._event("model.end", run_id=run_id, parent_run_id=self._model_parents.get(run_id),
                    status=status, category=category, elapsed_seconds=elapsed)
        call = self._model_calls.pop(run_id, None)
        if call is not None:
            index, role, after_tool = call
            self._event('boundary.model_call', call_index=index, role=role,
                        after_tool=after_tool, status=status, category=category,
                        elapsed_seconds=elapsed)

    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self._parents[run_id] = parent_run_id

    async def on_tool_start(self, serialized, input_str, *, run_id=None, parent_run_id=None, **kwargs):
        # 避免将嵌套适配器调用重复计为 Agent 的工具调用。
        if run_id is not None and run_id in self._tool_seen:
            return
        from ai_assistant.runtime.execution import tool_scope
        scope = tool_scope.get()
        if scope is not None:
            if scope in self._tool_scopes:
                return
            self._tool_scopes.add(scope)
        parent = parent_run_id
        while parent is not None:
            if parent in self._tool_seen:
                self._parents[run_id] = parent_run_id
                return  # 适配器内部核对归属父工具，不重复消耗 Agent 工具额度。
            parent = self._parents.get(parent)
        if run_id is not None:
            self._tool_seen.add(run_id)
            self._parents[run_id] = parent_run_id
        self.usage.tool_calls += 1
        self._tool_started[run_id] = monotonic()
        self._tool_names[run_id] = (serialized or {}).get("name")
        self._event("tool.start", run_id=run_id, parent_run_id=parent_run_id, tool=(serialized or {}).get("name"))

    def _finish_tool(self, run_id, status="succeeded", category=None):
        started = self._tool_started.pop(run_id, None)
        name = self._tool_names.pop(run_id, None)
        if name:
            self._last_tool = name
        if started is not None:
            elapsed = monotonic() - started
            self.usage.tool_seconds += elapsed
            self._event("tool.end", run_id=run_id, parent_run_id=self._parents.get(run_id),
                        status=status, category=category, elapsed_seconds=elapsed)

    async def on_tool_error(self, error, **kwargs):
        self._finish_tool(kwargs.get('run_id'), 'failed', type(error).__name__)

    async def on_tool_end(self, output, **kwargs):
        # 提取 artifact 中的 Document 证据，并按 ID 去重。
        failed = isinstance(output, ToolMessage) and output.status == 'error'
        self._finish_tool(kwargs.get('run_id'), 'failed' if failed else 'succeeded',
                          'tool_result_error' if failed else None)
        if isinstance(output, ToolMessage) and isinstance(output.artifact, list):
            for doc in output.artifact:
                if isinstance(doc, Document) and doc.id not in self._seen:
                    self._seen.add(doc.id)
                    self.sources.append(doc)
