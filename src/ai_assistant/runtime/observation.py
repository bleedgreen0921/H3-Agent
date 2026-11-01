"""Per-turn native callbacks; no persistent event bus or message copy."""
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
    raise_error = True
    run_inline = True

    def __init__(self, summary_limit, model_profile='', usage_store=None, session_id=None, *, input_id=None):
        self.store, self.session_id = usage_store, session_id
        self.input_id = input_id
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

    async def _record(self, *args):
        if self.store is None:
            return
        try:
            await self.store.record(*args)
        except Exception as error:
            self.store = None
            logging.getLogger(__name__).warning('Usage unavailable: input_id=%s session_id=%s phase=%s category=%s',
                self.input_id, self.session_id, 'record', type(error).__name__)

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        if run_id in self._model_started or run_id in self._completed:
            return
        self._model_started.add(run_id)
        metadata = metadata or {}
        if metadata.get('lc_source') == 'summarization':
            if self.usage.summary_model_calls >= self.summary_limit:
                raise SummaryCallLimitError('Summary call limit reached.')
            self.usage.summary_model_calls += 1
        else:
            self.usage.main_model_calls += 1
        from ai_assistant.agents.context import count_tokens
        self.usage.max_input_tokens = max(self.usage.max_input_tokens, *(count_tokens(m) for m in messages))
        profile = metadata.get('assistant_model_profile', self.model_profile)
        self._models[run_id] = (profile,
            'summary' if metadata.get('lc_source') == 'summarization' else 'main',
            datetime.now(timezone.utc))

    async def on_llm_end(self, response, *, run_id, **kwargs):
        if run_id in self._completed:
            return
        self._completed.add(run_id)
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
            await self._record(run_id, self.session_id, profile, role, started_at, 'succeeded', usage)

    async def on_llm_error(self, error, *, run_id, **kwargs):
        if run_id in self._completed:
            return
        self._completed.add(run_id)
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

    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self._parents[run_id] = parent_run_id

    async def on_tool_start(self, serialized, input_str, *, run_id=None, parent_run_id=None, **kwargs):
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

    def _finish_tool(self, run_id):
        started = self._tool_started.pop(run_id, None)
        if started is not None:
            self.usage.tool_seconds += monotonic() - started

    async def on_tool_error(self, error, **kwargs):
        self._finish_tool(kwargs.get('run_id'))

    async def on_tool_end(self, output, **kwargs):
        self._finish_tool(kwargs.get('run_id'))
        if isinstance(output, ToolMessage) and isinstance(output.artifact, list):
            for doc in output.artifact:
                if isinstance(doc, Document) and doc.id not in self._seen:
                    self._seen.add(doc.id)
                    self.sources.append(doc)
