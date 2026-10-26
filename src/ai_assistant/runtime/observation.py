"""Per-turn native callbacks; no persistent event bus or message copy."""
from time import monotonic
from dataclasses import dataclass, field

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

    def __init__(self, summary_limit, model_profile='', usage_store=None, session_id=None):
        self.store, self.session_id = usage_store, session_id
        self.summary_limit = summary_limit
        self._completed = set()
        self.model_profile = model_profile
        self.usage = TurnUsage()
        self.sources = []
        self._seen = set()
        self._models = {}
        self._tool_started = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        if run_id in self._models or run_id in self._completed:
            return
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
        self._models[run_id] = profile
        if self.store:
            await self.store.start(run_id, self.session_id, profile,
                'summary' if metadata.get('lc_source') == 'summarization' else 'main')

    async def on_llm_end(self, response, *, run_id, **kwargs):
        if run_id in self._completed:
            return
        self._completed.add(run_id)
        profile = self._models.pop(run_id, self.model_profile)
        message = response.generations[0][0].message if response.generations and response.generations[0] else None
        usage = response_usage(message) if message is not None else None
        if self.store:
            await self.store.finish(run_id, 'succeeded', usage)
        if usage is None:
            self.usage.unreported_model_calls += 1
        else:
            totals = self.usage.token_usage.setdefault(profile, {})
            for key, value in usage.items():
                totals[key] = totals.get(key, 0) + value

    async def on_llm_error(self, error, *, run_id, **kwargs):
        if run_id in self._completed:
            return
        self._completed.add(run_id)
        response = kwargs.get('response')
        message = (response.generations[0][0].message if response and response.generations and response.generations[0] else None)
        usage = response_usage(message) if message is not None else None
        if self.store:
            await self.store.finish(run_id, 'failed', usage)
        if run_id in self._models:
            profile = self._models.pop(run_id)
            if usage:
                totals = self.usage.token_usage.setdefault(profile, {})
                for key, value in usage.items():
                    totals[key] = totals.get(key, 0) + value
            else:
                self.usage.unreported_model_calls += 1

    async def on_tool_start(self, serialized, input_str, **kwargs):
        self.usage.tool_calls += 1
        self._tool_started[kwargs.get("run_id")] = monotonic()

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
