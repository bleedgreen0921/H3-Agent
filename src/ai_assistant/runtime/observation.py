"""Per-turn native callbacks; no persistent event bus or message copy."""
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


class TurnObserver(AsyncCallbackHandler):
    raise_error = True
    run_inline = True

    def __init__(self, summary_limit, model_profile=''):
        self.summary_limit = summary_limit
        self.model_profile = model_profile
        self.usage = TurnUsage()
        self.sources = []
        self._seen = set()
        self._models = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        metadata = metadata or {}
        if metadata.get('lc_source') == 'summarization':
            if self.usage.summary_model_calls >= self.summary_limit:
                raise SummaryCallLimitError('Summary call limit reached.')
            self.usage.summary_model_calls += 1
        else:
            self.usage.main_model_calls += 1
        self._models[run_id] = metadata.get('assistant_model_profile', self.model_profile)

    async def on_llm_end(self, response, *, run_id, **kwargs):
        profile = self._models.pop(run_id, self.model_profile)
        message = response.generations[0][0].message if response.generations and response.generations[0] else None
        usage = response_usage(message) if message is not None else None
        if usage is None:
            self.usage.unreported_model_calls += 1
        else:
            totals = self.usage.token_usage.setdefault(profile, {})
            for key, value in usage.items():
                totals[key] = totals.get(key, 0) + value

    async def on_llm_error(self, error, *, run_id, **kwargs):
        if run_id in self._models:
            self._models.pop(run_id)
            self.usage.unreported_model_calls += 1

    async def on_tool_start(self, serialized, input_str, **kwargs):
        self.usage.tool_calls += 1

    async def on_tool_end(self, output, **kwargs):
        if isinstance(output, ToolMessage) and isinstance(output.artifact, list):
            for doc in output.artifact:
                if isinstance(doc, Document) and doc.id not in self._seen:
                    self._seen.add(doc.id)
                    self.sources.append(doc)
