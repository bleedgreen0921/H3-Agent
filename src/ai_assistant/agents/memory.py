"""Reference-only personal memory injection; identity comes from trusted runtime."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from ai_assistant.tools.mcp import OfficeState


class MemoryState(OfficeState):
    pending_memories: list[dict]


class MemoryMiddleware(AgentMiddleware):
    def prepare(self, request):
        memory = getattr(request.runtime.context, 'memory', None)
        if memory is None:
            return request
        original = request.system_message.text if request.system_message else ''
        source = next((m for m in reversed(request.messages) if 'user_source' in m.additional_kwargs), None)
        identity = f'\n当前用户消息 ID：{source.id}' if source else ''
        return request.override(system_message=SystemMessage(original + '\n' + memory.reference + identity))

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))
