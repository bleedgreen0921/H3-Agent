"""统一注入可信当前时间；会话历史由原生 checkpoint 提供。"""
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage


class ContextPreparationMiddleware(AgentMiddleware):
    def prepare(self, request):
        context = request.runtime.context
        caller = getattr(context, 'caller', None)
        timezone = 'UTC' if caller and not caller.personal else getattr(context, 'timezone', 'UTC')
        original = request.system_message
        if original and original.additional_kwargs.get('assistant_time'):
            return request
        text = (original.text if original else '') + '\n可信当前时间：' + datetime.now(ZoneInfo(timezone)).isoformat(timespec='seconds') + '。日期转换工具只转换给定日期，不用于获取当前时间。'
        return request.override(system_message=SystemMessage(text, additional_kwargs={
            **(original.additional_kwargs if original else {}), 'assistant_time': True}))

    async def awrap_model_call(self, request, handler):
        return await handler(self.prepare(request))

    def wrap_model_call(self, request, handler):
        return handler(self.prepare(request))
