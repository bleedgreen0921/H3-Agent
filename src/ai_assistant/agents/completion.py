"""所有原生 Agent 共用响应检查，不解释业务正文。"""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from ai_assistant.llm.completion import IncompleteResponseError, validate_completion


class CompletionMiddleware(AgentMiddleware):
    def __init__(self, phase='execution'):
        self.phase = phase

    def validate(self, result, model):
        for message in result.result:
            if isinstance(message, AIMessage):
                try:
                    validate_completion(message, responses=getattr(model, 'use_responses_api', False) is True)
                except IncompleteResponseError as error:
                    error.phase = self.phase
                    raise
                if not message.tool_calls and not message.text.strip():
                    error = IncompleteResponseError('模型响应为空，本轮未完成。')
                    error.phase = self.phase
                    raise error
        return result

    async def awrap_model_call(self, request, handler):
        return self.validate(await handler(request), request.model)

    def wrap_model_call(self, request, handler):
        return self.validate(handler(request), request.model)
