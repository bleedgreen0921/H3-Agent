"""显式真实 Agent 验收：只使用内存会话和本地无副作用工具。"""
import asyncio
import math

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents.completion import CompletionMiddleware
from ai_assistant.config import ConfigError
from ai_assistant.llm.factory import create_model
from ai_assistant.llm.requests import model_session


async def accept_agent(config, name=None, timeout=60, *, factory=create_model):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigError("Timeout must be a finite positive number.")
    selected = name or config.default_model
    if selected not in config.models:
        raise ConfigError('Unknown model profile.')
    calls = []

    @tool
    def local_add(a: int, b: int) -> int:
        """Add two integers locally without external side effects."""
        calls.append((a, b))
        return a + b

    prompts = [
        '本会话校验码是 7319。请记住它，并回答 2+3 等于多少。',
        '必须调用 local_add 工具计算 17+25，并回答结果。',
        '不要调用工具。请说出本会话校验码，以及上一轮工具计算的结果。',
    ]
    async with model_session(config, selected, None, factory=factory) as model:
        agent = create_agent(model, [local_add], checkpointer=InMemorySaver(),
                             middleware=[CompletionMiddleware()])
        results = []
        for number, prompt in enumerate(prompts, 1):
            async with asyncio.timeout(timeout):
                result = await agent.ainvoke({'messages': [('user', prompt)]},
                    {'configurable': {'thread_id': 'local-acceptance'}})
            text = result['messages'][-1].text
            valid = ('5' in text if number == 1 else
                     calls == [(17, 25)] and '42' in text if number == 2 else
                     calls == [(17, 25)] and '7319' in text and '42' in text)
            results.append({'round': number, 'passed': valid})
            if not valid:
                break
    return results
