"""Scripted native create_agent routing; no paid model calls."""
import pytest
pytest.importorskip("langchain_postgres")

import asyncio
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.knowledge.retrieval import create_knowledge_tool, knowledge_middleware


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class Evidence(BaseRetriever):
    failure: str = ''
    def _get_relevant_documents(self, query, *, run_manager):
        if self.failure == 'expected':
            raise KnowledgeError('safe detail')
        if self.failure:
            raise RuntimeError('unexpected bug')
        return [Document(id='evidence', page_content='证据', metadata={})]


def call(i):
    return AIMessage(content='', tool_calls=[dict(name='search_knowledge', args={'query': '内部资料'}, id=str(i))])


def test_agent_can_call_or_skip():
    async def scenario():
        for responses, count in (([AIMessage(content='普通问候')], 0), ([call(1), AIMessage(content='根据证据回答')], 1)):
            agent = create_agent(ScriptedModel(responses=responses), [create_knowledge_tool(Evidence())], middleware=knowledge_middleware())
            result = await agent.ainvoke({'messages': [('user', '测试')]})
            outputs = [m for m in result['messages'] if isinstance(m, ToolMessage)]
            assert len(outputs) == count
            if outputs:
                assert outputs[0].artifact[0].id == 'evidence'
    asyncio.run(scenario())


def test_native_limit_and_expected_error():
    async def scenario():
        agent = create_agent(ScriptedModel(responses=[call(1), call(2), AIMessage(content='done')]),
            [create_knowledge_tool(Evidence())], middleware=knowledge_middleware(run_limit=1))
        result = await agent.ainvoke({'messages': [('user', '测试')]})
        messages = [m for m in result['messages'] if isinstance(m, ToolMessage)]
        assert len(messages) == 2
        assert messages[0].artifact and not messages[1].artifact
        agent = create_agent(ScriptedModel(responses=[call(1), AIMessage(content='done')]),
            [create_knowledge_tool(Evidence(failure='expected'))], middleware=knowledge_middleware())
        result = await agent.ainvoke({'messages': [('user', '测试')]})
        message = next(m for m in result['messages'] if isinstance(m, ToolMessage))
        assert '知识检索暂不可用' in message.content and 'safe detail' not in message.content
        agent = create_agent(ScriptedModel(responses=[call(1)]),
            [create_knowledge_tool(Evidence(failure='unknown'))], middleware=knowledge_middleware())
        with pytest.raises(RuntimeError, match='unexpected bug'):
            await agent.ainvoke({'messages': [('user', '测试')]})
    asyncio.run(scenario())
