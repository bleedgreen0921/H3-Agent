"""Offline native graph integration, including failed-branch isolation."""
import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent
from ai_assistant.config import AgentSettings
from ai_assistant.runtime import AssistantSession, final_text
from ai_assistant.tools.context import ToolContext
from ai_assistant.workspace.files import ReadOnlyWorkspace


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, **kwargs):
        if self.responses[self.i].content == 'FAIL':
            self.i += 1
            raise RuntimeError('private diagnostics')
        result = await super()._agenerate(messages, **kwargs)
        for generation in result.generations:
            generation.message = generation.message.model_copy(update={'id': str(uuid4())})
        return result


def session(tmp_path, responses, tools=(), settings=None):
    settings = settings or AgentSettings()
    agent = create_assistant_agent(ScriptedModel(responses=responses), list(tools),
        checkpointer=InMemorySaver(), settings=settings)
    return AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path)), settings)


def call(name, ident):
    return AIMessage(content='', tool_calls=[dict(name=name, args={}, id=ident)])


def test_context_and_new(tmp_path):
    async def scenario():
        s = session(tmp_path, [AIMessage(content='one'), AIMessage(content='two'), AIMessage(content='three')])
        assert len((await s.run_turn('first')).messages) == 2
        assert len((await s.run_turn('second')).messages) == 4
        await s.new_conversation()
        assert len((await s.run_turn('third')).messages) == 2
    asyncio.run(scenario())


def test_reopened_session_does_not_inject_workspace_task(tmp_path):
    async def scenario():
        s = session(tmp_path, [AIMessage(content='直接回答')])
        s._reopened = True
        result = await s.run_turn('介绍一下你自己')
        assert [m.content for m in result.messages if m.type == 'human'] == ['介绍一下你自己']
    asyncio.run(scenario())


def test_failed_branch_not_replayed(tmp_path):
    async def scenario():
        calls = []
        @tool
        async def effect():
            """Record a side effect."""
            calls.append(1)
            return 'observed'
        for history in (False, True):
            responses = ([AIMessage(content='baseline')] if history else []) + [
                call('effect', 'effect-id'), AIMessage(content='FAIL'), AIMessage(content='recovered')]
            s = session(tmp_path, responses, [effect])
            if history:
                await s.run_turn('good')
            with pytest.raises(RuntimeError, match='private diagnostics'):
                await s.run_turn('failed request')
            result = await s.run_turn('fresh request')
            assert final_text(result.messages) == 'recovered'
            assert all(m.content != 'failed request' for m in result.messages)
            assert not any(isinstance(m, ToolMessage) for m in result.messages)
            assert any(m.content == 'baseline' for m in result.messages) == history
        assert len(calls) == 2
    asyncio.run(scenario())


def test_tool_chain_events_and_per_turn_limit(tmp_path):
    async def scenario():
        @tool
        async def step():
            """Return an observation."""
            return 'observed'
        s = session(tmp_path, [call('step', 'a'), call('step', 'b'), AIMessage(content='done')], [step])
        events = []
        result = await s.run_turn('go', on_event=events.append)
        assert len([m for m in result.messages if isinstance(m, ToolMessage)]) == 2
        assert len([e for e in events if e['event'] == 'on_tool_end']) == 2
        from ai_assistant.storage.sessions import SessionError
        s = session(tmp_path, [call('step', 'a'), AIMessage(content='done')], [step], AgentSettings(model_call_limit=1))
        with pytest.raises(SessionError, match='本轮未记录远端写入'):
            await s.run_turn('go')
        assert final_text((await s.run_turn('new')).messages) == 'done'
    asyncio.run(scenario())


def test_timeout_waits_for_cleanup(tmp_path):
    async def scenario():
        cleaned = []
        @tool
        async def slow():
            """Wait until cancelled."""
            try:
                await asyncio.sleep(10)
            finally:
                await asyncio.sleep(.01)
                cleaned.append(True)
        s = session(tmp_path, [call('slow', 'a')], [slow], AgentSettings(turn_timeout_seconds=.05))
        with pytest.raises(TimeoutError):
            await s.run_turn('wait')
        assert cleaned == [True]
    asyncio.run(scenario())


def test_global_tool_batch_limit(tmp_path):
    async def scenario():
        from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
        calls = []
        @tool
        async def step():
            """Record invocation."""
            calls.append(1)
            return 'ok'
        batch = AIMessage(content='', tool_calls=[dict(name='step', args={}, id=str(i)) for i in range(3)])
        s = session(tmp_path, [batch], [step], AgentSettings(tool_call_limit=2))
        with pytest.raises(ToolCallLimitExceededError):
            await s.run_turn('go')
        assert not calls
    asyncio.run(scenario())


def test_artifacts_survive_tool_execution(tmp_path):
    from langchain_core.documents import Document
    async def scenario():
        @tool(response_format='content_and_artifact')
        async def evidence():
            """Return source artifacts."""
            return 'evidence', [Document(id='a', page_content='private'), Document(id='a', page_content='private')]
        s = session(tmp_path, [call('evidence', 'a'), AIMessage(content='done'), AIMessage(content='followup')], [evidence])
        assert [d.id for d in (await s.run_turn('go')).sources] == ['a']
        assert (await s.run_turn('next')).sources == []
    asyncio.run(scenario())


def test_settings_validation_and_disabled_tools():
    from pydantic import ValidationError
    from ai_assistant.tools.builtin import create_builtin_tools
    assert [t.name for t in create_builtin_tools()] == ['list_directory', 'read_text', 'search_text']
    for key in ('model_call_limit', 'tool_call_limit', 'knowledge_call_limit', 'turn_timeout_seconds'):
        with pytest.raises(ValidationError):
            AgentSettings(**{key: 0})
    for value in (float('nan'), float('inf')):
        with pytest.raises(ValidationError):
            AgentSettings(turn_timeout_seconds=value)


def test_assistant_session_resource_ownership(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    import ai_assistant.runtime as runtime
    from ai_assistant.config import AppConfig, ModelConfig, WorkspaceConfig
    closed = []
    @asynccontextmanager
    async def fake_model(*args, **kwargs):
        try:
            yield ScriptedModel(responses=[AIMessage(content='ok')])
        finally:
            closed.append(True)
    monkeypatch.setattr(runtime, 'model_session', fake_model)
    config = AppConfig(default_model='fake', models={'fake': ModelConfig(provider='ollama', model='fake')},
                       workspace=WorkspaceConfig(root=str(tmp_path)))
    async def scenario():
        async with runtime.assistant_session(config) as s:
            assert final_text((await s.run_turn('hello')).messages) == 'ok'
            saver = s.agent.checkpointer
            assert saver.storage
        assert not saver.storage
    asyncio.run(scenario())
    assert closed == [True]


def test_context_exit_waits_for_active_tool_before_closing_resources(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from ai_assistant.config import AppConfig, ModelConfig, WorkspaceConfig
    from ai_assistant.runtime import assistant_session
    trace = []
    entered = asyncio.Event()
    @tool
    async def slow():
        """Wait and clean up when the session closes."""
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(.01)
            trace.append('tool cleaned')
        return 'done'
    @asynccontextmanager
    async def fake(*args, **kwargs):
        try:
            yield ScriptedModel(responses=[call('slow', 'slow-id')])
        finally:
            trace.append('client closed')
    monkeypatch.setattr('ai_assistant.runtime.model_session', fake)
    monkeypatch.setattr('ai_assistant.runtime.create_builtin_tools', lambda **kwargs: [slow])
    c = AppConfig(default_model='fake', models={'fake': ModelConfig(provider='ollama', model='fake')},
                  workspace=WorkspaceConfig(root=str(tmp_path)))
    async def run():
        async with assistant_session(c) as s:
            task = asyncio.create_task(s.run_turn('wait'))
            await asyncio.wait_for(entered.wait(), 2)
        assert task.cancelled()
        with pytest.raises(RuntimeError, match='closed'):
            await s.run_turn('after close')
    asyncio.run(run())
    assert trace == ['tool cleaned', 'client closed']


def test_expanded_model_and_tool_budgets_are_enforced(tmp_path):
    async def run():
        from ai_assistant.storage.sessions import SessionError
        from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
        observed = []
        @tool
        async def step():
            """Record a fixture tool call."""
            observed.append(1)
            return 'ok'
        settings = AgentSettings()
        assert (settings.model_call_limit, settings.tool_call_limit, settings.summary_call_limit) == (20, 40, 10)
        responses = [call('step', str(i)) for i in range(20)] + [AIMessage(content='must not reach')]
        s = session(tmp_path, responses, [step], settings)
        with pytest.raises(SessionError):
            await s.run_turn('reach model budget')
        assert len(observed) == 20
        observed.clear()
        batch = AIMessage(content='', tool_calls=[dict(name='step', args={}, id=str(i)) for i in range(40)])
        s = session(tmp_path, [batch, AIMessage(content='done')], [step], settings)
        turn = await s.run_turn('forty tools')
        assert len(observed) == 40 and turn.usage.main_model_calls == 2 and turn.usage.tool_calls == 40
        observed.clear()
        too_many = batch.model_copy(update={'tool_calls': [dict(name='step', args={}, id=str(i)) for i in range(41)]})
        s = session(tmp_path, [too_many], [step], settings)
        with pytest.raises(ToolCallLimitExceededError):
            await s.run_turn('tool budget')
        assert not observed
    asyncio.run(run())
