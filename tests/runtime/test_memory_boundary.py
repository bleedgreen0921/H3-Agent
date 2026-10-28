"""新版记忆边界：会话内连续，新会话隔离；无需长期记忆服务。"""
import asyncio
from contextlib import asynccontextmanager
import sys

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent
from ai_assistant.config import AgentSettings
from ai_assistant.integrations.policy import Caller
from ai_assistant.runtime import AssistantSession, assistant_session
from ai_assistant.tools.context import ToolContext
from ai_assistant.workspace.files import ReadOnlyWorkspace
from tests.integrations.test_mcp import config
from tests.runtime.test_context import managed
from tests.runtime.test_session import ScriptedModel


class RecordingModel(ScriptedModel):
    seen: list = []
    bindings: list = []

    def bind_tools(self, tools, **kwargs):
        self.bindings.append(([t.name for t in tools], kwargs))
        return self

    async def _agenerate(self, messages, **kwargs):
        self.seen.append(list(messages))
        return await super()._agenerate(messages, **kwargs)


def test_same_session_keeps_instruction_new_session_only_keeps_fixed_rules(tmp_path):
    async def run():
        model = RecordingModel(responses=[AIMessage('收到')])
        settings = AgentSettings()
        agent = create_assistant_agent(model, [], checkpointer=InMemorySaver(), settings=settings, persona='固定规则：直接回答。')
        s = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path)), settings)
        await s.run_turn('本会话以后用 Python 写题解，面试题用口语回答。背景代号：isolation-fixture-7')
        await s.run_turn('解释这段代码')
        assert any('isolation-fixture-7' in m.text for m in model.seen[-1])
        assert any('用 Python 写题解' in m.text for m in model.seen[-1])
        previous = s.session_id
        await s.new_conversation()
        await s.run_turn('这是全新的任务')
        assert s.session_id != previous
        assert not any('isolation-fixture-7' in m.text or '面试题用口语' in m.text for m in model.seen[-1])
        assert '固定规则：直接回答' in model.seen[-1][0].text
        assert len(model.seen) == 3
    asyncio.run(run())


def test_summary_remains_session_local_and_does_not_extract_facts(tmp_path):
    async def run():
        summary = RecordingModel(responses=[AIMessage('本会话约束：用 Python 题解，面试用口语。背景 isolation-summary-9。')])
        s = managed(tmp_path, [AIMessage('答' * 300)], summary=summary)
        await s.run_turn('本会话用 Python 题解，面试用口语。isolation-summary-9' + '请求' * 150)
        for _ in range(4):
            await s.run_turn('继续' * 150)
        assert summary.seen
        assert 'memories' not in summary.seen[-1][0].text
        assert '本会话' in summary.seen[-1][0].text
        state = await s.agent.aget_state(s._complete)
        assert 'pending_memories' not in state.values
        await s.new_conversation()
        turn = await s.run_turn('新任务')
        assert not any('isolation-summary-9' in m.text for m in turn.messages)
    asyncio.run(run())


def test_runtime_cannot_import_memory_or_require_terminal_protocol(tmp_path, monkeypatch, caplog):
    c = config(tmp_path)
    model = RecordingModel(responses=[AIMessage('普通回答')])
    @asynccontextmanager
    async def fake_model(*args, **kwargs):
        yield model
    monkeypatch.setattr('ai_assistant.runtime.model_session', fake_model)
    # 不能初始化存储或加载旧模型工具。
    for module in ('ai_assistant.storage.memory', 'ai_assistant.tools.memory', 'ai_assistant.tools.turn'):
        monkeypatch.setitem(sys.modules, module, None)
    async def run():
        async with assistant_session(c, caller=Caller('feishu_private')) as s:
            turn = await s.run_turn('我喜欢茶，帮我分析这个任务')
            assert turn.messages[-1].text == '普通回答'
            assert turn.usage.main_model_calls == 1
            assert not hasattr(s.context, 'memory')
            assert not any(t.name in {'complete_turn', 'remember_memory', 'search_memories'} for t in s.tools)
    asyncio.run(run())
    assert all(kwargs.get('tool_choice') != 'required' for _, kwargs in model.bindings)


def test_legacy_memory_configuration_is_rejected(tmp_path):
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        config(tmp_path, memory={'enabled': True})
