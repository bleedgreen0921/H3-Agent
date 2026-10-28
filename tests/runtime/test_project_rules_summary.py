"""根规则与滚动整理回归；不访问外部模型。"""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ai_assistant.agents import load_project_rules
from ai_assistant.agents.context import is_summary
from ai_assistant.config import ContextConfig
from ai_assistant.tools.context import ToolContext
from ai_assistant.workspace.files import ReadOnlyWorkspace, WorkspaceError
from tests.runtime.test_context import managed


def test_rules_boundaries(tmp_path):
    context = ToolContext(ReadOnlyWorkspace(tmp_path))
    assert load_project_rules(context) == ''
    path = tmp_path / 'AGENTS.md'
    path.write_text('规则', encoding='utf-8')
    assert load_project_rules(context) == '规则'
    path.write_bytes(b'\xff')
    with pytest.raises(WorkspaceError):
        load_project_rules(context)
    path.write_text('x' * 16385)
    with pytest.raises(WorkspaceError):
        load_project_rules(context)
    path.unlink()
    path.symlink_to(tmp_path / 'missing')
    with pytest.raises(WorkspaceError):
        load_project_rules(context)
    path.unlink()
    path.write_text('规则')
    with pytest.raises(WorkspaceError):
        load_project_rules(ToolContext(ReadOnlyWorkspace(tmp_path, exclude_patterns=['AGENTS.md'])))
    from types import SimpleNamespace
    assert load_project_rules(ToolContext(ReadOnlyWorkspace(tmp_path), caller=SimpleNamespace(channel='feishu_group'))) == ''


def test_periodic_twenty_turns(tmp_path):
    async def run():
        session = managed(tmp_path, [AIMessage(content='回答')], budget=100000)
        for number in range(1, 21):
            result = await session.run_turn(f'请求{number}')
            state = (await session.agent.aget_state(session._complete)).values
            assert state['completed_turns'] == number
            assert result.usage.summary_model_calls == (1 if number in (10, 20) else 0)
            raw = [m.text for m in state['messages'] if isinstance(m, HumanMessage) and not is_summary(m)]
            start = 18 if number == 20 else 8 if number >= 10 else 1
            assert raw == [f'请求{i}' for i in range(start, number + 1)]
        await session.new_conversation()
        await session.run_turn('新会话')
        state = (await session.agent.aget_state(session._complete)).values
        assert state['completed_turns'] == 1
        assert not any(is_summary(m) for m in state['messages'])
    asyncio.run(run())


def test_config_summary_limits():
    assert ContextConfig().summary_tokens == 10000
    assert ContextConfig(summary_every_turns=0).summary_every_turns == 0
    for value in (-1, True, 1.5, '10'):
        with pytest.raises(ValueError):
            ContextConfig(summary_every_turns=value)


def test_periodic_failure_keeps_history_and_retries_next_turn(tmp_path):
    from tests.runtime.test_session import ScriptedModel

    async def run():
        summary = ScriptedModel(responses=[AIMessage(content='FAIL'), AIMessage(content='简短摘要')])
        session = managed(tmp_path, [AIMessage(content='业务回执')], summary=summary, budget=100000)
        for i in range(10):
            result = await session.run_turn(str(i))
        state = (await session.agent.aget_state(session._complete)).values
        assert result.messages[-1].text == '业务回执'
        assert len(state['messages']) == 20
        assert state.get('summarized_turns', 0) == 0
        result = await session.run_turn('10')
        state = (await session.agent.aget_state(session._complete)).values
        assert state['summarized_turns'] == 11
        assert result.usage.summary_model_calls == 1
    asyncio.run(run())


def test_rules_refresh_and_never_enter_history(tmp_path):
    from tests.runtime.test_session import ScriptedModel
    from ai_assistant.agents import create_assistant_agent
    from ai_assistant.config import AgentSettings
    from ai_assistant.runtime import AssistantSession
    from langgraph.checkpoint.memory import InMemorySaver

    seen = []

    class RecordingModel(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            seen.append(messages[0].text)
            return await super()._agenerate(messages, **kwargs)

    async def run():
        settings = AgentSettings()
        agent = create_assistant_agent(RecordingModel(responses=[AIMessage(content='答')]), [],
                                       settings=settings, checkpointer=InMemorySaver())
        session = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path)), settings)
        path = tmp_path / 'AGENTS.md'
        path.write_text('项目规则第一版')
        await session.run_turn('问')
        path.write_text('项目规则第二版')
        await session.run_turn('再问')
        assert '项目规则第一版' in seen[0]
        assert '项目规则第二版' in seen[1]
        state = (await agent.aget_state(session._complete)).values
        assert all('项目规则' not in m.text for m in state['messages'])
    asyncio.run(run())


def test_rules_case_conflict_on_case_sensitive_filesystem(tmp_path):
    (tmp_path / 'AGENTS.md').write_text('第一份')
    (tmp_path / 'agents.md').write_text('第二份')
    if len(list(tmp_path.iterdir())) < 2:
        pytest.skip('文件系统不区分大小写，无法创建两个变体')
    with pytest.raises(WorkspaceError, match='多个'):
        load_project_rules(ToolContext(ReadOnlyWorkspace(tmp_path)))


def test_summary_target_and_session_counter_restore(tmp_path):
    from tests.runtime.test_session import ScriptedModel
    from ai_assistant.runtime import AssistantSession
    seen = []

    class RecordingSummary(ScriptedModel):
        async def _agenerate(self, messages, **kwargs):
            seen.append(messages)
            return await super()._agenerate(messages, **kwargs)

    async def run():
        summary = RecordingSummary(responses=[AIMessage(content='短摘要')])
        session = managed(tmp_path, [AIMessage(content='答')], summary=summary, budget=100000)
        for i in range(9):
            await session.run_turn(str(i))
        restored = AssistantSession(session.agent, session.context, session.settings)
        restored._thread, restored._complete = session._thread, session._complete
        await restored.run_turn('第十轮')
        state = (await restored.agent.aget_state(restored._complete)).values
        assert state['completed_turns'] == state['summarized_turns'] == 10
        assert len(seen) == 1
        assert '10000 token' in seen[0][0].text
        assert any('短摘要' in m.text for m in state['messages'] if is_summary(m))
    asyncio.run(run())


def test_summary_topic_replaces_initial_preview(tmp_path):
    from tests.runtime.test_session import ScriptedModel

    async def run():
        summary = ScriptedModel(responses=[AIMessage(content='会话主题：飞书指令改造\n保留重要约束。')])
        session = managed(tmp_path, [AIMessage(content='答复')], summary=summary, budget=100000)
        for i in range(10):
            await session.run_turn(f'请求{i}')
        state = (await session.agent.aget_state(session._complete)).values
        assert state['conversation_title'] == '飞书指令改造'
    asyncio.run(run())
