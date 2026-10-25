"""Native context behavior with scripted models; no credentials or remote calls."""
import asyncio
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent
from ai_assistant.agents.context import (
    ContextBudgetError, InputBudgetMiddleware, SummaryCallLimitError,
    count_tokens, fit_messages, is_summary, validate_tool_pairs,
)
from ai_assistant.config import AgentSettings, ContextConfig
from ai_assistant.runtime import AssistantSession
from ai_assistant.runtime.observation import TurnObserver
from ai_assistant.tools.context import ToolContext
from ai_assistant.workspace.files import ReadOnlyWorkspace
from tests.runtime.test_session import ScriptedModel, call


def managed(tmp_path, responses, *, tools=(), summary=None, settings=None, budget=1600):
    settings = settings or AgentSettings()
    model = ScriptedModel(responses=responses)
    summary = summary or ScriptedModel(responses=[AIMessage(content='约束：保留用户目标；已观察到工具结果。')])
    agent = create_assistant_agent(model, list(tools), checkpointer=InMemorySaver(), settings=settings,
        context_settings=ContextConfig(), summary_model=summary, max_input_tokens=budget,
        summary_max_input_tokens=10000, summary_profile='summary')
    return AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path)), settings, model_name='main')


def test_native_summary_changes_state_and_counts_independently(tmp_path):
    async def run():
        s = managed(tmp_path, [AIMessage(content='答' * 300)], settings=AgentSettings(model_call_limit=1))
        turns = [await s.run_turn('问' * 300) for _ in range(4)]
        assert any(t.usage.summary_model_calls for t in turns)
        last = turns[-1]
        assert last.usage.main_model_calls == 1
        assert last.usage.summary_model_calls <= 1
        assert any(is_summary(m) for m in last.messages)
        assert len(last.messages) < 8
        # Summaries are persisted natively, and the original turn still exists in history.
        state = await s.agent.aget_state(s._complete)
        assert any(is_summary(m) for m in state.values['messages'])
        history = [old async for old in s.agent.aget_state_history({'configurable': {'thread_id': s._thread}})]
        assert any(not any(is_summary(m) for m in old.values.get('messages', [])) for old in history)
    asyncio.run(run())


def test_sources_survive_mid_turn_summary(tmp_path):
    async def run():
        @tool(response_format='content_and_artifact')
        async def evidence():
            """Return retrieved evidence."""
            return '证' * 650, [Document(id='actual', page_content='source')]
        s = managed(tmp_path, [AIMessage(content='答' * 250), AIMessage(content='答' * 250),
                               call('evidence', 'a'), AIMessage(content='done')], tools=[evidence])
        await s.run_turn('问' * 250)
        before = await s.run_turn('问' * 250)
        turn = await s.run_turn('retrieve')
        assert [d.id for d in turn.sources] == ['actual']
        assert turn.usage.summary_model_calls > 0
        assert len(turn.messages) <= len(before.messages) + 1
        assert turn.usage.tool_calls == 1
    asyncio.run(run())


def test_empty_summary_and_unknown_error_propagate(tmp_path):
    async def run():
        for output, error in [('', ContextBudgetError), ('FAIL', RuntimeError)]:
            summary = ScriptedModel(responses=[AIMessage(content=output)])
            s = managed(tmp_path, [AIMessage(content='答' * 300)], summary=summary)
            await s.run_turn('问' * 300)
            await s.run_turn('问' * 300)
            with pytest.raises(error):
                await s.run_turn('问' * 300)
            # One attempt, no framework summary retry.
            assert summary.i == (1 if output == 'FAIL' else 0)
    asyncio.run(run())


def test_summary_limit_callback_aborts_before_extra_model_call():
    async def run():
        observer = TurnObserver(1)
        model = ScriptedModel(responses=[AIMessage(content='ok')])
        config = {'callbacks': [observer], 'metadata': {'lc_source': 'summarization'}}
        await model.ainvoke('first', config=config)
        with pytest.raises(SummaryCallLimitError):
            await model.ainvoke('second', config=config)
        assert observer.usage.summary_model_calls == 1
    asyncio.run(run())


def test_trimming_preserves_summary_latest_input_and_parallel_pairs():
    summary = HumanMessage('用户约束', additional_kwargs={'lc_source': 'summarization'})
    messages = [summary, HumanMessage('旧' * 1000), AIMessage(content='旧' * 1000),
                HumanMessage('当前请求'), AIMessage(content='', tool_calls=[
                    dict(name='a', args={}, id='a'), dict(name='b', args={}, id='b')]),
                ToolMessage(content='a result', tool_call_id='a'), ToolMessage(content='b result', tool_call_id='b')]
    result = fit_messages(messages, 500)
    assert summary in result and any(m.content == '当前请求' for m in result)
    assert not any(m.content == '旧' * 1000 for m in result)
    validate_tool_pairs(result)
    with pytest.raises(ContextBudgetError):
        fit_messages([HumanMessage('大' * 2000)], 500)
    with pytest.raises(ContextBudgetError):
        validate_tool_pairs([messages[-1]])


def test_context_editing_is_request_only():
    from langchain.agents.middleware.types import ModelRequest
    messages = [HumanMessage('question')]
    for i in range(4):
        messages += [call('step', str(i)), ToolMessage(content='结果' * 100,
            artifact={'private': 'artifact'}, tool_call_id=str(i), name='step')]
    original = messages[2]
    middleware = ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=1, keep=3)], token_counter=count_tokens)
    request = ModelRequest(model=ScriptedModel(responses=[AIMessage(content='ok')]), messages=messages)
    seen = []
    middleware.wrap_model_call(request, lambda r: seen.extend(r.messages))
    assert seen[2].content == '[cleared]' and seen[2].artifact is None
    assert original.content != '[cleared]' and original.artifact == {'private': 'artifact'}


def test_request_budget_includes_system_and_tool_schemas():
    from langchain.agents.middleware.types import ModelRequest
    @tool
    def verbose_tool():
        """A tool with a description that must be included in the input budget."""
        return ''
    request = ModelRequest(model=ScriptedModel(responses=[AIMessage(content='ok')]),
        messages=[HumanMessage('tiny')], system_message=SystemMessage('系统' * 1000), tools=[verbose_tool])
    with pytest.raises(ContextBudgetError):
        InputBudgetMiddleware(100).prepare(request)


def test_large_task_schema_fits_followup_without_resetting_history():
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages.utils import count_tokens_approximately
    @tool
    def update_task(task_id: str, completed: bool):
        """Update an existing task."""
        return ''
    # 模拟官方 task.patch 的大 schema 描述；不调用真实账号。
    update_task = update_task.model_copy(update={'description': 'task field description ' * 500})
    messages = [HumanMessage('近期任务'), call('list_tasks', 'list'),
                ToolMessage(content='task_id=known-id; status=todo', tool_call_id='list'),
                AIMessage(content='有一条待办任务'), HumanMessage('把它设为已完成')]
    request = ModelRequest(model=ScriptedModel(responses=[AIMessage(content='ok')]),
        messages=messages, tools=[update_task])
    fitted = InputBudgetMiddleware(5000).prepare(request)
    assert fitted.messages == messages
    # schema 仍计入预算，过小的预算必须在模型调用前拒绝。
    assert count_tokens_approximately([], tools=[update_task]) > 1000
    with pytest.raises(ContextBudgetError):
        InputBudgetMiddleware(1000).prepare(request)


def test_selected_summary_profile_clients_usage_and_retry_parameters(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from ai_assistant.config import AppConfig, ModelConfig, WorkspaceConfig
    from ai_assistant.runtime import assistant_session
    opened, closed = [], []
    usage = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}
    @asynccontextmanager
    async def fake(config, selected, *args, **kwargs):
        opened.append((selected, config.models[selected].parameters))
        text = '答' * 300 if selected == 'main' else '保留用户约束和目标'
        try:
            yield ScriptedModel(responses=[AIMessage(content=text, usage_metadata=usage)])
        finally:
            closed.append(selected)
    monkeypatch.setattr('ai_assistant.runtime.model_session', fake)
    monkeypatch.setattr('ai_assistant.runtime.create_builtin_tools', lambda **kwargs: [])
    c = AppConfig(default_model='main', context=ContextConfig(summary_model='summary'),
        workspace=WorkspaceConfig(root=str(tmp_path)), models={
            'main': ModelConfig(provider='ollama', model='fake', max_input_tokens=1600, parameters={'max_retries': 2}),
            'summary': ModelConfig(provider='ollama', model='fake', max_input_tokens=10000,
                                   parameters={'max_retries': 1, 'num_predict': 1000})})
    async def run():
        async with assistant_session(c) as s:
            turns = [await s.run_turn('问' * 300) for _ in range(4)]
            summarized = next(t for t in turns if t.usage.summary_model_calls)
            assert summarized.usage.token_usage['summary'] == usage
            assert summarized.usage.token_usage['main'] == usage
            assert summarized.usage.unreported_model_calls == 0
    asyncio.run(run())
    assert opened == [('main', {'max_retries': 2}), ('summary', {'max_retries': 1, 'num_predict': 1000})]
    assert closed == ['summary', 'main']
