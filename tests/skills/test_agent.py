import asyncio

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_assistant.agents import create_assistant_agent
from ai_assistant.config import AgentSettings
from ai_assistant.runtime import AssistantSession
from ai_assistant.skills.catalog import SkillCatalog
from ai_assistant.tools.context import ToolContext
from ai_assistant.tools.skills import create_skill_tools
from ai_assistant.workspace.files import ReadOnlyWorkspace
from tests.runtime.test_session import ScriptedModel

def call(name, ident, **args):
    return AIMessage(content="", tool_calls=[dict(name=name, args=args, id=ident)])
from tests.skills.test_catalog import package


def session(tmp_path, responses):
    catalog = SkillCatalog([tmp_path / 'skills'])
    model = ScriptedModel(responses=responses)
    settings = AgentSettings()
    tools = create_skill_tools(False)
    agent = create_assistant_agent(model, tools, checkpointer=InMemorySaver(), settings=settings, skills_enabled=True)
    context = ToolContext(ReadOnlyWorkspace(tmp_path), skills=catalog)
    return AssistantSession(agent, context, settings), model


def test_load_read_unload_and_explicit_new(tmp_path):
    package(tmp_path / 'skills/example', body='MAGIC_SKILL_RULE')
    async def run():
        s, model = session(tmp_path, [AIMessage(content='ordinary'), call('load_skill', 'load', skill_id='example'),
            call('read_skill_resource', 'read', skill_id='example', resource='examples/test.txt'),
            AIMessage(content='done'), call('unload_skill', 'unload', skill_id='example'), AIMessage(content='unloaded'),
            AIMessage(content='explicit'), AIMessage(content='new')])
        try:
            assert not (await s.run_turn('hello')).skill_ids
            turn = await s.run_turn('use example')
            assert len(turn.skill_ids) == 1
            assert not (await s.run_turn('stop')).skill_ids
            assert (await s.run_turn('explicit', skill_ids=['example'])).skill_ids
            await s.new_conversation()
            assert not (await s.run_turn('new')).skill_ids
        finally:
            s.context.skills.close()
    asyncio.run(run())


def test_parallel_loads_and_failed_explicit_selection(tmp_path):
    import pytest
    package(tmp_path / 'skills/one', name='one')
    package(tmp_path / 'skills/two', name='two')
    parallel = AIMessage(content='', tool_calls=[dict(name='load_skill', args={'skill_id': name}, id=name) for name in ('one', 'two')])
    async def run():
        s, model = session(tmp_path, [AIMessage(content='baseline'), AIMessage(content='FAIL'),
                                    AIMessage(content='recovered'), parallel, AIMessage(content='done')])
        try:
            await s.run_turn('baseline')
            with pytest.raises(RuntimeError):
                await s.run_turn('fail', skill_ids=['one'])
            assert not (await s.run_turn('recover')).skill_ids
            assert len((await s.run_turn('load both')).skill_ids) == 2
        finally:
            s.context.skills.close()
    asyncio.run(run())


def test_request_body_protection_and_invalidation(tmp_path):
    import pytest
    from pydantic import PrivateAttr
    from ai_assistant.config import ContextConfig
    from ai_assistant.agents.context import ContextBudgetError
    class RecordingModel(ScriptedModel):
        _seen: list = PrivateAttr(default_factory=list)
        async def _agenerate(self, messages, **kwargs):
            self._seen.append(messages)
            return await super()._agenerate(messages, **kwargs)
    package(tmp_path / 'skills/example', body='MAGIC_BODY' * 1000)
    async def run():
        catalog = SkillCatalog([tmp_path / 'skills'])
        model = RecordingModel(responses=[AIMessage(content='ok')])
        settings = AgentSettings()
        context = ToolContext(ReadOnlyWorkspace(tmp_path), skills=catalog)
        agent = create_assistant_agent(model, create_skill_tools(False), checkpointer=InMemorySaver(),
            settings=settings, skills_enabled=True, max_input_tokens=5000)
        s = AssistantSession(agent, context, settings)
        try:
            await s.run_turn('normal')
            assert 'MAGIC_BODY' not in '\n'.join(m.text for m in model._seen[-1])
            with pytest.raises(ContextBudgetError):
                await s.run_turn('explicit', skill_ids=['example'])
        finally:
            catalog.close()
        catalog = SkillCatalog([tmp_path / 'skills'])
        context = ToolContext(ReadOnlyWorkspace(tmp_path), skills=catalog)
        model = RecordingModel(responses=[call('load_skill', 'load', skill_id='example'), AIMessage(content='ok'), AIMessage(content='reopened')])
        agent = create_assistant_agent(model, create_skill_tools(False), checkpointer=InMemorySaver(), settings=settings, skills_enabled=True)
        s = AssistantSession(agent, context, settings)
        try:
            await s.run_turn('load')
            assert 'MAGIC_BODY' not in '\n'.join(m.text for m in model._seen[0])
            assert sum('MAGIC_BODY' in m.text for m in model._seen[1]) == 1
            state = await agent.aget_state(s._complete)
            assert not any('MAGIC_BODY' in m.text for m in state.values['messages'])
            # A new catalog revision invalidates the old activation on reopened state.
            package(tmp_path / 'skills/example', body='NEW_BODY')
            fresh = SkillCatalog([tmp_path / 'skills'])
            try:
                s.context = ToolContext(ReadOnlyWorkspace(tmp_path), skills=fresh)
                await s.run_turn('reopen')
                assert not any('MAGIC_BODY' in m.text or 'NEW_BODY' in m.text for m in model._seen[-1])
                assert 'expired' in model._seen[-1][0].text
            finally:
                fresh.close()
        finally:
            catalog.close()
    asyncio.run(run())


def test_active_body_survives_native_summary(tmp_path):
    from pydantic import PrivateAttr
    from ai_assistant.config import ContextConfig
    from ai_assistant.agents.context import is_summary
    class RecordingModel(ScriptedModel):
        _seen: list = PrivateAttr(default_factory=list)
        async def _agenerate(self, messages, **kwargs):
            self._seen.append(messages)
            return await super()._agenerate(messages, **kwargs)
    package(tmp_path / 'skills/example', body='SUMMARY_PROTECTED_RULE' * 20)
    async def run():
        catalog = SkillCatalog([tmp_path / 'skills'])
        settings = AgentSettings()
        model = RecordingModel(responses=[AIMessage(content='答' * 600)])
        summary = ScriptedModel(responses=[AIMessage(content='摘要：用户仍在处理任务。')])
        agent = create_assistant_agent(model, create_skill_tools(False), checkpointer=InMemorySaver(), settings=settings,
            skills_enabled=True, context_settings=ContextConfig(), summary_model=summary, max_input_tokens=5000,
            summary_max_input_tokens=10000, summary_profile='summary')
        s = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path), skills=catalog), settings)
        try:
            turns = [await s.run_turn('问' * 400, skill_ids=['example'] if i == 0 else None) for i in range(6)]
            assert any(t.usage.summary_model_calls for t in turns)
            assert all('SUMMARY_PROTECTED_RULE' in messages[0].text for messages in model._seen)
            assert any(is_summary(m) for m in turns[-1].messages)
            assert not any('SUMMARY_PROTECTED_RULE' in m.text for m in turns[-1].messages)
        finally:
            catalog.close()
    asyncio.run(run())


def test_large_directory_uses_complete_search(tmp_path):
    from pydantic import PrivateAttr
    class RecordingModel(ScriptedModel):
        _seen: list = PrivateAttr(default_factory=list)
        async def _agenerate(self, messages, **kwargs):
            self._seen.append(messages)
            return await super()._agenerate(messages, **kwargs)
    for i in range(10):
        root = package(tmp_path / f'skills/example-{i}', name=f'example-{i}')
        text = (root / 'SKILL.md').read_text().replace('description: Test skill', 'description: ' + 'description ' * 50)
        (root / 'SKILL.md').write_text(text)
    async def run():
        catalog = SkillCatalog([tmp_path / 'skills'])
        settings = AgentSettings()
        model = RecordingModel(responses=[AIMessage(content='ok')])
        agent = create_assistant_agent(model, create_skill_tools(False), checkpointer=InMemorySaver(), settings=settings, skills_enabled=True)
        s = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path), skills=catalog), settings)
        try:
            await s.run_turn('hello')
            assert 'Use search_skills with pagination' in model._seen[0][0].text
            assert catalog.search()['total'] == 10
            assert 'example-0' not in model._seen[0][0].text
        finally:
            catalog.close()
    asyncio.run(run())
