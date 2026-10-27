"""Memory provenance, structured extraction and failure behavior without accounts."""
from datetime import datetime, timedelta, timezone
import json

from langchain_core.messages import HumanMessage, ToolMessage
import pytest

from ai_assistant.agents.context import ContextBudgetError
from ai_assistant.storage.memory import MemoryRepository, valid_candidate


def sample():
    now = datetime.now(timezone.utc)
    policy = {'version': 1, 'paused': False, 'watermark': now - timedelta(seconds=1)}
    message = HumanMessage('请记住，我喜欢简洁中文。', id='user-1', additional_kwargs={
        'user_source': '请记住，我喜欢简洁中文。', 'user_timestamp': now.isoformat()})
    candidate = dict(category='preference', topic='回答风格', values=['简洁中文'], cardinality='single',
                     quote='我喜欢简洁中文。', source_id='user-1')
    return policy, message, candidate


def test_source_identity_quote_credentials_and_watermark():
    policy, message, candidate = sample()
    assert valid_candidate(candidate, [message], policy)
    assert valid_candidate(candidate, [message], policy, explicit=True)
    assert valid_candidate(candidate, [ToolMessage('我喜欢简洁中文。', id='user-1', tool_call_id='x')], policy) is None
    assert valid_candidate({**candidate, 'source_id': 'guessed'}, [message], policy) is None
    # 规范内容可与原话不同；语义判断由主模型负责，来源仍严格匹配。
    assert valid_candidate({**candidate, 'values': ['简洁中文']}, [message], policy)
    assert valid_candidate({**candidate, 'topic': 'api_key'}, [message], policy) is None
    assert valid_candidate(candidate, [message], {**policy, 'paused': True}) is None
    assert valid_candidate(candidate, [message], {**policy, 'watermark': datetime.now(timezone.utc)}) is None
    assert valid_candidate(candidate, [message], {**policy, 'paused': True}, explicit=True)
    assert valid_candidate(candidate, [message], {**policy, 'watermark': datetime.now(timezone.utc)}, explicit=True) is None


def test_valid_summary_survives_invalid_candidates_and_no_repair_call():
    policy, message, candidate = sample()
    repo = object.__new__(MemoryRepository)
    repo.policy = policy
    summary, candidates = repo.parse_summary(json.dumps({'summary': '目标和约束', 'memories': [candidate, {}]}), [message])
    assert summary == '目标和约束' and len(candidates) == 1
    assert repo.parse_summary('{"summary":"有效摘要","memories":"invalid"}', [message]) == ('有效摘要', [])
    with pytest.raises(ContextBudgetError):
        repo.parse_summary('not json', [message])


def test_natural_statement_latest_authorization_and_forbidden_storage():
    policy, old, candidate = sample()
    natural = HumanMessage('以后默认简洁中文回复。', id='natural', additional_kwargs={
        'user_source': '以后默认简洁中文回复。', 'user_timestamp': datetime.now(timezone.utc).isoformat()})
    raw = {**candidate, 'values': ['简洁中文'], 'quote': natural.content, 'source_id': natural.id}
    assert valid_candidate(raw, [natural], policy, explicit=True)
    assert valid_candidate(candidate, [old, natural], policy, explicit=True) is None
    forbidden = natural.model_copy(update={'additional_kwargs': {
        **natural.additional_kwargs, 'user_source': '不要记住：以后默认简洁中文回复。'}})
    assert valid_candidate(raw, [forbidden], policy, explicit=True) is None


def test_timezones_and_invalid_revision_requests():
    from pydantic import ValidationError
    from ai_assistant.storage.memory_types import MemoryChange
    _, _, raw = sample()
    with pytest.raises(ValidationError):
        MemoryChange(**raw, valid_from='2026-10-26T10:00:00')
    with pytest.raises(ValidationError):
        MemoryChange(**raw, action='replace')
    with pytest.raises(ValidationError):
        MemoryChange(**raw, valid_from='2026-10-27T10:00:00+08:00', valid_until='2026-10-26T10:00:00+08:00')


def test_one_off_script_is_not_saved_even_in_session_scope():
    policy, _, candidate = sample()
    source = HumanMessage('今天用 Python 写个临时脚本，不用执行。', id='once', additional_kwargs={
        'user_source': '今天用 Python 写个临时脚本，不用执行。',
        'user_timestamp': datetime.now(timezone.utc).isoformat()})
    assert valid_candidate({**candidate, 'quote': source.content, 'values': ['Python'],
                            'source_id': source.id, 'scope': 'session'}, [source], policy, explicit=True) is None


def test_pronoun_context_is_evidence_not_old_authorization():
    policy, old, candidate = sample()
    source = HumanMessage('以后都这样。', id='new', additional_kwargs={
        'user_source': '以后都这样。', 'user_timestamp': datetime.now(timezone.utc).isoformat()})
    raw = {**candidate, 'source_id': source.id, 'quote': source.content,
           'context_evidence': [{'source_id': old.id, 'quote': candidate['quote']}]}
    validated = valid_candidate(raw, [old, source], policy, explicit=True)
    assert validated['source_id'] == 'new'
    assert validated['verified_context'][0]['source_id'] == old.id
    assert valid_candidate(raw, [source], policy, explicit=True) is None


def test_unknown_commit_tool_returns_receipt_and_suppresses_cache():
    import asyncio
    from types import SimpleNamespace
    from ai_assistant.tools.memory import remember_memory
    from ai_assistant.storage.memory_types import MemoryChange
    class BrokenMemory:
        reference = 'fixture-cached-private-value'
        async def remember(self, *args):
            raise ConnectionError('private connection detail')
    async def run():
        _, source, raw = sample()
        memory = BrokenMemory()
        runtime = SimpleNamespace(context=SimpleNamespace(memory=memory),
                                  state={'messages': [source]}, tool_call_id='unknown')
        command = await remember_memory.coroutine(MemoryChange(**raw), runtime)
        receipt = next(iter(command.update['memory_results'].values()))
        assert receipt['status'] == 'unknown'
        assert 'fixture-cached-private-value' not in memory.reference
        assert 'private connection detail' not in str(command)
    asyncio.run(run())


def test_add_cannot_bypass_revision_guard_with_target_id():
    from pydantic import ValidationError
    from ai_assistant.storage.memory_types import MemoryChange
    _, _, raw = sample()
    with pytest.raises(ValidationError):
        MemoryChange(**raw, action='add', memory_id='existing')


def test_final_answer_cannot_claim_success_after_unknown_memory_write():
    import asyncio
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage
    from ai_assistant.agents.memory import MemoryMiddleware
    async def run():
        update = await MemoryMiddleware().aafter_agent({
            'messages': [AIMessage('已记住你的偏好。', id='final')],
            'memory_results': {'write': {'status': 'unknown', 'reason': 'commit_not_confirmed'}}}, SimpleNamespace())
        assert '已记住' not in update['messages'][0].text
        assert '状态未知' in update['messages'][0].text
    asyncio.run(run())


def test_memory_preview_hides_unconfirmed_claims_across_chunks():
    from ai_assistant.agents.memory import MemoryPreview
    text = '其他业务完成。<memory_result>已保存草稿。</memory_result>后续说明。'
    for chunk_size in range(1, 20):
        preview = MemoryPreview()
        output = ''.join(preview.feed(text[i:i + chunk_size]) for i in range(0, len(text), chunk_size)) + preview.finish()
        assert output == '其他业务完成。后续说明。'


def test_mixed_answer_preserves_other_work_and_corrects_deleted_memory():
    import asyncio
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage
    from ai_assistant.agents.memory import MemoryMiddleware
    async def run():
        update = await MemoryMiddleware().aafter_agent({
            'messages': [AIMessage('其他业务结果：文件已写入。<memory_result>可能索引延迟</memory_result>')],
            'memory_reads': {'read': {'status': 'deleted', 'summary': '该记忆已删除'}}}, SimpleNamespace())
        text = update['messages'][0].text
        assert '文件已写入' in text and '索引延迟' not in text and '已删除' in text
    asyncio.run(run())


def test_current_source_is_runtime_bound_and_absent_from_model_schema():
    import asyncio
    from types import SimpleNamespace
    from ai_assistant.tools.memory import remember_memory
    from ai_assistant.storage.memory_types import MemoryRequest, MemoryChange
    assert 'source_id' not in MemoryRequest.model_fields
    async def run():
        _, old, raw = sample()
        current = HumanMessage('以后简洁中文回复。', id='runtime-current', additional_kwargs={
            'user_source': '以后简洁中文回复。', 'user_timestamp': datetime.now(timezone.utc).isoformat()})
        captured = []
        class Memory:
            async def remember(self, change, messages):
                captured.append(change)
                return json.dumps({'status': 'saved', 'reason': 'committed', 'summary': '持久化完成'})
        runtime = SimpleNamespace(context=SimpleNamespace(memory=Memory()), state={'messages': [old, current]},
                                  tool_call_id='bound')
        # 即使旧调用对象带错 ID，写入口仍仅绑定当前可信消息；公开 schema 不让模型提供 ID。
        request = MemoryChange(**{**raw, 'quote': current.content})
        await remember_memory.coroutine(request, runtime)
        assert captured[0]['source_id'] == current.id
    asyncio.run(run())


def test_read_outage_reaches_final_answer_and_failed_interface_event(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from ai_assistant.agents import create_assistant_agent
    from ai_assistant.config import AgentSettings
    from ai_assistant.runtime import AssistantSession
    from ai_assistant.tools.context import ToolContext
    from ai_assistant.tools.memory import search_memories
    from ai_assistant.workspace.files import ReadOnlyWorkspace
    from tests.runtime.test_session import ScriptedModel
    class UnavailableMemory:
        reference = ''
        async def recall(self, text):
            self.query = text
        async def get_policy(self):
            raise ConnectionError('fixture private database detail')
        async def search(self, *args, **kwargs):
            raise ConnectionError('fixture private database detail')
    async def run():
        model = ScriptedModel(responses=[
            AIMessage(content='', tool_calls=[{'name': 'search_memories', 'args': {}, 'id': 'read'}]),
            AIMessage(content='<memory_result>没有找到，可能是索引延迟。</memory_result>')])
        settings = AgentSettings()
        agent = create_assistant_agent(model, [search_memories], checkpointer=InMemorySaver(), settings=settings)
        session = AssistantSession(agent, ToolContext(ReadOnlyWorkspace(tmp_path), memory=UnavailableMemory()), settings)
        events = []
        turn = await session.run_turn('查找我的记忆', on_output=events.append)
        assert events[-1]['kind'] == 'failed'
        assert '暂不可用' in turn.messages[-1].text
        assert '索引延迟' not in turn.messages[-1].text
        assert 'fixture private database detail' not in turn.messages[-1].text
    asyncio.run(run())
