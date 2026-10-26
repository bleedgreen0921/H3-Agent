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
    candidate = dict(category='preference', topic='回答风格', content='我喜欢简洁中文',
                     quote='我喜欢简洁中文。', source_id='user-1')
    return policy, message, candidate


def test_source_identity_quote_credentials_and_watermark():
    policy, message, candidate = sample()
    assert valid_candidate(candidate, [message], policy)
    assert valid_candidate(candidate, [message], policy, explicit=True)
    assert valid_candidate(candidate, [ToolMessage('我喜欢简洁中文。', id='user-1', tool_call_id='x')], policy) is None
    assert valid_candidate({**candidate, 'source_id': 'guessed'}, [message], policy) is None
    assert valid_candidate({**candidate, 'content': '模型猜测'}, [message], policy) is None
    assert valid_candidate({**candidate, 'topic': 'api_key'}, [message], policy) is None
    assert valid_candidate(candidate, [message], {**policy, 'paused': True}) is None
    assert valid_candidate(candidate, [message], {**policy, 'watermark': datetime.now(timezone.utc)}) is None
    # 主人再次明确保存可以越过自动提取水位。
    assert valid_candidate(candidate, [message], {**policy, 'paused': True}, explicit=True)


def test_valid_summary_survives_invalid_candidates_and_no_repair_call():
    policy, message, candidate = sample()
    repo = object.__new__(MemoryRepository)
    repo.policy = policy
    summary, candidates = repo.parse_summary(json.dumps({'summary': '目标和约束', 'memories': [candidate, {}]}), [message])
    assert summary == '目标和约束' and len(candidates) == 1
    assert repo.parse_summary('{"summary":"有效摘要","memories":"invalid"}', [message]) == ('有效摘要', [])
    with pytest.raises(ContextBudgetError):
        repo.parse_summary('not json', [message])
