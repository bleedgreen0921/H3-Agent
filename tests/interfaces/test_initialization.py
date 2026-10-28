"""统一初始化无模型/云服务；部分成功可重跑。"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_assistant.config import ConfigError
from ai_assistant.interfaces.cli import main
from ai_assistant.interfaces.initialization import execute


@pytest.mark.parametrize('args', [
    ['config', 'check'], ['memory', 'list'], ['session', 'init'], ['mcp', 'init'],
    ['rag', 'init'], ['skill', 'inspect', 'id'], ['mcp', 'inspect', 'server'],
    ['mcp', 'check', '--network'],
])
def test_removed_commands_rejected(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_no_storage_is_explicit_error():
    with pytest.raises(ConfigError, match='没有可初始化'):
        asyncio.run(execute(SimpleNamespace(sessions=None, knowledge=None)))


def test_partial_initialization_and_idempotent_retry(monkeypatch, capsys):
    seen = []
    @asynccontextmanager
    async def sessions(*a, initialize=False):
        assert initialize
        seen.append('sessions')
        yield
    @asynccontextmanager
    async def operations(*a, initialize=False):
        assert initialize
        seen.append('operations')
        yield
    monkeypatch.setattr('ai_assistant.storage.sessions.session_storage', sessions)
    monkeypatch.setattr('ai_assistant.storage.integrations.operation_storage', operations)
    init = AsyncMock(side_effect=[RuntimeError('secret'), None])
    close = AsyncMock()
    monkeypatch.setattr('ai_assistant.storage.rag.RagStore', lambda c: SimpleNamespace(init=init, close=close))
    monkeypatch.setattr('ai_assistant.knowledge.sync.settings', lambda c: c.knowledge)
    c = SimpleNamespace(sessions=object(), knowledge=object())
    assert asyncio.run(execute(c)) == 2
    output = capsys.readouterr()
    assert '会话' in output.out and '知识库' in output.err and 'secret' not in output.err
    assert asyncio.run(execute(c)) == 0
    assert seen == ['sessions', 'operations', 'sessions', 'operations']
    assert close.await_count == 2
