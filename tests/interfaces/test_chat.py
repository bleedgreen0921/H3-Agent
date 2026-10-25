"""Exercise real POSIX pipe input without model clients or input threads."""
import asyncio
from contextlib import asynccontextmanager
import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage
import pytest

from ai_assistant.interfaces import chat as module
from ai_assistant.runtime import AgentTurn


@pytest.mark.parametrize('data', [b'/help\n/new\nhello\n/exit\n', b'/help\n/new\nhello'])
def test_pipe_commands_eof_and_exit(monkeypatch, capsys, data):
    calls = []
    class Session:
        _pending = None
        session_id = 'test-session'
        model_name = 'fake'
        context = SimpleNamespace(workspace=SimpleNamespace(root='/workspace'))
        tools = []
        async def new_conversation(self):
            calls.append('new')
        async def run_turn(self, text, **kwargs):
            calls.append(text)
            return AgentTurn([AIMessage(content='answer')], [])
    @asynccontextmanager
    async def fake_session(*args, **kwargs):
        try:
            yield Session()
        finally:
            calls.append('closed')
    async def scenario():
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd) as reader:
            monkeypatch.setattr(module.sys, 'stdin', reader)
            os.write(write_fd, data)
            os.close(write_fd)
            return await module.chat(SimpleNamespace(sessions=True, context=True, knowledge=None))
    monkeypatch.setattr(module, 'assistant_session', fake_session)
    assert asyncio.run(scenario()) == 0
    assert calls == ['new', 'hello', 'closed']
    assert 'answer' in capsys.readouterr().out


def test_cli_safe_startup_error(monkeypatch, capsys):
    from ai_assistant.interfaces.cli import main
    from ai_assistant.config import AppConfig, ModelConfig
    config = AppConfig(default_model='fake', models={'fake': ModelConfig(provider='ollama', model='fake')})
    monkeypatch.setattr('ai_assistant.interfaces.cli.load_config', lambda _: config)
    async def fail(*args, **kwargs):
        raise ValueError('private diagnostic')
    monkeypatch.setattr(module, 'chat', fail)
    assert main(['chat']) == 2
    assert 'private diagnostic' not in capsys.readouterr().err


def test_sigint_waits_for_cleanup_then_continues(monkeypatch, capsys):
    import signal
    trace = []
    class Session:
        _pending = None
        session_id = 'test-session'
        model_name = 'fake'
        context = SimpleNamespace(workspace=SimpleNamespace(root='/workspace'))
        tools = []
        async def run_turn(self, text, **kwargs):
            if text == 'cancel':
                loop = asyncio.get_running_loop()
                loop.call_later(.01, os.kill, os.getpid(), signal.SIGINT)
                try:
                    await asyncio.sleep(10)
                finally:
                    # A second signal must not interrupt cleanup.
                    loop.call_later(.005, os.kill, os.getpid(), signal.SIGINT)
                    await asyncio.sleep(.02)
                    trace.append('cleaned')
            trace.append(text)
            return AgentTurn([AIMessage(content='answer')], [])
    @asynccontextmanager
    async def fake_session(*args, **kwargs):
        yield Session()
    monkeypatch.setattr(module, 'assistant_session', fake_session)
    async def scenario():
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd) as reader:
            monkeypatch.setattr(module.sys, 'stdin', reader)
            os.write(write_fd, b'cancel\nnext\n/exit\n')
            os.close(write_fd)
            return await module.chat(SimpleNamespace(sessions=True, context=True, knowledge=None))
    assert asyncio.run(scenario()) == 0
    assert trace == ['cleaned', 'next']
    assert '已取消' in capsys.readouterr().out


def test_idle_sigint_exit(monkeypatch):
    import signal
    @asynccontextmanager
    async def fake_session(*args, **kwargs):
        yield SimpleNamespace(_pending=None, session_id='test-session', model_name='fake', tools=[], context=SimpleNamespace(
            workspace=SimpleNamespace(root='/workspace')))
    monkeypatch.setattr(module, 'assistant_session', fake_session)
    async def scenario():
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd) as reader:
                monkeypatch.setattr(module.sys, 'stdin', reader)
                asyncio.get_running_loop().call_later(.01, os.kill, os.getpid(), signal.SIGINT)
                return await module.chat(SimpleNamespace(sessions=True, context=True, knowledge=None))
        finally:
            os.close(write_fd)
    assert asyncio.run(scenario()) == 130


def test_regular_file_eof(monkeypatch, tmp_path):
    @asynccontextmanager
    async def fake_session(*args, **kwargs):
        yield SimpleNamespace(_pending=None, session_id='test-session', model_name='fake', tools=[], context=SimpleNamespace(
            workspace=SimpleNamespace(root='/workspace')))
    monkeypatch.setattr(module, 'assistant_session', fake_session)
    source = tmp_path / 'input.txt'
    source.write_text('/help\n')
    async def scenario():
        with source.open() as reader:
            monkeypatch.setattr(module.sys, 'stdin', reader)
            return await module.chat(SimpleNamespace(sessions=True, context=True, knowledge=None))
    assert asyncio.run(scenario()) == 0
