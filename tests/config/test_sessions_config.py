"""Persistence configuration stays pure and provider independent."""
import asyncio
from contextlib import asynccontextmanager
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage
from ai_assistant.config import AppConfig, ConfigError, ContextConfig, ModelConfig, SessionsConfig
from ai_assistant.runtime import validate_context_config
from ai_assistant.interfaces.cli import main


def test_budgets_are_explicit_positive_integers():
    for value in (0, -1, True, 1.5, '100'):
        with pytest.raises(ValidationError):
            ModelConfig(provider='ollama', model='fake', max_input_tokens=value)
    c = AppConfig(default_model='fake', models={'fake': ModelConfig(provider='ollama', model='fake')},
                  context=ContextConfig())
    with pytest.raises(ConfigError, match='max_input_tokens'):
        validate_context_config(c, 'fake')
    with pytest.raises(ValidationError):
        c.model_copy().model_validate({**c.model_dump(), 'context': {'summary_model': 'missing'}})


def test_database_url_is_not_in_safe_errors():
    with pytest.raises(ValidationError) as exc:
        SessionsConfig(database_url='postgresql+asyncpg://secret:password@host/db')
    assert 'password' not in str(exc.value)


def test_old_config_cli_check_works_and_interactive_chat_is_removed(tmp_path, capsys):
    path = tmp_path / 'assistant.toml'
    path.write_text('default_model="fake"\n[models.fake]\nprovider="ollama"\nmodel="fake"\n')
    assert main(['mcp', 'list', '--config', str(path)]) == 0
    with pytest.raises(SystemExit) as error:
        main(['chat', '--config', str(path)])
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err


def test_session_cli_dispatch_does_not_construct_models(monkeypatch):
    c = AppConfig(default_model='fake', models={'fake': ModelConfig(provider='ollama', model='fake')},
                  sessions=SessionsConfig(database_url='postgresql://localhost/test'))
    monkeypatch.setattr('ai_assistant.interfaces.cli.load_config', lambda _: c)
    monkeypatch.setattr('ai_assistant.llm.factory.create_model', lambda *args: pytest.fail('No model construction'))
    seen = []
    async def execute(config, args):
        seen.append(args.action)
        return 0
    monkeypatch.setattr('ai_assistant.interfaces.sessions.execute', execute)
    assert main(['session', 'list']) == 0
    assert main(['session', 'delete', '00000000-0000-0000-0000-000000000000']) == 0
    assert seen == ['list', 'delete']


def test_config_import_does_not_load_optional_database_or_gpu_modules():
    code = '''import sys
from ai_assistant.config import AppConfig
from ai_assistant.runtime import assistant_session
assert not any(x in sys.modules for x in ['psycopg', 'sqlalchemy', 'torch', 'ai_assistant.knowledge.retrieval'])
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_missing_session_extra_has_safe_error(monkeypatch):
    from ai_assistant.storage.sessions import session_storage
    monkeypatch.setitem(sys.modules, 'langgraph.checkpoint.postgres.aio', None)
    async def run():
        with pytest.raises(ConfigError, match='sessions extra'):
            async with session_storage(SessionsConfig(database_url='postgresql://secret:password@localhost/test')):
                pytest.fail('Must not connect')
    asyncio.run(run())
