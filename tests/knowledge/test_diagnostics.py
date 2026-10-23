"""Private, bounded diagnostics and explicit CLI viewing."""
import pytest
pytest.importorskip('langchain_postgres')
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch
from ai_assistant.knowledge.diagnostics import write_diagnostic, read_diagnostic, MAX_BYTES, attempt_path
from ai_assistant.config import AppConfig, ModelConfig
from ai_assistant.knowledge.config import KnowledgeConfig
from ai_assistant.knowledge.cli import execute


def context(tmp_path):
    config = SimpleNamespace(knowledge_base_id='../kb/中文', diagnostics_dir=tmp_path/'diagnostics')
    job = dict(id=str(uuid4()), build_id=str(uuid4()), attempts=1)
    return config, job


def test_permissions_budget_and_local_read(tmp_path):
    config, job = context(tmp_path)
    path = Path(write_diagnostic(config, job, 'parser_failed', '解析失败', '机密错误正文' * 20000))
    assert path.stat().st_size <= MAX_BYTES
    assert path.stat().st_mode & 0o777 == 0o600
    for directory in (config.diagnostics_dir, path.parent.parent, path.parent):
        assert directory.stat().st_mode & 0o777 == 0o700
    status = dict(id=job['id'], latest_build_id=job['build_id'], diagnostic_path=str(path))
    detail = read_diagnostic(config, status)
    assert len(detail) <= 16000 and '显示已截断' in detail
    assert job['id'] in detail and 'parser_failed' in detail
    assert '截断' in path.read_text()
    other = SimpleNamespace(knowledge_base_id='other', diagnostics_dir=config.diagnostics_dir)
    assert '受控目录' in read_diagnostic(other, status)
    path.unlink()
    assert '不存在' in read_diagnostic(config, status)


def test_path_validation_symlink_and_regular_file(tmp_path):
    config, job = context(tmp_path)
    status = dict(id=job['id'], latest_build_id=job['build_id'], diagnostic_path='/etc/passwd')
    assert '受控目录' in read_diagnostic(config, status)
    path = Path(write_diagnostic(config, job, 'parser_failed', '解析失败', 'details'))
    status['diagnostic_path'] = str(path)
    path.unlink()
    target = tmp_path/'outside'
    target.write_text('must not read')
    path.symlink_to(target)
    assert '符号链接' in read_diagnostic(config, status)
    path.unlink()
    path.mkdir()
    assert '普通文件' in read_diagnostic(config, status)
    path.rmdir()
    parent = path.parent
    parent.rmdir()
    parent.symlink_to(tmp_path, target_is_directory=True)
    assert '符号链接' in read_diagnostic(config, status)
    with pytest.raises(OSError):
        write_diagnostic(config, job, 'parser_failed', '解析失败', 'details')


def test_status_cli_only_displays_diagnostics_when_requested(tmp_path, capsys):
    c = KnowledgeConfig(database_url='postgresql+asyncpg://unused:unused@127.0.0.1/unused',
                        diagnostics_dir=tmp_path/'private')
    job = dict(id=str(uuid4()), build_id=str(uuid4()))
    path = write_diagnostic(c, job, 'parser_failed', '解析失败', 'private traceback')
    result = dict(id=job['id'], latest_build_id=job['build_id'], diagnostic_path=path,
                  error='解析失败', error_code='parser_failed', document_id='document')
    config = AppConfig(default_model='test', models={'test': ModelConfig(provider='ollama', model='test')}, knowledge=c)
    class Store:
        def __init__(self, settings):
            assert settings.diagnostics_dir == c.diagnostics_dir
        async def status(self, job_id): return result
        async def close(self): pass
    with patch('ai_assistant.storage.rag.RagStore', Store):
        for enabled in (False, True):
            assert asyncio.run(execute(config, SimpleNamespace(action='status', job_id=job['id'], diagnostics=enabled))) == 0
            captured = capsys.readouterr()
            assert json.loads(captured.out) == result
            assert ('private traceback' in captured.err) == enabled
            assert 'private traceback' not in captured.out


def test_relative_diagnostic_directory_resolves_against_toml(tmp_path, monkeypatch, capsys):
    config = AppConfig(default_model='test', models={'test': ModelConfig(provider='ollama', model='test')},
                       knowledge=KnowledgeConfig(database_url='postgresql+asyncpg://unused:unused@127.0.0.1/unused',
                                                 diagnostics_dir=Path('private')))
    config._config_path = tmp_path/'config'/'assistant.toml'
    unrelated = tmp_path/'cwd'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    class Store:
        def __init__(self, settings):
            assert settings.diagnostics_dir == tmp_path/'config'/'private'
        async def status(self, job_id): return {'id': job_id}
        async def close(self): pass
    with patch('ai_assistant.storage.rag.RagStore', Store):
        assert asyncio.run(execute(config, SimpleNamespace(action='status', job_id='job', diagnostics=False))) == 0
    assert not (tmp_path/'config'/'private').exists()  # querying does not create diagnostic files
