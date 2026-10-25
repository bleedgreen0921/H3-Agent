import json
import subprocess
import sys

from ai_assistant.interfaces.cli import main
from tests.skills.test_catalog import package


def configuration(tmp_path, *, skills=True):
    path = tmp_path / 'assistant.toml'
    path.write_text('default_model="test"\n[models.test]\nprovider="ollama"\nmodel="test"\n' + ('[skills]\nroots=["extra"]\n' if skills else ''))
    return path


def test_cli_local_inspection_and_removed_commands(tmp_path, capsys):
    import pytest
    config = configuration(tmp_path)
    source = package(tmp_path / 'extra/example')
    args = ['--config', str(config)]
    assert main(['skill', 'list', *args]) == 0
    listed = json.loads(capsys.readouterr().out)
    identity = next(s['id'] for s in listed['skills'] if s['name'] == 'example')
    assert main(['skill', 'inspect', identity, *args]) == 0
    assert 'source' not in json.loads(capsys.readouterr().out)
    assert main(['skill', 'check', identity, *args]) == 0
    assert json.loads(capsys.readouterr().out)['check']['format'] == 'valid'
    for command in ['install', 'discover', 'update', 'remove', 'deps']:
        with pytest.raises(SystemExit):
            main(['skill', command, *args])
    assert source.is_dir()


def test_config_does_not_import_integration(tmp_path):
    config = configuration(tmp_path)
    code = '''
import sys
from ai_assistant.config import load_config
config = load_config(sys.argv[1])
assert config.skills.roots == ["extra"]
assert not any(n.startswith('ai_assistant.skills') for n in sys.modules)
assert 'yaml' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(config)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_disabled_skills_cli_does_not_scan(tmp_path, capsys):
    config = configuration(tmp_path, skills=False)
    package(tmp_path / '.agents/skills/example')
    assert main(['skill', 'list', '--config', str(config)]) == 2
    assert 'Configure [skills]' in capsys.readouterr().err
