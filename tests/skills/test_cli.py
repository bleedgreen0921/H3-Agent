import json
from pathlib import Path
import subprocess
import sys

from ai_assistant.config import load_config
from ai_assistant.interfaces.cli import main
from tests.skills.test_catalog import package


def configuration(tmp_path, *, skills=True):
    path = tmp_path / 'assistant.toml'
    path.write_text('default_model="test"\n[models.test]\nprovider="ollama"\nmodel="test"\n' + ('[skills]\ndependency_install=false\nroots=["extra"]\n' if skills else ''))
    return path


def test_cli_local_lifecycle_and_community_ownership(tmp_path, capsys):
    config = configuration(tmp_path)
    source = package(tmp_path / 'source')
    args = ['--config', str(config)]
    assert main(['skill', 'install', str(source), *args]) == 0
    installed = json.loads(capsys.readouterr().out)
    assert main(['skill', 'list', *args]) == 0
    listed = json.loads(capsys.readouterr().out)
    identity = listed['skills'][0]['id']
    assert main(['skill', 'inspect', identity, *args]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected['source']['digest'] == installed['digest']
    assert main(['skill', 'check', identity, *args]) == 0
    assert json.loads(capsys.readouterr().out)['check']['format'] == 'valid'
    assert main(['skill', 'deps', 'install', identity, '--manager', 'pip', '--requirement', 'pypdf', *args]) == 2
    capsys.readouterr()
    assert main(['skill', 'remove', identity, *args]) == 0
    capsys.readouterr()
    package(tmp_path / 'extra/example')
    assert main(['skill', 'remove', 'example', *args]) == 2
    assert 'another installer' in capsys.readouterr().err
    assert (tmp_path / 'extra/example/SKILL.md').is_file()


def test_config_does_not_import_integration(tmp_path):
    config = configuration(tmp_path)
    code = '''
import sys
from ai_assistant.config import load_config
config = load_config(sys.argv[1])
assert config.skills.dependency_install is False
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
