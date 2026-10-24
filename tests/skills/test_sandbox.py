"""Actual Bubblewrap skill mounts and environment protection, offline."""
import asyncio
from pathlib import Path
import shutil
import subprocess
import sys
import shlex

import pytest

from ai_assistant.skills.catalog import SkillCatalog
from ai_assistant.tools.terminal import TerminalService, TerminalError
from ai_assistant.workspace.files import ReadOnlyWorkspace
from tests.skills.test_catalog import package


def test_readonly_skill_execution_and_maintenance_environment(tmp_path):
    if not shutil.which('bwrap'):
        pytest.skip('Bubblewrap unavailable')
    project = tmp_path / 'project'
    project.mkdir()
    env = tmp_path / 'venv'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(env)], check=True, capture_output=True)
    source = package(tmp_path / 'skills/example')
    (source / 'script.py').write_text('from pathlib import Path\nprint(Path(__file__).with_name("examples").joinpath("test.txt").read_text())\n')
    catalog = SkillCatalog([source.parent])
    skill = catalog.resolve('example')
    service = TerminalService(ReadOnlyWorkspace(project), venv=env, profile='maintenance', timeout=10)
    service.skills_mode = True
    service.npm_project = project
    async def run():
        try:
            try:
                result = await service.run_command('python ' + shlex.quote(skill.sandbox_path + '/script.py'), skills=[skill])
            except TerminalError as exc:
                if 'Bubblewrap' in str(exc) or '沙箱无法运行' in str(exc):
                    pytest.skip(str(exc))
                raise
            assert result['exit_code'] == 0, result
            assert 'Example' in result['output']
            assert (await service.run_command('echo overwrite > ' + shlex.quote(skill.sandbox_path + '/SKILL.md'), skills=[skill]))['exit_code'] != 0
            assert (await service.run_command('echo overwrite > ' + shlex.quote(str(env / 'probe')), skills=[skill]))['exit_code'] != 0
            assert not (env / 'probe').exists()
            assert (await service.run_command('echo output > result.txt', skills=[skill]))['exit_code'] == 0
            assert (project / 'result.txt').read_text().strip() == 'output'
            unavailable = await service.run_command('test -f ' + shlex.quote(skill.sandbox_path + '/SKILL.md'))
            assert unavailable['exit_code'] != 0
        finally:
            await service.aclose()
            catalog.close()
    asyncio.run(run())


def test_node_commonjs_esm_and_registered_internal_links(tmp_path):
    if not shutil.which('bwrap') or not shutil.which('node'):
        pytest.skip('Bubblewrap/Node unavailable')
    project = tmp_path / 'project'
    project.mkdir()
    tree = project / 'node_modules'
    (tree / 'fixture-package').mkdir(parents=True)
    (tree / 'fixture-package/package.json').write_text('{"name":"fixture-package","main":"index.js"}')
    (tree / 'fixture-package/index.js').write_text('module.exports = 42;')
    (tree / '.bin').mkdir()
    (tree / '.bin/fixture').symlink_to('../fixture-package/index.js')
    (project / 'package.json').write_text('{"name":"test"}')
    (project / 'package-lock.json').write_text('{}')
    source = package(tmp_path / 'skills/example')
    (source / 'test.cjs').write_text('console.log(require("fixture-package"));')
    (source / 'test.mjs').write_text('import value from "fixture-package"; console.log(value);')
    env = tmp_path / 'venv'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(env)], check=True, capture_output=True)
    catalog = SkillCatalog([source.parent])
    skill = catalog.resolve('example')
    service = TerminalService(ReadOnlyWorkspace(project), venv=env, profile='maintenance', timeout=10)
    service.skills_mode = True
    service.npm_project = project
    async def run():
        try:
            for name in ('test.cjs', 'test.mjs'):
                result = await service.run_command('node ' + shlex.quote(skill.sandbox_path + '/' + name), skills=[skill])
                assert result['exit_code'] == 0, result
                assert '42' in result['output']
            for target in (tree / 'fixture-package/index.js', project / 'package.json', project / 'package-lock.json'):
                result = await service.run_command('echo overwrite > ' + shlex.quote(str(target)), skills=[skill])
                assert result['exit_code'] != 0
        finally:
            await service.aclose()
            catalog.close()
    asyncio.run(run())
