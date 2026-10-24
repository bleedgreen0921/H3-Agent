"""Native offline resolvers in disposable environments; never mutate this test runner."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import zipfile

import pytest

from ai_assistant.config import SkillsConfig
from ai_assistant.skills.catalog import SkillError
from ai_assistant.skills.dependencies import DependencyService, Transaction, inventory
from ai_assistant.skills.manager import file_lock


def wheel(directory, version='1.0', extra=None, name='assistant_skills_fixture'):
    path = directory / f'{name}-{version}-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name + '.py', 'VALUE = 42\n')
        prefix = f'{name}-{version}.dist-info/'
        z.writestr(prefix + 'METADATA', f'Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: {version}\n')
        z.writestr(prefix + 'WHEEL', 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        z.writestr(prefix + 'RECORD', '')
        if extra:
            for name, content in extra.items():
                z.writestr(name, content)
    return path


def test_real_pip_offline_add_only_and_rollback(tmp_path):
    env = tmp_path / 'venv'
    subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True, capture_output=True)
    target = env / 'lib/python3.12/site-packages'
    # Test harness only: expose framework imports without installing or copying
    # the real environment, and keep all publication inside the disposable venv.
    (target / 'test-framework.pth').write_text(sysconfig.get_path('purelib') + '\n' + str(Path('src').resolve()) + '\n')
    wheel(tmp_path)
    wheel(tmp_path, extra={'bad.pth': 'import os\n'}, name='assistant_skills_hook')
    wheel(tmp_path, extra={'pip/__init__.py': '# overwrite\n'}, name='assistant_skills_collision')
    wheel(tmp_path, extra={'sitecustomize/__init__.py': 'raise RuntimeError("startup hook")\n'}, name='assistant_skills_package_hook')
    script = tmp_path / 'test.py'
    script.write_text('''
import asyncio, json, sys
from pathlib import Path
from ai_assistant.config import SkillsConfig
from ai_assistant.skills.dependencies import DependencyService, inventory
from ai_assistant.skills.catalog import SkillError
class Offline(DependencyService):
    fail_smoke = False
    def run(self, args, **kwargs):
        if self.fail_smoke and '-c' in args:
            raise SkillError('fixture validation failed')
        if 'pip' in args and ('install' in args or 'download' in args):
            args = [*args, '--no-index', '--find-links', sys.argv[1]]
        return super().run(args, **kwargs)
async def main():
    root = Path(sys.argv[1])
    service = Offline(root, SkillsConfig(dependency_timeout_seconds=90))
    service.sandbox_read_paths = [root]
    import sysconfig
    target = Path(sysconfig.get_path('purelib'))
    before = inventory(target)
    service.fail_smoke = True
    try:
        await service.install('pip', ['assistant-skills-fixture==1.0'])
    except SkillError as exc:
        assert 'validation failed' in str(exc)
    else:
        raise AssertionError('validation failure published')
    assert inventory(target) == before
    service.fail_smoke = False
    result = await service.install('pip', ['assistant-skills-fixture==1.0'])
    assert result['added'] == ['assistant-skills-fixture==1.0']
    existing = inventory(target)
    assert (await service.install('pip', ['assistant-skills-fixture==1.0']))['status'] == 'already satisfied'
    assert inventory(target) == existing
    try:
        await service.install('pip', ['assistant-skills-fixture==2.0'])
    except SkillError:
        pass
    else:
        raise AssertionError('existing version changed')
    assert inventory(target) == existing
    for requirement in ('assistant-skills-hook==1.0', 'assistant-skills-collision==1.0', 'assistant-skills-package-hook==1.0',
                        'https://example.invalid/source.whl', '--force-reinstall'):
        try:
            await service.install('pip', [requirement])
        except SkillError:
            pass
        else:
            raise AssertionError('unsafe requirement published: ' + requirement)
        assert inventory(target) == existing
    print('OFFLINE_PIP_OK')
asyncio.run(main())
''')
    result = subprocess.run([str(env / 'bin/python'), str(script), str(tmp_path)], capture_output=True, text=True,
                            env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'OFFLINE_PIP_OK' in result.stdout


def test_journal_cross_process_lock_and_external_manifest(tmp_path):
    with file_lock(tmp_path / 'lock'):
        with pytest.raises(SkillError, match='Another'):
            with file_lock(tmp_path / 'lock'):
                pass
    path = tmp_path / 'package.json'
    path.write_text('old')
    staged = tmp_path / 'new'
    staged.write_text('new')
    tx = Transaction(tmp_path / 'journal', [tmp_path])
    tx.replace(path, staged)
    path.write_text('external')
    with pytest.raises(SkillError, match='externally'):
        tx.recover()
    assert path.read_text() == 'external'


def test_cancellation_drains_worker_and_releases_lock(tmp_path):
    import threading
    class Waiting(DependencyService):
        def _install(self, manager, requirements):
            self.entered.set()
            assert self.cancelled.wait(5)
            self.exited.set()
            raise SkillError('cancelled')
    async def run():
        service = Waiting(tmp_path, SkillsConfig())
        service.entered, service.exited = threading.Event(), threading.Event()
        task = asyncio.create_task(service.install('pip', ['example']))
        await asyncio.to_thread(service.entered.wait, 2)
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service.exited.is_set()
        assert not service.lock.locked()
    asyncio.run(run())


def test_native_npm_offline_add_only(tmp_path):
    npm, node = shutil.which('npm'), shutil.which('node')
    if not npm or not node:
        pytest.skip('Existing npm/Node runtime unavailable')
    # Override only registry package acquisition, with a local tarball generated
    # by npm pack. The product still parses registry-only structured input.
    fixture = tmp_path / 'fixture'
    fixture.mkdir()
    (fixture / 'package.json').write_text(json.dumps({'name': 'assistant-skills-npm-fixture', 'version': '1.0.0'}))
    (fixture / 'index.js').write_text('module.exports = 42;\n')
    for name in ('user-npm-config', 'global-npm-config'):
        (tmp_path / name).touch()
    npm_env = {'PATH': str(Path(node).parent) + ':/usr/bin:/bin', 'HOME': str(tmp_path),
               'NPM_CONFIG_USERCONFIG': str(tmp_path / 'user-npm-config'),
               'NPM_CONFIG_GLOBALCONFIG': str(tmp_path / 'global-npm-config'),
               'NPM_CONFIG_CACHE': str(tmp_path / 'npm-cache')}
    packed = subprocess.run([npm, 'pack', '--ignore-scripts', '--offline', '--json'], cwd=fixture,
                            env=npm_env, capture_output=True, text=True, check=True)
    archive = fixture / json.loads(packed.stdout)[0]['filename']
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'package.json').write_text(json.dumps({'name': 'test-project', 'version': '1.0.0'}))
    (project / 'package-lock.json').write_text(json.dumps({'name': 'test-project', 'version': '1.0.0', 'lockfileVersion': 3,
        'requires': True, 'packages': {'': {'name': 'test-project', 'version': '1.0.0'}}}))
    class Offline(DependencyService):
        def _install(self, manager, requirements):
            # Isolate all journal files from the real virtual environment.
            self.env_root = tmp_path / 'test-environment-journal'
            return super()._install(manager, requirements)
        def run(self, args, **kwargs):
            if args[:2] == ['npm', 'install']:
                args = [*args[:args.index('--')], '--offline', '--', str(archive)]
            return super().run(args, **kwargs)
    async def run():
        service = Offline(tmp_path, SkillsConfig(npm_project='project'))
        service.sandbox_read_paths = [tmp_path]
        result = await service.install('npm', ['assistant-skills-npm-fixture'])
        assert result['manager'] == 'npm'
        assert (project / 'node_modules/assistant-skills-npm-fixture/index.js').is_file()
        existing = inventory(project / 'node_modules')
        await service.install('npm', ['assistant-skills-npm-fixture'])
        assert inventory(project / 'node_modules') == existing
    asyncio.run(run())


def test_crash_journal_recovery(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    source = tmp_path / 'source'
    source.write_text('new')
    code = '''
import os, sys
from pathlib import Path
from ai_assistant.skills.dependencies import Transaction
root = Path(sys.argv[1])
tx = Transaction(root / 'journal', [root / 'live'])
tx.publish(root / 'source', root / 'live/pkg/module.py')
os._exit(7)
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True)
    assert result.returncode == 7
    assert (live / 'pkg/module.py').is_file()
    Transaction(tmp_path / 'journal', [live]).recover()
    assert not (live / 'pkg').exists()


def test_real_dependency_subprocess_timeout_and_readonly_environment(tmp_path):
    import time
    service = DependencyService(tmp_path, SkillsConfig())
    service.temporary = str(tmp_path)
    service.deadline = time.monotonic() + 10
    probe = Path(sys.prefix) / 'assistant-controlled-write-probe'
    assert not probe.exists()
    with pytest.raises(SkillError, match='resolver or validation failed'):
        service.run([sys.executable, '-c', f'from pathlib import Path; Path({str(probe)!r}).write_text("bad")'])
    assert not probe.exists()
    service.deadline = time.monotonic() + .3
    started = time.monotonic()
    with pytest.raises(SkillError, match='timed out'):
        service.run([sys.executable, '-c', 'import time; time.sleep(30)'])
    assert time.monotonic() - started < 5
