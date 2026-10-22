import asyncio
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from ai_assistant.tools.terminal import BubblewrapExecutionPolicy, TerminalError, TerminalService
from ai_assistant.workspace import ReadOnlyWorkspace


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('bwrap'), 'Linux Bubblewrap required')
class TerminalTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory(prefix='sandbox-fixture-')
        cls.base = Path(cls.temp.name)
        cls.venv = cls.base / 'venv'
        subprocess.run([sys.executable, '-m', 'venv', str(cls.venv)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.project_temp = TemporaryDirectory(prefix='sandbox-project-')
        self.addCleanup(self.project_temp.cleanup)
        self.root = Path(self.project_temp.name)
        (self.root / 'input.txt').write_text('input')
        (self.root / 'custom.toml').write_text('SENTINEL_CONFIG')
        (self.root / '.env').write_text('SENTINEL_ENV')
        (self.root / 'private').mkdir()
        (self.root / 'private' / 'data').write_text('SENTINEL_PRIVATE')
        self.workspace = ReadOnlyWorkspace(self.root, config_path=self.root / 'custom.toml',
                                           exclude_patterns=['private'])

    async def run_shell(self, command, profile='project', **kwargs):
        service = TerminalService(self.workspace, profile=profile, timeout=5, venv=self.venv)
        try:
            return await service.run_command(command, **kwargs)
        finally:
            await service.aclose()

    async def test_project_and_scratch_mounts(self):
        code = ('import os,sys; from pathlib import Path; '
                f'assert sys.prefix == {str(self.venv)!r}; '
                f'assert Path({str(self.root / "input.txt")!r}).read_text() == "input"; '
                'assert "SENTINEL_SECRET" not in os.environ; '
                'assert not Path("/run/docker.sock").exists(); '
                'print(Path.cwd())')
        with patch.dict(os.environ, {'SENTINEL_SECRET': 'secret', 'BASH_ENV': '/missing'}):
            result = await self.run_shell('python -c ' + shlex.quote(code))
        self.assertEqual(result['exit_code'], 0, result)
        self.assertIn(str(self.root), result['output'])
        self.assertEqual((await self.run_shell('printf made > output.txt'))['exit_code'], 0)
        self.assertEqual((self.root / 'output.txt').read_text(), 'made')
        result = await self.run_shell('python -c ' + shlex.quote(code), 'scratch')
        self.assertEqual(result['exit_code'], 0, result)
        self.assertIn('/tmp', result['output'])
        result = await self.run_shell(f'printf denied > {shlex.quote(str(self.root / "denied"))}', 'scratch')
        self.assertNotEqual(result['exit_code'], 0)
        self.assertFalse((self.root / 'denied').exists())
        result = await self.run_shell('printf temporary > scratch.txt; test -f scratch.txt', 'scratch')
        self.assertEqual(result['exit_code'], 0, result)
        self.assertFalse((self.root / 'scratch.txt').exists())

    async def test_credentials_and_host_paths_are_not_exposed(self):
        with TemporaryDirectory(prefix='host-sentinel-') as host:
            sentinel = Path(host) / 'secret'
            sentinel.write_text('HOST_SECRET')
            code = ('from pathlib import Path; '
                    f'assert not Path({str(sentinel)!r}).exists(); '
                    'assert Path("custom.toml").read_text() == ""; '
                    'assert Path(".env").read_text() == ""; '
                    'assert list(Path("private").iterdir()) == []; '
                    f'assert not Path({str(Path.home() / ".ssh")!r}).exists(); '
                    'print("masked")')
            for profile in ('project', 'maintenance'):
                result = await self.run_shell('python -c ' + shlex.quote(code), profile)
                self.assertEqual(result['exit_code'], 0, result)
                self.assertNotEqual((await self.run_shell('echo fail > custom.toml', profile))['exit_code'], 0)
        self.assertEqual((self.root / 'custom.toml').read_text(), 'SENTINEL_CONFIG')

    async def test_virtualenv_permissions_and_network(self):
        target = shlex.quote(str(self.venv / 'sandbox-write-probe'))
        for profile in ('project', 'scratch'):
            result = await self.run_shell(f'echo fail > {target}', profile)
            self.assertNotEqual(result['exit_code'], 0)
            code = ('import socket; s=socket.socket(); '
                    's.settimeout(.1); '
                    'print(s.connect_ex(("192.0.2.1", 9)))')
            result = await self.run_shell('python -c ' + shlex.quote(code), profile)
            self.assertEqual(result['exit_code'], 0, result)
            # No route in isolated namespace; this never contacts the Internet.
            self.assertNotIn('\n0\n', '\n' + result['output'])
        result = await self.run_shell(f'echo allowed > {target}; rm {target}', 'maintenance')
        self.assertEqual(result['exit_code'], 0, result)
        code = ('from pathlib import Path; '
                'print(Path("/proc/self/ns/net").readlink())')
        host_network = str(Path('/proc/self/ns/net').readlink())
        self.assertIn(host_network, (await self.run_shell('python -c ' + shlex.quote(code), 'maintenance'))['output'])

    async def test_pipelines_exit_codes_output_timeout_and_cleanup(self):
        result = await self.run_shell('printf "hello" | tr a-z A-Z')
        self.assertEqual(result['exit_code'], 0)
        self.assertIn('HELLO', result['output'])
        self.assertNotEqual((await self.run_shell('false | true'))['exit_code'], 0)
        result = await self.run_shell('python -c ' + shlex.quote('print("x\\n" * 300)'))
        self.assertTrue(result['truncated_by_lines'])
        result = await self.run_shell('python -c ' + shlex.quote('print("x" * 40000)'))
        self.assertTrue(result['truncated_by_bytes'])
        self.assertLessEqual(len(result['output'].encode()), 32768)
        paths = []
        processes = []
        real_temp = TemporaryDirectory
        original_spawn = BubblewrapExecutionPolicy.spawn
        def temporary(**kwargs):
            instance = real_temp(**kwargs)
            paths.append(Path(instance.name))
            return instance
        def spawn(policy, **kwargs):
            process = original_spawn(policy, **kwargs)
            processes.append(process)
            return process
        with patch('ai_assistant.tools.terminal.TemporaryDirectory', side_effect=temporary), \
             patch.object(BubblewrapExecutionPolicy, 'spawn', new=spawn):
            result = await self.run_shell('sleep 20', timeout=.3)
        self.assertTrue(result['timed_out'])
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertTrue(all(not path.exists() for path in paths))

    async def test_cancel_context_close_and_serialization(self):
        service = TerminalService(self.workspace, timeout=5, venv=self.venv)
        task = asyncio.create_task(service.run_command('sleep 20'))
        await asyncio.sleep(.15)
        cancelled_at = time.monotonic()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertLess(time.monotonic() - cancelled_at, 2)
        self.assertEqual((await service.run_command('true'))['exit_code'], 0)
        first = asyncio.create_task(service.run_command('sleep .2; echo first >> order'))
        second = asyncio.create_task(service.run_command('echo second >> order'))
        await asyncio.gather(first, second)
        self.assertEqual((self.root / 'order').read_text().splitlines(), ['first', 'second'])
        task = asyncio.create_task(service.run_command('sleep 20'))
        await asyncio.sleep(.15)
        await service.aclose()
        self.assertTrue(task.cancelled())
        with self.assertRaises(TerminalError):
            await service.run_command('true')

    async def test_fail_closed_and_input_limits(self):
        for command, kwargs in [(' ', {}), ('x' * 32001, {}), ('true', {'cwd': '..'}),
                                ('true', {'timeout': 6}), ('true', {'timeout': float('nan')})]:
            with self.assertRaises(TerminalError):
                await self.run_shell(command, **kwargs)
        with self.assertRaises(TerminalError):
            await self.run_shell('true', 'scratch', cwd='sub')
        with patch('ai_assistant.tools.terminal.shutil.which', return_value=None):
            with self.assertRaises(TerminalError):
                await self.run_shell('touch must-not-exist')
        self.assertFalse((self.root / 'must-not-exist').exists())
        (self.root / 'link').symlink_to(self.root / 'input.txt')
        with self.assertRaises(TerminalError):
            await self.run_shell('touch must-not-exist')
        self.assertFalse((self.root / 'must-not-exist').exists())

    async def test_background_children_die_when_call_returns(self):
        # A detached child in a new session must still die with its PID namespace.
        # Otherwise it would create this fixture file after the tool returned.
        command = ('(setsid /bin/bash -c ' + shlex.quote('sleep .3; echo leaked > child-survived')
                   + ' </dev/null >/dev/null 2>&1 &) ; echo finished')
        result = await self.run_shell(command)
        self.assertEqual(result['exit_code'], 0, result)
        await asyncio.sleep(.5)
        self.assertFalse((self.root / 'child-survived').exists())

    async def test_local_wheel_install_only_in_maintenance(self):
        wheel = self.root / 'assistant_fixture_package-0.0.1-py3-none-any.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('assistant_fixture_package.py', 'VALUE = 42\n')
            prefix = 'assistant_fixture_package-0.0.1.dist-info/'
            archive.writestr(prefix + 'METADATA', 'Metadata-Version: 2.1\nName: assistant-fixture-package\nVersion: 0.0.1\n')
            archive.writestr(prefix + 'WHEEL', 'Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
            archive.writestr(prefix + 'RECORD', '')
        command = f'python -m pip install --no-index --no-deps --no-cache-dir --force-reinstall {shlex.quote(str(wheel))}'
        for profile in ('project', 'scratch'):
            result = await self.run_shell(command, profile)
            self.assertNotEqual(result['exit_code'], 0, result)
        result = await self.run_shell(command, 'maintenance')
        self.assertEqual(result['exit_code'], 0, result)
        result = await self.run_shell('python -c "import assistant_fixture_package as p; assert p.VALUE == 42"')
        self.assertEqual(result['exit_code'], 0, result)
