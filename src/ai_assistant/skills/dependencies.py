"""Add-only wheel/npm transactions with durable rollback manifests.

No installer writes to the live target: native resolution and installation use
staging; publication only copies reviewed new files and npm manifest deltas.
"""
import asyncio
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
import threading
import time
from uuid import uuid4

from .catalog import SkillError
from .manager import file_lock
from ai_assistant.tools.terminal import _drain_worker


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    """Include links and regular files; never follow dependency directory links."""
    root = Path(root)
    result = {}
    if root.is_symlink():
        raise SkillError('Registered dependency tree roots cannot be symlinks.')
    if not root.exists():
        return result
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                if not path.resolve().is_relative_to(root.resolve()):
                    raise SkillError('Dependency link escapes its registered tree.')
                result[relative] = ['link', os.readlink(path)]
            elif path.is_file():
                result[relative] = ['file', sha(path), path.stat().st_mode & 0o777]
            elif not path.is_dir():
                raise SkillError('Special dependency files are unsupported.')
    return result


def unchanged(before, after):
    if any(after.get(path) != value for path, value in before.items()):
        raise SkillError('Existing dependency versions or files would change; transaction rejected.')


class Transaction:
    def __init__(self, root, allowed):
        self.root = Path(root)
        self.path = self.root / 'transaction.json'
        self.allowed = [Path(p).absolute() for p in allowed]
        self.data = {'new': [], 'replaced': [], 'directories': []}

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(self.data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)

    def _valid(self, path):
        path = Path(path)
        if not path.is_absolute() or not any(path.is_relative_to(root) for root in self.allowed):
            raise SkillError('Invalid recovery journal target; manual inspection required.')
        # Parent symlinks would redirect a reviewed path outside its destination.
        if path.parent.resolve() != path.parent:
            raise SkillError('Dependency destination contains a parent symlink.')
        return path

    def recover(self):
        if not self.path.exists():
            return
        self.data = json.loads(self.path.read_text())
        for record in reversed(self.data['replaced']):
            path = self._valid(record['path'])
            if path.is_symlink():
                raise SkillError('Manifest replaced by an external link; recovery stopped.')
            current = sha(path) if path.is_file() else None
            if current == record['new']:
                self._atomic(path, bytes.fromhex(record['original']))
            elif current != record['old']:
                raise SkillError('Manifest changed externally; rollback requires manual inspection.')
        for record in reversed(self.data['new']):
            path = self._valid(record['path'])
            if not path.exists() and not path.is_symlink():
                continue
            if self.fingerprint(path) != record['value']:
                raise SkillError('Published file changed externally; rollback stopped safely.')
            path.unlink()
        for directory in reversed(self.data.get('directories', [])):
            path = self._valid(directory)
            if path.exists():
                try:
                    path.rmdir()
                except OSError:
                    raise SkillError('New dependency directory changed externally; recovery stopped.') from None
        self.path.unlink()
        self.data = {'new': [], 'replaced': [], 'directories': []}

    @staticmethod
    def fingerprint(path):
        if path.is_symlink():
            return ['link', os.readlink(path)]
        if not path.is_file():
            raise SkillError('Dependency file replaced by a special entry; recovery stopped.')
        return ['file', sha(path), path.stat().st_mode & 0o777]

    @staticmethod
    def _atomic(path, content):
        temp = path.with_name('.assistant-publish-' + path.name)
        with temp.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    def publish(self, source, target):
        target = self._valid(str(target))
        if target.exists() or target.is_symlink():
            raise SkillError('Refusing to overwrite an existing package file.')
        missing = []
        parent = target.parent
        while not parent.exists():
            self._valid(str(parent))
            missing.append(parent)
            parent = parent.parent
        for parent in reversed(missing):
            self.data['directories'].append(str(parent))
            self.save()
            parent.mkdir()
        value = self.fingerprint(source)
        self.data['new'].append({'path': str(target), 'value': value})
        self.save()
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            # Build on the target filesystem, then publish a complete file with
            # link(2)'s no-overwrite semantics. A crash cannot expose half a file.
            prepared = self.root / ('prepared-' + uuid4().hex)
            if not prepared.exists():
                with prepared.open('xb') as out, source.open('rb') as incoming:
                    shutil.copyfileobj(incoming, out)
                    out.flush()
                    os.fsync(out.fileno())
                prepared.chmod(source.stat().st_mode & 0o777)
            os.link(prepared, target)
            prepared.unlink()

    def replace(self, path, source, expected=None):
        path = self._valid(str(path))
        if path.is_symlink() or not path.is_file():
            raise SkillError('Dependency manifest is no longer a regular file.')
        original = path.read_bytes()
        if expected is not None and hashlib.sha256(original).hexdigest() != expected:
            raise SkillError('Dependency manifest changed before publication; retry.')
        self.data['replaced'].append(dict(path=str(path), old=sha(path), new=sha(source), original=original.hex()))
        self.save()
        self._atomic(path, source.read_bytes())

    def commit(self):
        self.path.unlink(missing_ok=True)


class DependencyService:
    def __init__(self, workspace, settings, *, lock=None, python=None):
        self.workspace = Path(workspace).resolve()
        self.settings = settings
        self.lock = lock or asyncio.Lock()
        self.python = str(python or sys.executable)
        self.prefix = Path(sys.prefix).absolute()
        self.env_root = self.prefix / '.assistant-dependencies'
        self.project = (self.workspace / settings.npm_project).resolve()
        self.cancelled = threading.Event()
        self.deadline = 0
        self.npm = shutil.which('npm')
        self.node = shutil.which('node')

    def _sandbox(self, args, cwd):
        """Resolvers see only read-only runtimes and writable transaction staging."""
        binary = shutil.which('bwrap')
        if sys.platform != 'linux' or binary is None:
            raise SkillError('Controlled dependency installation requires Linux and Bubblewrap.')
        command = [binary, '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts',
                   '--die-with-parent', '--new-session', '--cap-drop', 'ALL']
        mounted = set()
        def readonly(path):
            path = Path(path).absolute()
            if path in mounted or not path.exists() or path == Path('/'):
                return
            mounted.add(path)
            command.extend(['--ro-bind', str(path.resolve()), str(path)])
        for name in ('/usr', '/bin', '/sbin', '/lib', '/lib64'):
            path = Path(name)
            if path.is_symlink():
                command.extend(['--symlink', os.readlink(path), name])
            else:
                readonly(path)
        if not Path(sys.base_prefix).is_relative_to('/usr'):
            readonly(sys.base_prefix)
        for name in ('/etc/ld.so.cache', '/etc/ssl/certs', '/etc/ssl/openssl.cnf',
                     '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf', '/etc/passwd', '/etc/group'):
            readonly(name)
        command += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
        readonly(self.prefix)
        readonly(Path(__file__).resolve().parents[2])
        # Existing dependency paths, including test-only inherited site packages.
        for entry in sys.path:
            if entry and ('site-packages' in Path(entry).parts or 'dist-packages' in Path(entry).parts):
                readonly(entry)
        if self.node:
            readonly(Path(self.node).resolve().parent.parent)
        for path in getattr(self, 'sandbox_read_paths', ()):
            readonly(path)
        directory = Path(cwd or self.temporary).absolute()
        if not directory.is_relative_to(self.temporary):
            readonly(directory)
        command += ['--bind', self.temporary, self.temporary]
        # Isolated pip still reads site-level config; hide it explicitly.
        empty = Path(self.temporary) / 'empty-config'
        empty.touch(exist_ok=True)
        for path in (self.prefix / 'pip.conf', directory / '.npmrc'):
            if path.is_file():
                command += ['--ro-bind', str(empty), str(path)]
        return command + ['--chdir', str(directory), '--', *args]

    def run(self, args, *, cwd=None, accept=(0,)):
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise SkillError('Dependency transaction cancelled or timed out.')
        runtime_path = str(Path(self.node).parent) + ':' if self.node else ''
        env = {'PATH': runtime_path + str(Path(self.python).parent) + ':/usr/local/bin:/usr/bin:/bin',
               'HOME': self.temporary, 'TMPDIR': self.temporary, 'LANG': 'C.UTF-8',
               'PIP_CONFIG_FILE': '/dev/null', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
               'NPM_CONFIG_USERCONFIG': self.temporary + '/npm-user-config',
               'NPM_CONFIG_GLOBALCONFIG': self.temporary + '/npm-global-config',
               'NPM_CONFIG_CACHE': self.temporary + '/npm-cache', 'NPM_CONFIG_IGNORE_SCRIPTS': 'true'}
        # Output goes to disk, so a hostile resolver cannot exhaust parent memory.
        if args[0] == 'npm':
            if self.npm is None or self.node is None:
                raise SkillError('An existing npm and Node runtime is required.')
            args = [self.npm, *args[1:]]
        with TemporaryDirectory(dir=self.temporary) as output_dir:
            output = Path(output_dir) / 'output'
            with output.open('wb') as stream:
                process = subprocess.Popen(self._sandbox(args, cwd), cwd=self.temporary, env=env, stdout=stream, stderr=stream, start_new_session=True)
                try:
                    while process.poll() is None:
                        if self.cancelled.wait(.05) or time.monotonic() >= self.deadline:
                            raise SkillError('Dependency transaction cancelled or timed out.')
                        if output.stat().st_size > 8 * 1024 * 1024:
                            raise SkillError('Dependency subprocess output limit exceeded.')
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if output.stat().st_size > 8 * 1024 * 1024:
                raise SkillError('Dependency subprocess output limit exceeded.')
            text = output.read_text(errors='replace')
            if process.returncode not in accept:
                # Never return npm config or network error text containing credentials.
                raise SkillError('Dependency resolver or validation failed; existing dependencies were preserved.')
            return text, process.returncode

    async def install(self, manager, requirements):
        if manager not in ('pip', 'npm') or not requirements or len(requirements) > 50:
            raise SkillError('Use pip or npm and 1–50 structured requirements.')
        async with self.lock:
            self.cancelled.clear()
            worker = asyncio.create_task(asyncio.to_thread(self._install, manager, requirements))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                self.cancelled.set()
                await _drain_worker(worker)
                raise

    def _install(self, manager, requirements):
        if self.python != str(sys.executable):
            raise SkillError('Dependency service must use the current Python environment.')
        if not (self.prefix / 'pyvenv.cfg').is_file():
            raise SkillError('Controlled pip installation requires the current existing virtual environment.')
        self.deadline = time.monotonic() + self.settings.dependency_timeout_seconds
        with ExitStack() as stack:
            stack.enter_context(file_lock(self.env_root / 'environment.lock'))
            if self.project.is_relative_to(self.workspace):
                stack.enter_context(file_lock(self.workspace / '.assistant-dependencies/project.lock'))
            elif manager == 'npm':
                raise SkillError('npm_project must be inside the workspace.')
            temporary = stack.enter_context(TemporaryDirectory(prefix='assistant-dependencies-'))
            self.temporary = temporary
            for name in ('npm-user-config', 'npm-global-config'):
                (Path(temporary) / name).touch()
            pip_target = Path(sysconfig.get_path('purelib')).absolute()
            npm_root = self.workspace / '.assistant-dependencies'
            pip_tx = Transaction(self.env_root, [pip_target, self.prefix / 'bin'])
            npm_tx = Transaction(npm_root, [self.project])
            pip_tx.recover()
            if self.project.is_relative_to(self.workspace):
                npm_tx.recover()
            transaction = pip_tx if manager == 'pip' else npm_tx
            try:
                result = self._pip(requirements, transaction, pip_target) if manager == 'pip' else self._npm(requirements, transaction)
                if self.cancelled.is_set() or time.monotonic() >= self.deadline:
                    raise SkillError('Dependency transaction cancelled or timed out.')
                transaction.commit()
                return result
            except BaseException:
                transaction.recover()
                raise

    def _pip(self, requirements, transaction, target):
        from packaging.requirements import Requirement, InvalidRequirement
        from packaging.utils import canonicalize_name
        parsed = []
        for value in requirements:
            try:
                requirement = Requirement(value)
            except (InvalidRequirement, TypeError):
                raise SkillError('Invalid Python package requirement.') from None
            if requirement.url or value.startswith('-'):
                raise SkillError('URLs, VCS, editable and shell options are unsupported.')
            parsed.append(str(requirement))
        baseline = {canonicalize_name(d.metadata['Name']): d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
        baseline_files = inventory(target)
        before_check, before_status = self.run([self.python, '-m', 'pip', '--isolated', 'check'], accept=(0, 1))
        constraints = Path(self.temporary) / 'constraints.txt'
        constraints.write_text('\n'.join(f'{name}=={version}' for name, version in sorted(baseline.items())))
        report = Path(self.temporary) / 'report.json'
        self.run([self.python, '-m', 'pip', '--isolated', 'install', '--dry-run', '--report', str(report),
                  '--only-binary=:all:', '--constraint', str(constraints), *parsed])
        plan = json.loads(report.read_text())['install']
        for item in plan:
            name = canonicalize_name(item['metadata']['name'])
            if name in baseline:
                raise SkillError('Resolver attempted to change an existing Python distribution.')
            if any(Requirement(value).url for value in item['metadata'].get('requires_dist', [])):
                raise SkillError('Transitive direct-URL dependencies are unsupported.')
            if not item['download_info']['url'].split('?')[0].endswith('.whl'):
                raise SkillError('Only wheels are supported.')
        if not plan:
            return dict(manager='pip', added=[], status='already satisfied')
        wheels = Path(self.temporary) / 'wheels'
        wheels.mkdir()
        exact = [f"{item['metadata']['name']}=={item['metadata']['version']}" for item in plan]
        self.run([self.python, '-m', 'pip', '--isolated', 'download', '--only-binary=:all:', '--no-deps', '--dest', str(wheels), *exact])
        expected = {item['download_info']['archive_info']['hashes']['sha256'] for item in plan}
        if {sha(path) for path in wheels.iterdir()} != expected:
            raise SkillError('Resolved and downloaded wheel hashes differ.')
        import zipfile
        from pathlib import PurePosixPath
        for wheel in wheels.iterdir():
            with zipfile.ZipFile(wheel) as archive:
                for entry in archive.infolist():
                    path = PurePosixPath(entry.filename)
                    kind = (entry.external_attr >> 16) & 0o170000
                    if path.is_absolute() or '..' in path.parts or '\\' in entry.filename or kind not in (0, 0o100000, 0o040000):
                        raise SkillError('Unsafe wheel archive path or special file.')
        stage = Path(self.temporary) / 'stage'
        self.run([self.python, '-m', 'pip', '--isolated', 'install', '--no-deps', '--no-compile', '--only-binary=:all:', '--target', str(stage), *map(str, wheels.iterdir())])
        staged = inventory(stage)
        destinations = {}
        for name, value in staged.items():
            p = Path(name)
            if value[0] != 'file' or p.suffix == '.pth' or any(part.split('.')[0] in ('sitecustomize', 'usercustomize') for part in p.parts):
                raise SkillError('Wheel startup hooks and links are unsupported.')
            destination = self.prefix / name if p.parts[0] == 'bin' else target / name
            destinations[name] = destination
            inherited_collision = p.parts[0] != 'bin' and any(
                (Path(entry) / name).is_file() for entry in sys.path
                if entry and Path(entry).absolute() != target and ('site-packages' in Path(entry).parts or 'dist-packages' in Path(entry).parts))
            if destination.exists() or destination.is_symlink() or inherited_collision:
                raise SkillError('Wheel would overwrite or shadow existing package files.')
        if inventory(target) != baseline_files:
            raise SkillError('Python environment changed during resolution; retry.')
        for name in staged:
            transaction.publish(stage / name, destinations[name])
        after_check, after_status = self.run([self.python, '-m', 'pip', '--isolated', 'check'], accept=(0, 1))
        if after_status != before_status or after_check != before_check:
            raise SkillError('New dependency consistency errors; publication rolled back.')
        self.run([self.python, '-c', 'import langchain, langgraph, ai_assistant; import ai_assistant.interfaces.cli'])
        self.run([self.python, '-m', 'ai_assistant', '--help'])
        unchanged(baseline_files, inventory(target))
        current = {canonicalize_name(d.metadata['Name']): d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
        if any(current.get(name) != version for name, version in baseline.items()):
            raise SkillError('Existing Python distribution changed.')
        return dict(manager='pip', added=exact, wheel_sha256=sorted(expected))

    def _npm(self, requirements, transaction):
        if not self.project.is_relative_to(self.workspace):
            raise SkillError('npm_project must be within the workspace.')
        for value in requirements:
            if not isinstance(value, str) or not re.fullmatch(r'(?:@[a-z0-9_.-]+/)?[a-z0-9][a-z0-9_.-]*(?:@[~^<>=*|. 0-9A-Za-z+-]+)?', value):
                raise SkillError('Only registry npm package names and version ranges are supported.')
        manifests = [self.project / 'package.json', self.project / 'package-lock.json']
        if not all(p.is_file() and not p.is_symlink() for p in manifests):
            raise SkillError('npm requires an existing package.json and npm lockfile v2/v3.')
        package, lock = [json.loads(p.read_text()) for p in manifests]
        if package.get('workspaces') or lock.get('lockfileVersion') not in (2, 3):
            raise SkillError('Only single-package npm projects with lockfile v2/v3 are supported.')
        for entry in lock.get('packages', {}).values():
            if entry.get('link') or entry.get('hasInstallScript'):
                raise SkillError('npm workspaces, local dependencies and installation hooks are unsupported.')
        baseline_manifests = [sha(p) for p in manifests]
        tree = self.project / 'node_modules'
        before = inventory(tree)
        internal_before = before.pop('.package-lock.json', None)
        stage = Path(self.temporary) / 'npm-project'
        stage.mkdir()
        for path in manifests:
            shutil.copyfile(path, stage / path.name)
        if tree.exists():
            shutil.copytree(tree, stage / 'node_modules', symlinks=True)
        self.run(['npm', 'install', '--ignore-scripts', '--no-audit', '--no-fund', '--', *requirements], cwd=stage)
        new_lock = json.loads((stage / 'package-lock.json').read_text())
        for name, entry in lock.get('packages', {}).items():
            if name and new_lock.get('packages', {}).get(name) != entry:
                raise SkillError('npm would change an existing lockfile resolution.')
        for name, entry in lock.get('dependencies', {}).items():
            if new_lock.get('dependencies', {}).get(name) != entry:
                raise SkillError('npm would change an existing legacy lockfile resolution.')
        for path in (stage / 'node_modules').rglob('package.json'):
            scripts = json.loads(path.read_text()).get('scripts', {})
            if any(key in scripts for key in ('preinstall', 'install', 'postinstall', 'prepare')):
                raise SkillError('Packages requiring installation hooks are unsupported.')
        staged_package = json.loads((stage / 'package.json').read_text())
        for key, value in package.items():
            if key in ('dependencies', 'devDependencies', 'optionalDependencies'):
                if any(staged_package.get(key, {}).get(name) != version for name, version in value.items()):
                    raise SkillError('npm would change an existing package requirement.')
            elif staged_package.get(key) != value:
                raise SkillError('npm would change existing project metadata.')
        after = inventory(stage / 'node_modules')
        internal_after = after.pop('.package-lock.json', None)
        unchanged(before, after)
        # npm rewrites its internal lock; it too is an existing file and cannot change.
        current_tree = inventory(tree)
        current_internal = current_tree.pop('.package-lock.json', None)
        if current_tree != before or current_internal != internal_before or [sha(p) for p in manifests] != baseline_manifests:
            raise SkillError('npm project changed during resolution; retry.')
        self.run(['npm', 'ls', '--all', '--json', '--ignore-scripts'], cwd=stage)
        for name in sorted(after.keys() - before.keys(), key=lambda name: after[name][0] == 'link'):
            transaction.publish(stage / 'node_modules' / name, tree / name)
        internal = tree / '.package-lock.json'
        staged_internal = stage / 'node_modules/.package-lock.json'
        if internal_after is not None:
            if internal_before is None:
                transaction.publish(staged_internal, internal)
            elif sha(internal) != sha(staged_internal):
                transaction.replace(internal, staged_internal, expected=internal_before[1])
        for path, expected_hash in zip(manifests, baseline_manifests):
            if sha(path) != sha(stage / path.name):
                transaction.replace(path, stage / path.name, expected=expected_hash)
        self.run(['npm', 'ls', '--all', '--json', '--ignore-scripts'], cwd=self.project)
        return dict(manager='npm', added=sorted(after.keys() - before.keys()))
