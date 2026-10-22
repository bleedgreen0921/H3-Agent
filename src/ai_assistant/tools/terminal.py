"""LangChain ShellSession adapter with a fail-closed Linux Bubblewrap policy.

Framework-internal policy imports are deliberately confined to this module and
covered by integration tests against the pinned LangChain version.
"""

import asyncio
from dataclasses import asdict
import math
import os
import queue
from pathlib import Path
import shlex
import shutil
import signal
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time

from langchain.agents.middleware._execution import BaseExecutionPolicy
from langchain.agents.middleware.shell_tool import ShellSession

from ai_assistant.workspace.files import ReadOnlyWorkspace, WorkspaceError


class TerminalError(ValueError):
    """Safe terminal setup or invocation failure."""


class _CancellableQueue:
    """Wake native output collection after cancellation, without rewriting it.

    Native ShellSession ignores EOF while collecting until its deadline. A
    killed command could otherwise leave cancellation waiting for 30 minutes.
    """

    def __init__(self, source, cancelled):
        self.source = source
        self.cancelled = cancelled

    def put(self, item):
        self.source.put(item)

    def get_nowait(self):
        return self.source.get_nowait()

    def get(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.cancelled.is_set():
                raise TerminalError('命令已取消。')
            remaining = .1 if deadline is None else min(.1, deadline - time.monotonic())
            if remaining <= 0:
                raise queue.Empty
            try:
                return self.source.get(timeout=remaining)
            except queue.Empty:
                continue


class _CancellableShellSession(ShellSession):
    def start(self):
        super().start()
        self._queue = _CancellableQueue(self._queue, self._policy.cancelled)


async def _drain_worker(worker):
    """Join a cancelled call's thread even if its owner is cancelled again."""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if worker.done() and not worker.cancelled():
        # Retrieve any exception so cancellation does not leave an unhandled task.
        worker.exception()


# Search exclusions that are normal inputs/outputs for command-line programs.
_OPERATIONAL = {'node_modules', '__pycache__', 'build', 'dist'}


class BubblewrapExecutionPolicy(BaseExecutionPolicy):
    """Only the process launcher is application-specific; shell I/O stays native."""

    def __init__(self, workspace: ReadOnlyWorkspace, profile: str, venv: Path,
                 temporary: Path, timeout: float):
        super().__init__(command_timeout=timeout, termination_timeout=0.5,
                         max_output_lines=200, max_output_bytes=32768)
        self.workspace = workspace
        self.profile = profile
        self.venv = venv
        self.temporary = temporary
        self._aborted = False
        self.cancelled = threading.Event()
        self._processes: list[subprocess.Popen] = []
        self._lock = threading.Lock()

    def abort(self):
        """Kill tracked namespace supervisors, including any timeout restart."""
        with self._lock:
            self._aborted = True
            self.cancelled.set()
            for process in self._processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def reap(self):
        self.abort()
        for process in self._processes:
            process.wait(timeout=5)

    def _masks(self) -> list[tuple[Path, bool]]:
        masks = []
        pending = [self.workspace.root]
        examined = 0
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as children:
                for child in children:
                    examined += 1
                    if examined > 100000:
                        raise TerminalError('项目目录过大，无法完整落实沙箱遮蔽规则。')
                    path = Path(child.path)
                    relative = path.relative_to(self.workspace.root)
                    if path == self.venv:
                        if any(self.workspace._excluded(Path(part)) for part in relative.parts[:-1]):
                            raise TerminalError('虚拟环境位于被排除目录内。')
                        if self._user_excluded(relative):
                            raise TerminalError('虚拟环境与自定义排除规则冲突。')
                        continue
                    info = child.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        # Do not feed attacker-controlled symlink mount targets to bwrap.
                        raise TerminalError('项目包含符号链接，无法安全生成沙箱挂载；请移除链接或改用独立工作区。')
                    is_directory = stat.S_ISDIR(info.st_mode)
                    policy_path = Path(*(part for part in relative.parts
                                          if part not in _OPERATIONAL and not part.endswith('.egg-info')))
                    blocked = (path == self.workspace.config_path or self._user_excluded(relative)
                               or self.workspace._excluded(policy_path))
                    if not is_directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink > 1):
                        blocked = True
                    if blocked:
                        if not is_directory and not stat.S_ISREG(info.st_mode):
                            raise TerminalError('项目包含无法安全遮蔽的特殊文件。')
                        masks.append((path, is_directory))
                    elif is_directory:
                        pending.append(path)
        return masks

    def _user_excluded(self, relative: Path) -> bool:
        from fnmatch import fnmatchcase
        return any(fnmatchcase(relative.as_posix(), pattern)
                   or any(fnmatchcase(part, pattern) for part in relative.parts)
                   for pattern in self.workspace.exclude_patterns)

    def _command(self, cwd: Path, command) -> list[str]:
        binary = shutil.which('bwrap')
        if sys.platform != 'linux' or binary is None:
            raise TerminalError('终端工具需要 Linux 和可用的 Bubblewrap。')
        if self.workspace.root == Path('/') or self.venv == Path('/'):
            raise TerminalError('不允许将根文件系统作为项目或虚拟环境。')
        if self.workspace.root.is_relative_to(self.venv):
            raise TerminalError('项目根目录不能位于虚拟环境内部。')
        args = [binary, '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts',
                '--die-with-parent', '--new-session', '--cap-drop', 'ALL', '--clearenv']
        if self.profile != 'maintenance':
            args += ['--unshare-net']
        # A new tmpfs root is constructed by bwrap. No host /, HOME or /run mount.
        for name in ('/usr', '/bin', '/sbin', '/lib', '/lib64'):
            path = Path(name)
            if path.is_symlink():
                args += ['--symlink', os.readlink(path), name]
            elif path.exists():
                args += ['--ro-bind', name, name]
        base = Path(sys.base_prefix).resolve()
        if not base.is_relative_to('/usr') and base != Path('/'):
            args += ['--ro-bind', str(base), str(base)]
        for name in ('/etc/ld.so.cache', '/etc/ssl/certs', '/etc/ssl/openssl.cnf',
                     '/etc/passwd', '/etc/group', '/etc/localtime'):
            if Path(name).exists():
                args += ['--ro-bind', str(Path(name).resolve()), name]
        if self.profile == 'maintenance':
            for name in ('/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf'):
                if Path(name).exists():
                    args += ['--ro-bind', str(Path(name).resolve()), name]
        # Mount temporary /tmp first, otherwise it would hide projects/venvs
        # located under the host's /tmp (including test fixtures).
        args += ['--proc', '/proc', '--dev', '/dev', '--bind', str(self.temporary), '/tmp']
        root = str(self.workspace.root)
        args += ['--ro-bind' if self.profile == 'scratch' else '--bind', root, root]
        args += ['--bind' if self.profile == 'maintenance' else '--ro-bind',
                 str(self.venv), str(self.venv)]
        for path, directory in self._masks():
            source = self.temporary / ('empty-dir' if directory else 'empty-file')
            args += ['--ro-bind', str(source), str(path)]
        args += ['--chdir', str(cwd)]
        return args + ['--', *command]

    def spawn(self, *, workspace: Path, env, command):
        args = self._command(workspace, command)
        for key, value in env.items():
            # Insert fixed, host-owned environment before the command separator.
            position = args.index('--')
            args[position:position] = ['--setenv', key, value]
        with self._lock:
            if self._aborted:
                raise TerminalError('命令已取消。')
            process = subprocess.Popen(
                args, cwd=self.workspace.root, env=dict(env), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding='utf-8', errors='replace', bufsize=1, start_new_session=True,
            )
            self._processes.append(process)
        return process


class TerminalService:
    """One disposable sandbox/session per call; no implicit host fallback."""

    def __init__(self, workspace: ReadOnlyWorkspace, *, profile='project', timeout=120,
                 venv: Path | None = None, lock: asyncio.Lock | None = None):
        if profile not in ('project', 'scratch', 'maintenance'):
            raise TerminalError('未知终端权限档位。')
        if not math.isfinite(timeout) or not 0 < timeout <= 1800:
            raise TerminalError('终端超时必须在 0～1800 秒之间。')
        self.workspace = workspace
        self.profile = profile
        self.timeout = timeout
        self.venv = (venv or Path(sys.prefix)).absolute()
        self.lock = lock or asyncio.Lock()
        self._active: set[asyncio.Task] = set()
        self._closed = False

    def _execute(self, policy, command, cwd, timeout):
        env = {'PATH': f'{self.venv}/bin:/usr/bin:/bin', 'HOME': '/tmp/home',
               'TMPDIR': '/tmp/tmp', 'XDG_CACHE_HOME': '/tmp/cache',
               'PIP_CACHE_DIR': '/tmp/cache/pip', 'PIP_CONFIG_FILE': '/dev/null',
               'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'VIRTUAL_ENV': str(self.venv),
               'PYTHONDONTWRITEBYTECODE': '1'}
        session = _CancellableShellSession(cwd, policy, ('/bin/bash', '--noprofile', '--norc'), env)
        try:
            session.start()
            ready = session.execute(':', timeout=min(timeout, 5))
            if ready.timed_out or ready.exit_code != 0:
                raise TerminalError('Bubblewrap 启动失败；未执行命令，也不会回退本机执行。')
            wrapped = ('/bin/bash --noprofile --norc -o pipefail -c ' + shlex.quote(command)
                       + '; _assistant_status=$?; printf "\\n"; (exit "$_assistant_status")')
            result = session.execute(wrapped, timeout=timeout)
            return asdict(result)
        except (OSError, RuntimeError):
            raise TerminalError('沙箱无法运行；请检查 Bubblewrap、挂载规则和内核支持。') from None
        finally:
            # Kill PID namespace supervisor even when bash exited successfully;
            # descendants must not survive a completed call.
            policy.reap()
            session.stop(0.5)
            for reader in (session._stdout_thread, session._stderr_thread):
                if reader is not None:
                    reader.join(timeout=1)
            for process in policy._processes:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    async def run_command(self, command: str, cwd='.', timeout: float | None = None) -> dict:
        if not command.strip() or len(command) > 32000 or '\0' in command:
            raise TerminalError('命令必须非空、不含 NUL，且不超过 32,000 字符。')
        effective = self.timeout if timeout is None else timeout
        if not math.isfinite(effective) or not 0 < effective <= self.timeout:
            raise TerminalError('调用超时不能超过应用配置的上限。')
        task = asyncio.current_task()
        self._active.add(task)
        try:
            async with self.lock:
                if self._closed:
                    raise TerminalError('终端上下文已关闭。')
                if self.profile == 'scratch' and cwd != '.':
                    raise TerminalError('scratch 档位只接受默认临时工作目录。')
                if not (self.venv / 'pyvenv.cfg').is_file() or not (self.venv / 'bin/python').exists():
                    raise TerminalError('终端工具必须在有效的 Python 虚拟环境中运行。')
                if self.venv.is_symlink():
                    raise TerminalError('不支持符号链接虚拟环境根目录。')
                try:
                    directory = self.workspace._path(cwd)
                except WorkspaceError as exc:
                    raise TerminalError(str(exc)) from None
                if not directory.is_dir():
                    raise TerminalError('命令工作目录必须是目录。')
                started = time.monotonic()
                with TemporaryDirectory(prefix='assistant-command-') as temporary:
                    root = Path(temporary)
                    for name in ('home', 'tmp', 'cache', 'empty-dir'):
                        (root / name).mkdir(mode=0o700)
                    (root / 'empty-file').touch(mode=0o600)
                    policy = BubblewrapExecutionPolicy(self.workspace, self.profile, self.venv, root, effective)
                    worker = asyncio.create_task(asyncio.to_thread(
                        self._execute, policy, command,
                        Path('/tmp') if self.profile == 'scratch' else directory, effective))
                    try:
                        result = await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        policy.abort()
                        # Do not clean mounts/temporary files or release the lock
                        # until the underlying execution thread has actually exited.
                        await _drain_worker(worker)
                        raise
                result.update(profile=self.profile, elapsed_seconds=time.monotonic() - started)
                # Native stderr prefixes may add bytes beyond the framework limit.
                output = result['output'].encode('utf-8')
                result['output'] = output[:32768].decode('utf-8', errors='ignore')
                result['truncated_by_bytes'] |= len(output) > 32768
                return result
        finally:
            self._active.discard(task)

    async def aclose(self):
        self._closed = True
        tasks = list(self._active)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
