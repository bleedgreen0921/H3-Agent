"""Native ShellSession with a restricted macOS Seatbelt policy."""

import asyncio
from dataclasses import asdict
import json
from functools import lru_cache
import math
import os
from pathlib import Path
import queue
import shlex
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time

from langchain.agents.middleware._execution import BaseExecutionPolicy
from langchain.agents.middleware.shell_tool import ShellSession
from ai_assistant.cancellation import _drain_worker
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
                raise TerminalError("命令已取消。")
            remaining = (
                0.1 if deadline is None else min(0.1, deadline - time.monotonic())
            )
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


@lru_cache(maxsize=16)
def _linked_runtime_paths(binary):
    """只允许已选运行时实际链接的 Homebrew 库，不开放整个安装前缀。"""
    paths, visited, pending = [], set(), [Path(binary).resolve()]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        result = subprocess.run(
            ["/usr/bin/otool", "-L", str(current)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in result.stdout.splitlines()[1:]:
            value = line.strip().split(" (", 1)[0]
            if value.startswith("@rpath/"):
                path = current.parent.parent / "lib" / value.removeprefix("@rpath/")
            elif value.startswith("@loader_path/"):
                path = current.parent / value.removeprefix("@loader_path/")
            else:
                path = Path(value)
            if (
                path.is_absolute()
                and not str(path).startswith(("/usr/", "/System/"))
                and path.exists()
            ):
                path = path.resolve()
                paths.append(path.parent)
                if path not in visited:
                    pending.append(path)
    return tuple(dict.fromkeys(paths))


class RestrictedExecutionPolicy(BaseExecutionPolicy):
    """Only launch policy is custom; shell I/O and result types stay native."""

    def __init__(
        self,
        workspace,
        venv,
        temporary,
        timeout,
        *,
        allow_network=False,
        read_paths=(),
        skills=(),
    ):
        super().__init__(
            command_timeout=timeout,
            termination_timeout=0.5,
            max_output_lines=200,
            max_output_bytes=32768,
        )
        self.workspace, self.venv = workspace, Path(venv).resolve()
        self.temporary = Path(temporary).resolve()
        self.allow_network = allow_network
        self.skills = list(skills)
        self.read_paths = [
            Path(p).expanduser().resolve(strict=True) for p in read_paths
        ]
        self.cancelled = threading.Event()
        self._processes = []
        self._lock = threading.Lock()

    def abort(self):
        with self._lock:
            self.cancelled.set()
            for process in self._processes:
                # 禁止改变 session/process group，确保后台进程仍受进程组清理约束。
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if process.poll() is None:
                        raise

    def reap(self):
        self.abort()
        for process in self._processes:
            process.wait(timeout=5)

    def _paths(self):
        paths = [
            self.venv,
            Path(sys.base_prefix).resolve(),
            Path(__file__).resolve().parents[2],
        ]
        paths.extend(self.read_paths)
        paths.extend(skill.path.resolve() for skill in self.skills)
        for name in ("node", "npm", "mmx"):
            binary = shutil.which(name)
            if binary:
                target = Path(binary).resolve()
                package = next(
                    (p for p in target.parents if (p / "package.json").is_file()), None
                )
                paths.append(
                    package or (target.parent.parent if name == "node" else target)
                )
                paths.extend(_linked_runtime_paths(target))
        return list(dict.fromkeys(p for p in paths if p.exists() and p != Path("/")))

    def _restrictions(self):
        """共享文件策略；依赖树只读，凭据与排除项拒绝读写。"""
        denied, readonly = [], []
        pending = [self.workspace.root]
        examined = 0
        while pending:
            for path in pending.pop().iterdir():
                examined += 1
                if examined > 100000:
                    raise TerminalError("工作区过大，无法完整落实沙箱规则。")
                relative = path.relative_to(self.workspace.root)
                if path == self.venv:
                    readonly.append(path)
                elif path.name in {
                    "node_modules",
                    "__pycache__",
                    "build",
                    "dist",
                } or path.name.endswith(".egg-info"):
                    if path.is_symlink():
                        raise TerminalError("依赖目录不能是符号链接。")
                    readonly.append(path)
                elif self.workspace._is_config(path) or self.workspace._excluded(
                    relative
                ):
                    denied.append(path)
                elif path.is_symlink():
                    raise TerminalError("工作区包含不受支持的符号链接。")
                elif path.is_dir():
                    pending.append(path)
                elif not path.is_file() or path.stat().st_nlink > 1:
                    raise TerminalError("工作区包含特殊文件或硬链接。")
        return denied, readonly

    def _seatbelt(self, command):
        binary = "/usr/bin/sandbox-exec"
        if not Path(binary).is_file():
            raise TerminalError("缺少 macOS sandbox-exec；不会无沙箱执行。")
        denied, readonly = self._restrictions()
        quote = lambda p: json.dumps(str(Path(p).resolve()), ensure_ascii=False)
        # 不允许改变 session/group，防止 setsid 或 setpgid 脱离取消清理。
        rules = [
            "(version 1)",
            "(deny default)",
            "(allow process-fork process-exec)",
            "(deny process-info-setcontrol)",
            "(deny syscall-unix (syscall-number 147) (syscall-number 82) (syscall-number 244))",
            "(allow sysctl-read)",
            "(allow file-read-metadata)",
            '(allow file-read* (literal "/"))',
            "(allow signal (target self))",
        ]
        for path in [
            "/System/Library",
            "/System/Volumes/Preboot/Cryptexes/OS",
            "/System/Cryptexes",
            "/usr",
            "/bin",
            "/sbin",
            "/Library/Apple/System/Library",
            *self._paths(),
            self.workspace.root,
        ]:
            rules.append(
                f"(allow file-read* file-map-executable (subpath {quote(path)}))"
            )
        rules += [
            f"(allow file-write* (subpath {quote(self.workspace.root)}))",
            f"(allow file-read* file-write* (subpath {quote(self.temporary)}))",
            '(allow file-read* file-write* (literal "/dev/null"))',
            '(allow file-read* (literal "/dev/urandom") (literal "/dev/random"))',
        ]
        for path in [
            self.venv,
            *readonly,
            *(skill.path for skill in self.skills),
            *self.read_paths,
        ]:
            rules.append(f"(deny file-write* (subpath {quote(path)}))")
        for path in denied:
            rules.append(f"(deny file-read-data file-write* (subpath {quote(path)}))")
        if self.allow_network:
            rules += [
                "(allow network*)",
                "(allow system-socket)",
                '(allow mach-lookup (global-name "com.apple.system.opendirectoryd") '
                '(global-name "com.apple.SystemConfiguration.configd") '
                '(global-name "com.apple.networkd"))',
                '(allow file-read* (subpath "/private/etc/ssl") '
                '(literal "/private/etc/hosts") (literal "/private/etc/resolv.conf"))',
            ]
        return [binary, "-p", "\n".join(rules), *command]

    def spawn(self, *, workspace, env, command):
        if (
            self.workspace.root == Path("/")
            or self.venv == Path("/")
            or self.workspace.root.is_relative_to(self.venv)
        ):
            raise TerminalError("工作区或虚拟环境范围无效。")
        if sys.platform != "darwin":
            raise TerminalError("受限终端仅支持 macOS；不会无沙箱执行。")
        args = self._seatbelt(command)
        with self._lock:
            if self.cancelled.is_set():
                raise TerminalError("命令已取消。")
            process = subprocess.Popen(
                args,
                cwd=workspace,
                env=dict(env),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            self._processes.append(process)
        return process


class TerminalService:
    def __init__(
        self,
        workspace: ReadOnlyWorkspace,
        *,
        timeout=120,
        venv=None,
        lock=None,
        allow_network=False,
        env=None,
        read_paths=(),
    ):
        if not math.isfinite(timeout) or not 0 < timeout <= 1800:
            raise TerminalError("终端超时必须在 0～1800 秒之间。")
        self.workspace, self.timeout = workspace, timeout
        self.venv = Path(venv or sys.prefix).resolve()
        self.lock = lock or asyncio.Lock()
        self.allow_network, self.env, self.read_paths = (
            allow_network,
            dict(env or {}),
            list(read_paths),
        )
        if set(self.env) & {
            "PATH",
            "HOME",
            "TMPDIR",
            "BASH_ENV",
            "ENV",
            "PYTHONPATH",
            "PYTHONHOME",
        } or any(k.startswith("DYLD_") for k in self.env):
            raise TerminalError("终端环境变量不能覆盖运行时和启动钩子。")
        self._active, self._closed = set(), False

    def _execute(self, policy, command, cwd, timeout):
        root = policy.temporary
        bins = [str(root / "bin"), str(self.venv / "bin")]
        for name in ("node", "npm", "mmx"):
            binary = shutil.which(name)
            if binary:
                bins.append(str(Path(binary).parent))
        env = {
            **self.env,
            "PATH": ":".join(dict.fromkeys(bins + ["/usr/bin", "/bin"])),
            "HOME": str(root / "home"),
            "TMPDIR": str(root / "tmp"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "PIP_CONFIG_FILE": "/dev/null",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "VIRTUAL_ENV": str(self.venv),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OPENSSL_CONF": "/dev/null",
        }
        session = _CancellableShellSession(
            cwd, policy, ("/bin/bash", "--noprofile", "--norc"), env
        )
        try:
            session.start()
            ready = session.execute(":", timeout=min(timeout, 5))
            if ready.timed_out or ready.exit_code != 0:
                raise TerminalError("沙箱启动失败；不会回退本机执行。")
            wrapped = "/bin/bash --noprofile --norc -o pipefail -c " + shlex.quote(
                command
            )
            result = session.execute(
                wrapped
                + '; _assistant_status=$?; printf "\\n"; (exit "$_assistant_status")',
                timeout=timeout,
            )
            return asdict(result)
        except (OSError, RuntimeError):
            raise TerminalError(
                "受限终端无法运行，请检查平台沙箱及只读运行时路径。"
            ) from None
        finally:
            policy.reap()
            session.stop(0.5)
            for reader in (session._stdout_thread, session._stderr_thread):
                if reader is not None:
                    reader.join(timeout=1)
            for process in policy._processes:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()

    async def run_command(self, command, cwd=".", timeout=None, *, skills=()):
        if sys.platform != "darwin":
            raise TerminalError("受限终端仅支持 macOS；不会无沙箱执行。")
        if not command.strip() or len(command) > 32000 or "\0" in command:
            raise TerminalError("命令必须非空、不含 NUL，且不超过 32,000 字符。")
        effective = self.timeout if timeout is None else timeout
        if not math.isfinite(effective) or not 0 < effective <= self.timeout:
            raise TerminalError("调用超时不能超过配置上限。")
        task = asyncio.current_task()
        self._active.add(task)
        try:
            async with self.lock:
                if self._closed:
                    raise TerminalError("终端上下文已关闭。")
                if not (self.venv / "pyvenv.cfg").is_file():
                    raise TerminalError("终端工具必须使用有效的 Python 虚拟环境。")
                directory = self.workspace._path(cwd)
                if not directory.is_dir():
                    raise TerminalError("命令工作目录必须是目录。")
                started = time.monotonic()
                with TemporaryDirectory(prefix="assistant-command-") as temporary:
                    root = Path(temporary).resolve()
                    for name in ("home", "tmp", "cache", "empty-dir", "bin"):
                        (root / name).mkdir(mode=0o700)
                    (root / "empty-file").touch(mode=0o600)
                    # Homebrew Python 的启动器使用 posix_spawn，改为直接 exec 同一解释器。
                    app = (
                        Path(sys.base_prefix)
                        / "Resources/Python.app/Contents/MacOS/Python"
                    )
                    if app.is_file():
                        launcher = root / "bin/python"
                        launcher.write_text(
                            "#!/bin/bash\nexport __PYVENV_LAUNCHER__="
                            + shlex.quote(str(self.venv / "bin/python"))
                            + "\nexec "
                            + shlex.quote(str(app.resolve()))
                            + ' "$@"\n'
                        )
                        launcher.chmod(0o700)
                        (root / "bin/python3").symlink_to(launcher)

                    try:
                        policy = RestrictedExecutionPolicy(
                            self.workspace,
                            self.venv,
                            root,
                            effective,
                            allow_network=self.allow_network,
                            read_paths=self.read_paths,
                            skills=skills,
                        )
                    except OSError:
                        raise TerminalError("终端只读路径不存在或不可访问。") from None
                    worker = asyncio.create_task(
                        asyncio.to_thread(
                            self._execute, policy, command, directory, effective
                        )
                    )
                    try:
                        result = await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        policy.abort()
                        await _drain_worker(worker)
                        raise
                result.update(
                    profile="restricted", elapsed_seconds=time.monotonic() - started
                )
                raw = result["output"].encode("utf-8")
                result["output"] = raw[:32768].decode("utf-8", errors="ignore")
                for secret in self.env.values():
                    if secret:
                        result["output"] = result["output"].replace(
                            secret, "[REDACTED]"
                        )
                result["truncated_by_bytes"] |= len(raw) > 32768
                return result
        except WorkspaceError as exc:
            raise TerminalError(str(exc)) from None
        finally:
            self._active.discard(task)

    async def aclose(self):
        self._closed = True
        tasks = list(self._active)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
