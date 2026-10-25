"""Actual platform sandbox checks in disposable workspaces; no business credentials."""

import asyncio
from pathlib import Path
import shlex
import shutil
import socket
import sys

import pytest
from ai_assistant.tools.terminal import TerminalService, TerminalError
from ai_assistant.workspace.files import ReadOnlyWorkspace

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="macOS Seatbelt required",
)


async def run(root, command, **settings):
    service = TerminalService(
        ReadOnlyWorkspace(
            root, config_path=root / "private.toml", exclude_patterns=["private"]
        ),
        timeout=5,
        **settings,
    )
    try:
        return await service.run_command(command)
    finally:
        await service.aclose()


def test_runtime_files_credentials_and_environment(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "private.toml").write_text("SECRET_CONFIG")
    (root / ".env").write_text("SECRET_ENV")
    (root / "private").mkdir()
    (root / "private/token").write_text("SECRET_TOKEN")
    (tmp_path / "outside").write_text("SECRET_OUTSIDE")
    monkeypatch.setenv("PRIVATE_SENTINEL", "SECRET_INHERITED")

    async def scenario():
        code = 'import os,sys; print(sys.prefix); print(os.environ.get("PRIVATE_SENTINEL")); open("output.txt","w").write("ok")'
        result = await run(root, "python -c " + shlex.quote(code))
        assert result["exit_code"] == 0, result
        assert (
            sys.prefix in result["output"]
            and "SECRET_INHERITED" not in result["output"]
        )
        assert (root / "output.txt").read_text() == "ok"
        for path in [
            root / "private.toml",
            root / ".env",
            root / "private/token",
            tmp_path / "outside",
        ]:
            code = f"from pathlib import Path; print(Path({str(path)!r}).read_text())"
            result = await run(root, "python -c " + shlex.quote(code))
            assert "SECRET_" not in result["output"]
            result = await run(root, "echo overwritten > " + shlex.quote(str(path)))
            assert result["exit_code"] != 0
        code = f'from pathlib import Path; Path({str(Path(sys.prefix) / "sandbox-probe")!r}).write_text("bad")'
        assert (await run(root, "python -c " + shlex.quote(code)))["exit_code"] != 0

    asyncio.run(scenario())


def test_network_switch_against_local_listener(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(3)
        port = listener.getsockname()[1]
        code = f'import socket; s=socket.socket(); s.settimeout(1); s.connect(("127.0.0.1",{port})); print("CONNECTED")'

        async def scenario():
            assert (await run(tmp_path, "python -c " + shlex.quote(code)))[
                "exit_code"
            ] != 0
            result = await run(
                tmp_path, "python -c " + shlex.quote(code), allow_network=True
            )
            assert result["exit_code"] == 0 and "CONNECTED" in result["output"], result

        asyncio.run(scenario())


def test_output_pipe_timeout_cancel_and_reuse(tmp_path):
    async def scenario():
        s = TerminalService(ReadOnlyWorkspace(tmp_path), timeout=5)
        try:
            assert (await s.run_command("false | true"))["exit_code"] != 0
            result = await s.run_command("python -c " + shlex.quote('print("x"*50000)'))
            assert (
                result["truncated_by_bytes"] and len(result["output"].encode()) <= 32768
            )
            result = await s.run_command("sleep 10", timeout=0.2)
            assert result["timed_out"]
            task = asyncio.create_task(s.run_command("sleep 10"))
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert (await s.run_command("echo recovered"))["exit_code"] == 0
        finally:
            await s.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "detach", ["setsid", "setpgid", "double_fork", "posix_spawn", "ordinary"]
)
def test_background_children_cannot_escape_cleanup(tmp_path, detach):
    output = tmp_path / "survived"
    action = {
        "setsid": "os.setsid()",
        "setpgid": "os.setpgid(0,0)",
        "double_fork": "os.fork() and os._exit(0); os.setsid()",
        "posix_spawn": f'os.posix_spawn("/bin/sh",["sh","-c","sleep 0.8; touch {output}"],{{}},setsid=True); os._exit(0)',
        "ordinary": "pass",
    }[detach]
    code = f'import os,time; {action}; time.sleep(.8); open({str(output)!r},"w").write("leaked")'
    command = (
        "python -c " + shlex.quote(code) + " </dev/null >/dev/null 2>&1 & echo finished"
    )

    async def scenario():
        assert (await run(tmp_path, command))["exit_code"] == 0
        await asyncio.sleep(1)
        assert not output.exists()

    asyncio.run(scenario())


def test_invalid_inputs_and_no_host_fallback(tmp_path, monkeypatch):
    async def scenario():
        s = TerminalService(ReadOnlyWorkspace(tmp_path), timeout=1)
        try:
            for command, kwargs in [
                (" ", {}),
                ("true", {"cwd": ".."}),
                ("true", {"timeout": 2}),
            ]:
                with pytest.raises(TerminalError):
                    await s.run_command(command, **kwargs)
            monkeypatch.setattr(
                "ai_assistant.tools.terminal.sys.platform", "unsupported"
            )
            with pytest.raises(TerminalError):
                await s.run_command("touch must-not-exist")
            assert not (tmp_path / "must-not-exist").exists()
        finally:
            await s.aclose()

    asyncio.run(scenario())
