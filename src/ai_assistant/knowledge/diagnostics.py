"""Bounded, private local diagnostics associated with an ingestion attempt."""
import hashlib
import os
from pathlib import Path
import stat
from uuid import UUID

MAX_BYTES = 64 * 1024
DISPLAY_CHARS = 16000
MARKER = '[诊断已截断]\n'


def bounded(text: str, limit: int = MAX_BYTES) -> str:
    raw = text.encode('utf-8', errors='replace')
    if len(raw) <= limit:
        return raw.decode('utf-8')
    budget = limit - len(MARKER.encode())
    if budget <= 0:
        return MARKER.encode()[:max(0, limit)].decode('utf-8', errors='ignore')
    return MARKER + raw[-budget:].decode('utf-8', errors='ignore')


def attempt_path(config, job_id: str, build_id: str) -> Path:
    kb = hashlib.sha256(config.knowledge_base_id.encode()).hexdigest()
    return Path(config.diagnostics_dir).absolute() / kb / str(UUID(job_id)) / (str(UUID(build_id)) + '.log')


def write_diagnostic(config, job, code: str, message: str, detail: str) -> str:
    path = attempt_path(config, job['id'], job['build_id'])
    # Reject directory symlinks, including those inside the configured root.
    current = Path(path.anchor)
    for part in path.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise OSError('Diagnostic directory must not be a symlink')
        current.mkdir(mode=0o700, exist_ok=True)
    root = Path(config.diagnostics_dir).absolute()
    for directory in (root, path.parent.parent, path.parent):
        directory.chmod(0o700)
    header = (f'job_id={job["id"]}\nbuild_id={job["build_id"]}\n'
              f'attempt={job.get("attempts", "unknown")}\nerror_code={code}\nmessage={message}\n')
    # Retain the identifying header even when the diagnostic tail is truncated.
    body = bounded(detail, MAX_BYTES - len(header.encode()))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(header + body)
    return str(path)


def read_diagnostic(config, status: dict) -> str:
    value = status.get('diagnostic_path')
    build_id = status.get('latest_build_id')
    if not value or not build_id:
        return '该任务最近一次尝试没有可用诊断文件。'
    try:
        expected = attempt_path(config, status['id'], build_id)
    except (ValueError, KeyError):
        return '诊断引用不在当前知识库的受控目录中。'
    if Path(value) != expected:
        return '诊断引用不在当前知识库的受控目录中。'
    # Open each component relative to its parent FD, refusing symlinks throughout.
    descriptor = os.open(expected.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in expected.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        fd = os.open(expected.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return '诊断目标不是普通文件。'
        with os.fdopen(fd, 'rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        text = raw[:MAX_BYTES].decode('utf-8', errors='replace')
        if len(text) > DISPLAY_CHARS or len(raw) > MAX_BYTES:
            marker = '\n[诊断显示已截断]'
            text = text[:DISPLAY_CHARS-len(marker)] + marker
        return text
    except OSError:
        return '诊断文件不存在、不可访问或路径包含符号链接。'
    finally:
        os.close(descriptor)
