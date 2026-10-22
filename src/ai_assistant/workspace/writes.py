"""Create-only and optimistic, atomic text edits using directory file descriptors."""

from contextlib import contextmanager
import difflib
from hashlib import sha256
import os
import secrets
import stat

from ai_assistant.workspace.files import MAX_FILE_BYTES, ReadOnlyWorkspace, WorkspaceError


class WritableWorkspace:
    """Uses the read workspace's policy; never grants access to excluded paths."""

    def __init__(self, workspace: ReadOnlyWorkspace):
        self.workspace = workspace

    @contextmanager
    def _parent(self, path: str, *, create: bool = False):
        relative = self.workspace._relative(path)
        if not relative.name:
            raise WorkspaceError('必须指定文件路径。')
        fd = os.open(self.workspace.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in relative.parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            yield fd, relative
        finally:
            os.close(fd)

    @staticmethod
    def _encode(content: str) -> bytes:
        try:
            data = content.encode('utf-8')
        except UnicodeError:
            raise WorkspaceError('内容必须是有效 UTF-8 文本。') from None
        if b'\0' in data or len(data) > MAX_FILE_BYTES:
            raise WorkspaceError('内容含 NUL 或超过 1 MiB 限制。')
        return data

    @staticmethod
    def _read(parent: int, name: str) -> tuple[bytes, os.stat_result]:
        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise WorkspaceError('编辑仅支持无硬链接的普通文件。')
            if info.st_size > MAX_FILE_BYTES:
                raise WorkspaceError('文件超过 1 MiB 限制。')
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES or b'\0' in raw:
            raise WorkspaceError('不支持二进制或超大文件。')
        try:
            raw.decode('utf-8-sig')
        except UnicodeError:
            raise WorkspaceError('文件不是有效 UTF-8 文本。') from None
        return raw, info

    @staticmethod
    def _stage(parent: int, data: bytes, mode: int = 0o600) -> str:
        name = f'.assistant-write-{secrets.token_hex(16)}'
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), mode)
                os.fsync(stream.fileno())
        except BaseException:
            os.unlink(name, dir_fd=parent)
            raise
        return name

    def create_file(self, path: str, content: str) -> dict:
        data = self._encode(content)
        try:
            with self._parent(path, create=True) as (parent, relative):
                temp = self._stage(parent, data)
                try:
                    # Atomic publication that never replaces an existing name.
                    os.link(temp, relative.name, src_dir_fd=parent, dst_dir_fd=parent,
                            follow_symlinks=False)
                finally:
                    os.unlink(temp, dir_fd=parent)
                os.fsync(parent)
        except FileExistsError:
            raise WorkspaceError('目标已存在；创建不会覆盖文件。') from None
        except OSError:
            raise WorkspaceError('文件无法创建；请检查路径和访问权限。') from None
        return {'path': relative.as_posix(), 'bytes': len(data), 'sha256': sha256(data).hexdigest()}

    def edit_file(self, path: str, edits: list[dict], expected_sha256: str) -> dict:
        if not 1 <= len(edits) <= 20:
            raise WorkspaceError('每次编辑需要 1～20 项替换。')
        try:
            with self._parent(path) as (parent, relative):
                raw, info = self._read(parent, relative.name)
                before_hash = sha256(raw).hexdigest()
                if before_hash != expected_sha256:
                    raise WorkspaceError('文件摘要不匹配，请重新读取后编辑。')
                text = raw.decode('utf-8-sig')
                matches = []
                for edit in edits:
                    old, new = edit['old_text'], edit['new_text']
                    index = text.find(old)
                    # Search with one-character advance to count overlapping matches too.
                    if not old or index < 0 or text.find(old, index + 1) >= 0:
                        raise WorkspaceError('每段原文必须在文件中恰好匹配一次。')
                    matches.append((index, index + len(old), new))
                matches.sort(key=lambda item: item[0])
                if any(left[1] > right[0] for left, right in zip(matches, matches[1:])):
                    raise WorkspaceError('编辑的原文区间不能重叠。')
                after = text
                for begin, end, replacement in reversed(matches):
                    after = after[:begin] + replacement + after[end:]
                data = self._encode(('\ufeff' if raw.startswith(b'\xef\xbb\xbf') else '') + after)
                temp = self._stage(parent, data, stat.S_IMODE(info.st_mode) & 0o777)
                try:
                    current, current_info = self._read(parent, relative.name)
                    if ((info.st_dev, info.st_ino) != (current_info.st_dev, current_info.st_ino)
                            or sha256(current).hexdigest() != before_hash):
                        raise WorkspaceError('文件已发生变化，请重新读取后编辑。')
                    os.replace(temp, relative.name, src_dir_fd=parent, dst_dir_fd=parent)
                    os.fsync(parent)
                finally:
                    try:
                        os.unlink(temp, dir_fd=parent)
                    except FileNotFoundError:
                        pass
        except OSError:
            raise WorkspaceError('文件无法编辑；请检查路径和访问权限。') from None
        diff = ''.join(difflib.unified_diff(text.splitlines(keepends=True),
                                          after.splitlines(keepends=True),
                                          fromfile=relative.as_posix(), tofile=relative.as_posix()))
        return {'path': relative.as_posix(), 'before_sha256': before_hash,
                'sha256': sha256(data).hexdigest(), 'replacements': len(edits),
                'diff': diff[:20000], 'truncated': len(diff) > 20000}
