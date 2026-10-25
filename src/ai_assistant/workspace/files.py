"""Bounded read-only filesystem operations, independent of Agent orchestration."""

from fnmatch import fnmatchcase
from hashlib import sha256
import os
from pathlib import Path
import stat


class WorkspaceError(ValueError):
    """Safe, expected file-access failure."""


MAX_FILE_BYTES = 1024 * 1024
EXCLUDED = {
    'venv', '__pycache__', 'node_modules', 'build', 'dist',
    'assistant.toml', 'assistant.office.toml', 'models.local.toml', 'models.toml', 'credentials', 'credentials.json',
    'credentials.toml', 'secrets.json', 'secrets.toml', 'env',
    'service-account.json', 'service_account.json', 'token.json', 'tokens.json',
    'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519',
}


class ReadOnlyWorkspace:
    """A filesystem access policy, not a sandbox against concurrent local writers."""

    def __init__(self, root: Path, *, exclude_patterns=(), config_path: Path | None = None):
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceError('工作区根目录必须是目录。')
        self.exclude_patterns = tuple(exclude_patterns)
        self.config_path = config_path.resolve() if config_path else None

    def _excluded(self, relative: Path) -> bool:
        return any(
            part.startswith('.') or part.lower() in EXCLUDED
            or part.lower().endswith(('.pem', '.key', '.p12', '.pfx', '.egg-info'))
            or fnmatchcase(part.lower(), '*credentials*')
            or fnmatchcase(part.lower(), '*secrets*')
            for part in relative.parts
        ) or any(fnmatchcase(relative.as_posix(), pattern)
                 or any(fnmatchcase(part, pattern) for part in relative.parts)
                 for pattern in self.exclude_patterns)

    def _is_config(self, path):
        if self.config_path is None:
            return False
        if path == self.config_path:
            return True
        try:
            # APFS 可能不区分大小写及 Unicode 表示，不能仅比较路径字符串。
            return path.samefile(self.config_path)
        except OSError:
            return False

    def _relative(self, value: str) -> Path:
        relative = Path(value)
        if relative.is_absolute() or '..' in relative.parts:
            raise WorkspaceError('仅允许工作区内不含 .. 的相对路径。')
        if self._excluded(relative):
            raise WorkspaceError('该路径不允许访问。')
        if self._is_config(self.root / relative):
            raise WorkspaceError('该路径不允许访问。')
        return relative

    def _path(self, value: str) -> Path:
        relative = self._relative(value)
        current = self.root
        try:
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise WorkspaceError('不允许访问符号链接。')
            resolved = current.resolve(strict=True)
            if not resolved.is_relative_to(self.root) or self._is_config(resolved):
                raise WorkspaceError('该路径不允许访问。')
            mode = current.stat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise WorkspaceError('只允许普通文件和目录。')
        except OSError:
            raise WorkspaceError('路径不存在或无法访问。') from None
        return current

    def _raw(self, path: Path) -> bytes:
        try:
            # O_NOFOLLOW also prevents a final-component symlink swap before open.
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise WorkspaceError('只允许读取普通文件。')
                if info.st_size > MAX_FILE_BYTES:
                    raise WorkspaceError('文件超过 1 MiB 限制。')
                raw = stream.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                raise WorkspaceError('文件超过 1 MiB 限制。')
            if b'\x00' in raw:
                raise WorkspaceError('不支持二进制文件。')
            raw.decode('utf-8-sig')
            return raw
        except UnicodeError:
            raise WorkspaceError('文件不是有效的 UTF-8 文本。') from None
        except OSError:
            raise WorkspaceError('文件无法读取。') from None

    def _text(self, path: Path) -> str:
        return self._raw(path).decode('utf-8-sig')

    def list_directory(self, path='.', limit=100) -> dict:
        target = self._path(path)
        if not target.is_dir():
            raise WorkspaceError('路径必须是目录。')
        entries = []
        examined = 0
        try:
            with os.scandir(target) as children:
                for child in children:
                    examined += 1
                    if examined > 5000:
                        break
                    relative = Path(child.path).relative_to(self.root)
                    try:
                        item = self._path(relative.as_posix())
                        entries.append({'path': relative.as_posix(),
                                        'type': 'directory' if item.is_dir() else 'file'})
                    except WorkspaceError:
                        continue
        except OSError:
            raise WorkspaceError('目录无法读取。') from None
        entries.sort(key=lambda item: item['path'])
        return {'entries': entries[:limit], 'truncated': len(entries) > limit or examined > 5000}

    def read_text(self, path, start_line=1, max_lines=200) -> dict:
        target = self._path(path)
        if not target.is_file():
            raise WorkspaceError('路径必须是普通文件。')
        raw = self._raw(target)
        lines = raw.decode('utf-8-sig').splitlines()
        selected = lines[start_line - 1:start_line - 1 + max_lines]
        content = '\n'.join(f'{number}: {line}' for number, line in enumerate(selected, start_line))
        return {'path': target.relative_to(self.root).as_posix(), 'content': content[:20000],
                'sha256': sha256(raw).hexdigest(),
                'truncated': len(content) > 20000 or start_line - 1 + len(selected) < len(lines)}

    def search_text(self, query, path='.', max_results=50) -> dict:
        start = self._path(path)
        matches, skipped = [], 0
        scanned = total_bytes = content_size = 0
        truncated = False

        def files(target):
            nonlocal scanned, skipped, truncated
            pending = [target]
            while pending:
                target = pending.pop()
                if target.is_file():
                    yield target
                    continue
                try:
                    with os.scandir(target) as children:
                        for child in children:
                            scanned += 1
                            if scanned > 5000:
                                truncated = True
                                return
                            try:
                                item = self._path(Path(child.path).relative_to(self.root).as_posix())
                            except WorkspaceError:
                                skipped += 1
                                continue
                            if item.is_dir():
                                pending.append(item)
                            else:
                                yield item
                except OSError:
                    skipped += 1

        iterator = files(start)
        try:
            for item in iterator:
                try:
                    size = item.stat().st_size
                    if size > MAX_FILE_BYTES:
                        skipped += 1
                        continue
                    if total_bytes + size > 20 * MAX_FILE_BYTES:
                        truncated = True
                        break
                    total_bytes += size
                    text = self._text(item)
                except (WorkspaceError, OSError):
                    skipped += 1
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    if query not in line:
                        continue
                    if len(matches) >= max_results:
                        truncated = True
                        break
                    relative = item.relative_to(self.root).as_posix()
                    excerpt = line[:500]
                    cost = len(relative) + len(excerpt) + 30
                    if content_size + cost > 20000:
                        truncated = True
                        break
                    matches.append({'path': relative, 'line': number, 'text': excerpt,
                                    'line_truncated': len(line) > 500})
                    content_size += cost
                if truncated:
                    break
        finally:
            iterator.close()
        return {'matches': matches, 'truncated': truncated, 'skipped': skipped}
