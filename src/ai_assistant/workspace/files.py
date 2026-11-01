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

    def __init__(self, root: Path, *, exclude_patterns=(), config_path: Path | None = None,
                 max_file_bytes=8 * MAX_FILE_BYTES, result_chars=48000,
                 scan_entries=20000, scan_bytes=128 * 1024 * 1024):
        self.max_file_bytes = max_file_bytes
        self.result_chars = result_chars
        self.scan_entries, self.scan_bytes = scan_entries, scan_bytes
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
                if info.st_size > self.max_file_bytes:
                    raise WorkspaceError(f'文件超过 {self.max_file_bytes} 字节限制。')
                raw = stream.read(self.max_file_bytes + 1)
            if len(raw) > self.max_file_bytes:
                raise WorkspaceError(f'文件超过 {self.max_file_bytes} 字节限制。')
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
                    if examined > self.scan_entries:
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
        return {'entries': entries[:limit], 'truncated': len(entries) > limit or examined > self.scan_entries}

    def read_text(self, path, start_line=1, max_lines=200, start_char=0, char_budget=None) -> dict:
        if not 1 <= max_lines <= 2000 or start_line < 1:
            raise WorkspaceError("行范围无效，每次最多读取 2000 行。")
        if start_char < 0:
            raise WorkspaceError('start_char 不能为负。')
        target = self._path(path)
        if not target.is_file():
            raise WorkspaceError('路径必须是普通文件。')
        raw = self._raw(target)
        lines = raw.decode('utf-8-sig').splitlines()
        window = lines[start_line - 1:start_line - 1 + max_lines]
        if window and start_char:
            if start_char > len(window[0]):
                raise WorkspaceError('start_char 超出行文本长度。')
            # 窗口首行跳过前 start_char 个字符（偏移基于原始行文本，不含行号前缀）。
            window[0] = window[0][start_char:]
        budget = self.result_chars if char_budget is None else char_budget
        # 逐行渲染并在字符预算内截断；next 记录可前进的续读位置（基于原始行文本）。
        parts, next_pos, consumed = [], None, 0
        for offset, text in enumerate(window):
            number, prefix = start_line + offset, f'{start_line + offset}: '
            # base：该行渲染文本在原始行内的起始偏移（首行受 start_char 影响）。
            base = start_char if offset == 0 else 0
            if consumed + len(prefix) + len(text) > budget:
                room = budget - consumed - len(prefix)
                if room <= 0:
                    # 预算落在行号前缀内（或行尾换行边界）：不输出该行，续读指向该行原起点。
                    next_pos = {'start_line': number, 'start_char': base}
                else:
                    parts.append(prefix + text[:room])
                    next_pos = {'start_line': number, 'start_char': base + room}
                break
            parts.append(prefix + text)
            consumed += len(prefix) + len(text) + 1  # 行间换行符
        if next_pos is None and start_line - 1 + len(window) < len(lines):
            # 行窗口截断：续读从窗口后第一行行首开始。
            next_pos = {'start_line': start_line + len(window), 'start_char': 0}
        return {'path': target.relative_to(self.root).as_posix(), 'content': '\n'.join(parts),
                'sha256': sha256(raw).hexdigest(), 'truncated': next_pos is not None,
                **({'next': next_pos} if next_pos else {})}

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
                            if scanned > self.scan_entries:
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
                    if size > self.max_file_bytes:
                        skipped += 1
                        continue
                    if total_bytes + size > self.scan_bytes:
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
                    idx = line.find(query)
                    if len(line) <= 500:
                        start, excerpt = 0, line
                    else:
                        # 摘要窗口：保留命中前上下文，且完整覆盖命中词（长查询时窗口相应扩展）。
                        start = max(0, idx - 120)
                        end = min(len(line), max(start + 500, idx + len(query)))
                        start = max(0, end - 500)
                        excerpt = line[start:end]
                    cost = len(relative) + len(excerpt) + 30
                    if content_size + cost > self.result_chars:
                        truncated = True
                        break
                    matches.append({'path': relative, 'line': number, 'text': excerpt,
                                    'match_offset': idx - start, 'line_start_char': start,
                                    'line_truncated': len(line) > len(excerpt)})
                    content_size += cost
                if truncated:
                    break
        finally:
            iterator.close()
        return {'matches': matches, 'truncated': truncated, 'skipped': skipped}
