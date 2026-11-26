"""Explicit read-only source adapters; cloud content never becomes instructions."""

import fnmatch
import hashlib
import json
import os
from pathlib import Path
from ai_assistant.documents.formats import RAG_SCAN_EXTENSIONS
from ai_assistant.knowledge import KnowledgeError
from ai_assistant.workspace.files import ReadOnlyWorkspace, WorkspaceError

# 配置来源可扫描的本地文件扩展名及飞书只读接口地址。
EXTENSIONS = RAG_SCAN_EXTENSIONS


class SourceFailure(KnowledgeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def local_descriptor(path, root):
    """构造本地文件来源描述。"""
    path, root = Path(path), Path(root)
    return {
        "kind": "local",
        "path": str(path.resolve()),
        "root": str(root.resolve()),
        "name": path.name,
    }


def local_inventory(config, source):
    """在显式来源目录内枚举允许的文件，排除链接和受保护路径。"""
    base = config.config_path.parent if config.config_path else Path.cwd()
    root = Path(source["path"]).expanduser()
    root = (root if root.is_absolute() else base / root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise SourceFailure("local_directory_unavailable")
    root = root.resolve()
    workspace = ReadOnlyWorkspace(
        root,
        exclude_patterns=config.workspace.exclude_patterns,
        config_path=config.config_path, config_files=config.config_files,
    )
    blocked = [
        Path(getattr(config.knowledge, k)).resolve()
        for k in ("materials_dir", "diagnostics_dir")
    ]
    result = []

    def fail(error):
        raise SourceFailure("local_enumeration_failed")

    # 不跟随目录符号链接，也不扫描隐藏目录和知识库内部资料目录。
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and not (Path(directory) / d).is_symlink()
            and not any(
                (Path(directory) / d).resolve().is_relative_to(p) for p in blocked
            )
        ]
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if not any(
                fnmatch.fnmatch(relative, p) for p in source.get("include", ["*"])
            ):
                continue
            if any(fnmatch.fnmatch(relative, p) for p in source.get("exclude", [])):
                continue
            try:
                checked = workspace._path(relative)
                if checked.is_symlink() or not checked.is_file():
                    continue
            except WorkspaceError:
                continue
            result.append((checked.as_uri(), local_descriptor(checked, root)))
    return result


def save_documents(directory, docs):
    """使用稳定正文与定位信息计算版本，检查时间不参与 hash。"""
    raw = json.dumps(
        [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / (digest + ".ragjson")
    import tempfile

    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.chmod(0o400)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise KnowledgeError("不可变云文档快照校验失败。")
    finally:
        temporary.unlink(missing_ok=True)
    return digest, path
