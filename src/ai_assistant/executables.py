"""Resolve trusted executables before constructing a minimal child environment."""

import os
from pathlib import Path
import shutil
from ai_assistant.config import ConfigError


def executable(name, *, path=None):
    found = shutil.which(name, path=path)
    if found is None:
        raise ConfigError("找不到所需程序；请安装并配置 PATH 或绝对路径。")
    return str(Path(found).absolute())


def executable_env(binary):
    return {
        "PATH": str(Path(binary).parent) + os.pathsep + os.defpath,
        "LANG": "en_US.UTF-8",
    }
