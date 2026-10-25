"""Explicit foreground llama.cpp launcher; no downloads or background supervisor."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("embedding", "reranker"))
    parser.add_argument("--config", type=Path, default=Path("models.local.toml"))
    args = parser.parse_args()
    try:
        config = tomllib.loads(args.config.read_text())
        binary = shutil.which(config.get("server", "llama-server"))
        if binary is None:
            raise ValueError(
                "找不到 llama-server，请安装 llama.cpp 或配置 server 绝对路径。"
            )
        path = Path(config[args.service + "_model_file"]).expanduser()
        if not path.is_absolute():
            path = args.config.resolve().parent / path
        if not path.is_file():
            raise ValueError("模型文件不存在；不会自动下载。")
        context = config.get("context_size", 2048)
        port = config.get(
            args.service + "_port", 1236 if args.service == "embedding" else 1235
        )
        if (
            type(context) is not int
            or context < 512
            or type(port) is not int
            or not 1 <= port <= 65535
        ):
            raise ValueError("context_size 或端口配置无效。")
        name = (
            "qwen3-embedding-0.6b"
            if args.service == "embedding"
            else "qwen3-reranker-0.6b"
        )
        name = config.get(args.service + "_model_name", name)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("模型标识不能为空。")
        cmd = [
            binary,
            "-m",
            str(path),
            "--alias",
            name,
            "--embedding" if args.service == "embedding" else "--reranking",
            "--pooling",
            "last" if args.service == "embedding" else "rank",
            "--gpu-layers",
            "all",
            "-c",
            str(context),
            "-b",
            str(context),
            "-ub",
            str(context),
            "-np",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        os.execv(binary, cmd)
    except (OSError, ValueError, KeyError, TypeError):
        print(
            "模型服务启动失败：检查配置、模型文件、llama-server、context_size 和端口。",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
