#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_python="${AI_ASSISTANT_PYTHON:-}"
if [[ -z "$project_python" && -f "$project_root/.python-path" ]]; then
  IFS= read -r project_python < "$project_root/.python-path" || true
fi
if [[ -z "$project_python" || ! -x "$project_python" ]]; then
  echo '请设置 AI_ASSISTANT_PYTHON 或在 .python-path 中填写项目 Python 3.12 解释器的绝对路径。' >&2
  exit 2
fi
if ! "$project_python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo '需要 Python 3.12。' >&2
  exit 2
fi
cd -- "$project_root"
