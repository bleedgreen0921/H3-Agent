#!/usr/bin/env bash
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 2
cd -- "$project_root" || exit 2

project_python="${AI_ASSISTANT_PYTHON:-}"
if [[ -z "$project_python" && -r .python-path ]]; then
    IFS= read -r project_python < .python-path || :
fi
if [[ -z "$project_python" || ! -f "$project_python" || ! -x "$project_python" ]]; then
    echo "请通过 AI_ASSISTANT_PYTHON 或 .python-path 指定可执行的 Python 3.12。" >&2
    exit 2
fi
if ! "$project_python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1; then
    echo "指定解释器无法运行或不是 Python 3.12。" >&2
    exit 2
fi

exec "$project_python" -m ai_assistant "$@"
