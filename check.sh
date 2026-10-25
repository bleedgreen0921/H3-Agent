#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/entrypoint.sh"
exec "$project_python" "$project_root/scripts/smoke.py" "$@"
