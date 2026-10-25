#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/entrypoint.sh"
exec "$project_python" -m ai_assistant "$@"
