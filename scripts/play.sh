#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_wuji.sh"
exec "${SCRIPT_DIR}/run_isaaclab.sh" "${SCRIPT_DIR}/rsl_rl/play.py" "$@"
