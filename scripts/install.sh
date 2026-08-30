#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
    echo "[ERROR] Set ISAACLAB_PYTHON or source scripts/activate_dexgym.sh first." >&2
    exit 2
fi

"${ISAACLAB_PYTHON}" -m pip install --editable "${REPO_ROOT}"
"${ISAACLAB_PYTHON}" -m pip install --requirement "${REPO_ROOT}/requirements-dev.txt"
echo "DexGym extension installed into: ${ISAACLAB_PYTHON}"
