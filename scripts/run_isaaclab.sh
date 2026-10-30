#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${ISAACLAB_ROOT:-}" ]]; then
    for candidate in \
        "${REPO_ROOT}/../IsaacLab" \
        "${HOME}/IsaacLab"; do
        if [[ -d "${candidate}/source/isaaclab" ]]; then
            export ISAACLAB_ROOT="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi

if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
    candidates=()
    if [[ -n "${ISAACLAB_ROOT:-}" ]]; then
        candidates+=("${ISAACLAB_ROOT}/env_isaaclab/bin/python")
        candidates+=("${ISAACLAB_ROOT}/_isaac_sim/python.sh")
    fi
    candidates+=(
        "${REPO_ROOT}/../.venv-isaacsim/bin/python"
        "${HOME}/.venv-isaacsim/bin/python"
        "${HOME}/dex/.venv-isaacsim/bin/python"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}" ]]; then
            export ISAACLAB_PYTHON="${candidate}"
            break
        fi
    done
fi

if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
    echo "[ERROR] Isaac Sim Python was not found." >&2
    echo "Set ISAACLAB_PYTHON=/path/to/isaacsim/python before running this command." >&2
    exit 2
fi

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
    echo "[ERROR] ISAACLAB_PYTHON is not executable: ${ISAACLAB_PYTHON}" >&2
    exit 2
fi

if [[ -n "${ISAACLAB_ROOT:-}" && ! -d "${ISAACLAB_ROOT}/source/isaaclab" ]]; then
    echo "[ERROR] ISAACLAB_ROOT must contain source/isaaclab: ${ISAACLAB_ROOT}" >&2
    exit 2
fi

export DEXGYM_ROOT="${REPO_ROOT}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export PYTHONPATH="${REPO_ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${ISAACLAB_ROOT:-}" ]]; then
    for extension in isaaclab isaaclab_physx isaaclab_contrib isaaclab_rl isaaclab_tasks isaaclab_visualizers; do
        if [[ -d "${ISAACLAB_ROOT}/source/${extension}" ]]; then
            export PYTHONPATH="${ISAACLAB_ROOT}/source/${extension}:${PYTHONPATH}"
        fi
    done
fi

exec "${ISAACLAB_PYTHON}" "$@"
