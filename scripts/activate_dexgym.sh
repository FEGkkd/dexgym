#!/usr/bin/env bash

if [[ -n "${BASH_VERSION:-}" ]]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    SCRIPT_PATH="${(%):-%x}"
else
    echo "This script must be sourced from bash or zsh." >&2
    return 1 2>/dev/null || exit 1
fi

REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
export DEXGYM_ROOT="${REPO_ROOT}"
export DEXGYM_CONDA_ENV_NAME="${DEXGYM_CONDA_ENV_NAME:-dexgym-isaaclab}"

CONDA_BASE="${CONDA_ROOT:-}"
if [[ -z "${CONDA_BASE}" && -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
fi
if [[ -z "${CONDA_BASE}" ]]; then
    CONDA_COMMAND="$(command -v conda || true)"
    if [[ "${CONDA_COMMAND}" == /* ]]; then
        CONDA_BASE="$(cd "$(dirname "${CONDA_COMMAND}")/.." && pwd)"
    fi
fi
if [[ -z "${CONDA_BASE}" ]]; then
    for conda_root in "${HOME}/miniconda3" "${HOME}/anaconda3"; do
        if [[ -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="${conda_root}"
            break
        fi
    done
fi
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${DEXGYM_CONDA_ENV_NAME}" ]]; then
        conda activate "${DEXGYM_CONDA_ENV_NAME}" || {
            echo "[ERROR] Failed to activate Conda environment: ${DEXGYM_CONDA_ENV_NAME}" >&2
            return 2 2>/dev/null || exit 2
        }
    fi
else
    echo "[ERROR] Conda was not found. Run scripts/setup_conda.sh first." >&2
    return 2 2>/dev/null || exit 2
fi

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

if [[ -z "${WUJI_DESCRIPTION_ROOT:-}" ]]; then
    for candidate in \
        "${REPO_ROOT}/../wuji-description" \
        "${HOME}/wuji-description"; do
        if [[ -d "${candidate}" ]]; then
            export WUJI_DESCRIPTION_ROOT="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi

if [[ -z "${ISAACLAB_PYTHON:-}" ]]; then
    for candidate in \
        "${REPO_ROOT}/../.venv-isaacsim/bin/python" \
        "${HOME}/.venv-isaacsim/bin/python" \
        "${HOME}/dex/.venv-isaacsim/bin/python"; do
        if [[ -x "${candidate}" ]]; then
            export ISAACLAB_PYTHON="${candidate}"
            break
        fi
    done
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export DEXGYM_SOURCE_ROOT="${REPO_ROOT}/source"
export PYTHONPATH="${REPO_ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${ISAACLAB_ROOT:-}" ]]; then
    export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${PYTHONPATH}"
fi

echo "Activated Conda environment: ${CONDA_DEFAULT_ENV}"
echo "DexGym root: ${DEXGYM_ROOT}"
echo "Isaac Lab root: ${ISAACLAB_ROOT:-not set}"
echo "Isaac Sim Python: ${ISAACLAB_PYTHON:-not set}"
echo "WUJI description: ${WUJI_DESCRIPTION_ROOT:-not set}"
