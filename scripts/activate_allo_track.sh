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
CONDA_BASE="${CONDA_ROOT:-}"
if [[ -z "${CONDA_BASE}" && -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
fi
if [[ -z "${CONDA_BASE}" ]]; then
    for candidate in "/home/ubuntu/miniconda3" "${HOME}/miniconda3" "${HOME}/anaconda3"; do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="${candidate}"
            break
        fi
    done
fi
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "[ERROR] Miniconda was not found." >&2
    return 2 2>/dev/null || exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate allo_track || {
    echo "[ERROR] Conda environment allo_track was not found." >&2
    return 2 2>/dev/null || exit 2
}

export DEXGYM_ROOT="${REPO_ROOT}"
export DEXGYM_SOURCE_ROOT="${REPO_ROOT}/source"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${REPO_ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}"
echo "Activated Conda environment: ${CONDA_DEFAULT_ENV}"
echo "DexGym root: ${DEXGYM_ROOT}"
echo "Isaac Sim runtime: external; this shell uses allo_track for tests/tools"
