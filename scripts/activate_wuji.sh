#!/usr/bin/env bash
# Source this file before running WUJI training or play.
if [[ -n "${BASH_VERSION:-}" ]]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
    SCRIPT_PATH="${(%):-%x}"
else
    echo "[ERROR] This script must be sourced from bash or zsh." >&2
    return 1 2>/dev/null || exit 1
fi

REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
CONDA_BASE="${CONDA_ROOT:-/home/ubuntu/miniconda3}"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "[ERROR] Conda initialization script not found under ${CONDA_BASE}." >&2
    return 2 2>/dev/null || exit 2
fi
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate wuji || {
    echo "[ERROR] Conda environment wuji was not found." >&2
    return 2 2>/dev/null || exit 2
}

export DEXGYM_ROOT="${REPO_ROOT}"
export DEXGYM_SOURCE_ROOT="${REPO_ROOT}/source"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/ubuntu/IsaacLab}"
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-/home/ubuntu/isaacsim-6.0.1}"
export ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-${CONDA_PREFIX}/bin/python}"
export WUJI_DESCRIPTION_ROOT="${WUJI_DESCRIPTION_ROOT:-/home/ubuntu/wuji-description}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export PYTHONPATH="${REPO_ROOT}/source:${ISAACLAB_ROOT}/source/isaaclab_rl:${ISAACLAB_ROOT}/source/isaaclab_contrib:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_physx:${ISAACLAB_ROOT}/source/isaaclab_mimic:${ISAACSIM_ROOT}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ISAACSIM_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for required_path in "${ISAACLAB_ROOT}/source/isaaclab" "${ISAACSIM_ROOT}/lib/python3.12/site-packages" "${WUJI_DESCRIPTION_ROOT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[ERROR] Required local path does not exist: ${required_path}" >&2
        return 2 2>/dev/null || exit 2
    fi
done

echo "Activated Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Isaac Lab root: ${ISAACLAB_ROOT}"
echo "Isaac Sim root: ${ISAACSIM_ROOT}"
echo "Isaac Lab Python: ${ISAACLAB_PYTHON}"
echo "WUJI description: ${WUJI_DESCRIPTION_ROOT}"
