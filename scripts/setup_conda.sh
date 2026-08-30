#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_NAME="${DEXGYM_CONDA_ENV_NAME:-dexgym-isaaclab}"

CONDA_BIN="$(command -v conda || true)"
if [[ -z "${CONDA_BIN}" ]]; then
    for candidate in \
        "${CONDA_ROOT:-}/bin/conda" \
        "${HOME}/miniconda3/bin/conda" \
        "${HOME}/anaconda3/bin/conda"; do
        if [[ -x "${candidate}" ]]; then
            CONDA_BIN="${candidate}"
            break
        fi
    done
fi

if [[ -z "${CONDA_BIN}" ]]; then
    echo "[ERROR] Conda was not found. Install Miniconda or set CONDA_ROOT." >&2
    exit 2
fi

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    "${CONDA_BIN}" env update --name "${ENV_NAME}" --file "${ENV_FILE}"
else
    "${CONDA_BIN}" env create --file "${ENV_FILE}" --name "${ENV_NAME}"
fi

echo
echo "Conda environment ready: ${ENV_NAME}"
echo "Next: source ${REPO_ROOT}/scripts/activate_dexgym.sh"
