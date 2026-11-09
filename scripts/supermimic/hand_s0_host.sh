#!/usr/bin/env bash
# Process-local environment; never changes the shared venv or the PPO job.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="/tmp/dexgym-numpy-fresh:/tmp/dexgym-isaaclab-fresh:/tmp/dexgym-scipy-fresh:/tmp/dexgym-omniclient-fresh:/tmp/dexgym-usd-overlay:/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:$repo/source"
cd "$repo"
exec /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/hand_s0_runtime.py "$@"
