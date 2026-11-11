#!/usr/bin/env bash
# Process-local environment; uses the host runtime overlays.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Cached binary dependencies are shared with the validated training runtime.
# Their historical directory name does not select a task or training algorithm.
runtime=${DEXGYM_RUNTIME_OVERLAY:-/home/ubuntu/dexgym/artifacts/hand_s0/runtime_v1}
export PYTHONPATH="$runtime/dexgym-numpy-fresh:$runtime/dexgym-isaaclab-fresh:$runtime/dexgym-scipy-fresh:$runtime/dexgym-omniclient-fresh:$runtime/dexgym-usd-overlay:/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:$repo/source"
cd "$repo"
exec /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/runtime.py "$@"
