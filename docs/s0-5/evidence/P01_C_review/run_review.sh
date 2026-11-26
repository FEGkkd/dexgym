#!/usr/bin/env bash
set -eu
run=$1
old_c=$2
old_a=$3
old_b=$4
old_review=$5
attempt=$6
case "$run" in /home/ubuntu/dexgym_data/s0-5/p01/runs/P01-C-REVIEW-*) ;; *) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
step="$run/$attempt"
sha256sum -c "$run/freeze.sha256"
mkdir -m 700 -- "$step"
run_step() {
  local name=$1 value=0
  shift
  printf '%q ' "$@" > "$step/$name.command"
  env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1     PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0 S05_CR_PRODUCTS="$step/products"     bash --noprofile --norc -c '
      source /opt/ros/jazzy/setup.bash &&
      source "$HOME/teleop_ws/install/local_setup.bash" &&
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
      export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
      exec "$PWD/.venv/bin/python" -P "$@"
    ' review "$@" > "$step/$name.stdout" 2> "$step/$name.stderr" || value=$?
  printf '%s\n' "$value" > "$step/$name.exit"
  printf '%s process exit=%s\n' "$name" "$value"
  return "$value"
}
run_step unit -m unittest discover -s tests/s0_5/unit -p 'test_p01*.py' -v
run_step legacy_order docs/s0-5/evidence/P01_C_review/legacy_order_witness.py
run_step paired -m dexgym.tasks.direct.wholebody_s05.data.review_audit \
  --plan docs/s0-5/evidence/P01_C/sample_plan.json --old-a "$old_a" --old-b "$old_b" \
  --old-review "$old_review" --resource "$(dirname "$old_c")/resource_preflight.json" \
  --legacy "$old_c/products" --output "$step/products"
run_step integration -m unittest discover -s tests/s0_5/integration -p 'test_p01c_review_integration.py' -v
