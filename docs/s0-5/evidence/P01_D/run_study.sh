#!/usr/bin/env bash
set -eu
root=$1
old_a=$2
old_b=$3
old_review=$4
original_c=$5
review_c=$6
final_c=$7
attempt=$8
case "$root" in /home/ubuntu/dexgym_data/s0-5/p01/runs/P01-D-*) ;; *) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
sha256sum -c "$root/freeze.sha256"
step="$root/$attempt"
mkdir -m 700 -- "$step"
run_step() {
  local name=$1 value=0
  shift
  printf '%q ' "$@" > "$step/$name.command"
  env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0 S05_PD_PRODUCTS="$step/products" S05_PD_A1="$old_a" \
    bash --noprofile --norc -c '
      source /opt/ros/jazzy/setup.bash &&
      source "$HOME/teleop_ws/install/local_setup.bash" &&
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
      export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
      exec "$PWD/.venv/bin/python" -P "$@"
    ' p01d "$@" > "$step/$name.stdout" 2> "$step/$name.stderr" || value=$?
  printf '%s\n' "$value" > "$step/$name.exit"
  printf '%s process exit=%s\n' "$name" "$value"
  return "$value"
}
run_step unit -m unittest discover -s tests/s0_5/unit -p 'test_p01*.py' -v
run_step study -m dexgym.tasks.direct.wholebody_s05.data.p01d_freeze study \
  --plan docs/s0-5/evidence/P01_D/sample_plan.json --old-a "$old_a" --old-b "$old_b" \
  --old-review "$old_review" --resource "$(dirname "$original_c")/resource_preflight.json" \
  --original-c "$original_c" --review-c "$review_c" --final-c "$final_c" --output "$step/products"
run_step integration -m unittest discover -s tests/s0_5/integration -p 'test_p01d_real_hmb.py' -v
