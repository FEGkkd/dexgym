#!/usr/bin/env bash
# Scoped P01-C run. No production, hardware device or policy launch paths.
set -eu
run=$1
old_a=$2
old_b=$3
old_review=$4
attempt=$5
case "$run" in /home/ubuntu/dexgym_data/s0-5/p01/runs/P01-C-*) ;; *) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
step="$run/$attempt"
test -f "$run/freeze.sha256"
sha256sum -c "$run/freeze.sha256"
mkdir -m 700 -- "$step"
clean_python() {
  env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 MPLCONFIGDIR="$step/matplotlib" \
    S05_PC_PRODUCTS="$step/products" S05_PC_OLD_A="$old_a" \
    bash --noprofile --norc -c '
      source /opt/ros/jazzy/setup.bash &&
      source "$HOME/teleop_ws/install/local_setup.bash" &&
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
      export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
      exec "$PWD/.venv/bin/python" -P "$@"
    ' p01-c "$@"
}
run_step() {
  local name=$1
  shift
  local value=0
  printf '%q ' "$@" > "$step/$name.command"
  clean_python "$@" > "$step/$name.stdout" 2> "$step/$name.stderr" || value=$?
  printf '%s\n' "$value" > "$step/$name.exit"
  printf 'P01-C %s process exit=%s\n' "$name" "$value"
  return "$value"
}
run_step unit -m unittest discover -s tests/s0_5/unit -p 'test_p01*.py' -v
tail -6 "$step/unit.stderr"
run_step audit -m dexgym.tasks.direct.wholebody_s05.data.sync_audit \
  --plan docs/s0-5/evidence/P01_C/sample_plan.json --old-a "$old_a" --old-b "$old_b" \
  --old-review "$old_review" --resource "$run/resource_preflight.json" --output "$step/products"
cat "$step/audit.stdout"
run_step integration -m unittest discover -s tests/s0_5/integration -p 'test_p01c_real_sync.py' -v
tail -10 "$step/integration.stderr"
