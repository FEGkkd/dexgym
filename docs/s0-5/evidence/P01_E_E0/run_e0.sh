#!/usr/bin/env bash
set -eu
run=$1
old_a=$2
attempt=$3
case "$run" in /home/ubuntu/dexgym_data/s0-5/p01/runs/P01-E-E0-*) ;; *) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
sha256sum -c "$run/freeze.sha256"
step="$run/$attempt"
mkdir -m 700 -- "$step"
execute() {
  local step_name=$1 status=0
  shift
  printf '%q ' "$@" > "$step/$step_name.command"
  env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0 bash --noprofile --norc -c '
      source /opt/ros/jazzy/setup.bash &&
      source "$HOME/teleop_ws/install/local_setup.bash" &&
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
      export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
      exec "$PWD/.venv/bin/python" -P "$@"
    ' p01e "$@" > "$step/$step_name.stdout" 2> "$step/$step_name.stderr" || status=$?
  printf '%s\n' "$status" > "$step/$step_name.exit"
  printf '%s process exit=%s\n' "$step_name" "$status"
  return "$status"
}
module=dexgym.tasks.direct.wholebody_s05.data.p01e_build
execute unit -m unittest discover -s tests/s0_5/unit -p 'test_p01*.py' -v
execute prepare -m "$module" prepare --old-a "$old_a" --output "$step/candidate"
options=(--old-a "$old_a" --config "$step/candidate/candidate_config.json" --lock "$step/candidate/execution_lock.json"
         --output "$step/trial_products" --fixture "$run/fixture_plan.json")
if execute interrupted -m "$module" probe "${options[@]}" --session 1 --inject-index 1; then
  echo 'Expected controlled publication failure'; exit 3
fi
test "$(cat "$step/interrupted.exit")" = 1
grep -q 'InjectedPublicationError: Injected stop after fsync before atomic publication' "$step/interrupted.stderr"
execute resume -m "$module" probe "${options[@]}" --session 2
execute idempotent -m "$module" probe "${options[@]}" --session 3
execute integration docs/s0-5/evidence/P01_E_E0/verify_e0.py "$step" "$old_a"
