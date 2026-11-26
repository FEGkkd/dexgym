#!/usr/bin/env bash
set -eu
mode=$1
attempt=$2
case "$attempt" in ""|*[!0-9]*) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
base=/home/ubuntu/dexgym_data/s0-5/p01/runs
run="$base/P01-E-E1-20261126-3f90652"
evidence=docs/s0-5/evidence
old_a="$base/P01-A-20261126-09c1c9e-r03/registry"
out="$run/$mode-$attempt"
mkdir -m 700 -- "$out"
args=()
case "$mode" in
  smoke) args=(--smoke "$run");;
  review)
    test -f "$run/products/build_index.json"
    args=("$run" "$out")
    ;;
  *) exit 2;;
esac
cmd=("$evidence/P01_E_E1/verify_e1.py" "${args[@]}" "$old_a"
     "$evidence/P01_E_E0/candidate_config.json" "$evidence/P01_E_E0/execution_lock.json")
printf '%q ' "${cmd[@]}" > "$out/review.command"
status=0
runtime=(env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0 bash --noprofile --norc)
/usr/bin/time -v -o "$out/review.time" "${runtime[@]}" -c '
    source /opt/ros/jazzy/setup.bash &&
    source "$HOME/teleop_ws/install/local_setup.bash" &&
    source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
    export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
    exec "$PWD/.venv/bin/python" -P "$@"
  ' p01e "${cmd[@]}" > "$out/review.stdout" 2> "$out/review.stderr" || status=$?
printf '%s\n' "$status" > "$out/review.exit"
exit "$status"
