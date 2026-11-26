#!/usr/bin/env bash
set -eu
session=$1
case "$session" in ""|*[!0-9]*) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
base=/home/ubuntu/dexgym_data/s0-5/p01/runs
run="$base/P01-E-E1-20261126-3f90652"
old_a="$base/P01-A-20261126-09c1c9e-r03/registry"
evidence=docs/s0-5/evidence
config="$evidence/P01_E_E0/candidate_config.json"
lock="$evidence/P01_E_E0/execution_lock.json"
step="$run/session-$session"
umask 077
mkdir -p -- "$run"
mkdir -- "$step"
date --iso-8601=seconds > "$step/started_at.txt"
git rev-parse HEAD > "$step/git_head.txt"
git status --porcelain=v1 > "$step/worktree.txt"
test ! -s "$step/worktree.txt"
python3 -B "$evidence/P00/check_documents.py" --require-history > "$step/documents.stdout" 2> "$step/documents.stderr"
execute() {
  local name=$1 status=0
  shift
  printf '%q ' "$@" > "$step/$name.command"
  /usr/bin/time -v -o "$step/$name.time" +    env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GIT_OPTIONAL_LOCKS=0 +    bash --noprofile --norc -c '
      source /opt/ros/jazzy/setup.bash &&
      source "$HOME/teleop_ws/install/local_setup.bash" &&
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
      export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
      exec "$PWD/.venv/bin/python" -P "$@"
    ' p01e "$@" > "$step/$name.stdout" 2> "$step/$name.stderr" || status=$?
  printf '%s\n' "$status" > "$step/$name.exit"
  printf '%s process exit=%s\n' "$name" "$status"
  return "$status"
}
execute preflight "$evidence/P01_E_E1/preflight.py" "$run" "$session" "$old_a" "$config" "$lock"
if [ "$session" = 1 ]; then
  execute unit -m unittest discover -s tests/s0_5/unit -p 'test_p01*.py' -v
fi
execute build -m dexgym.tasks.direct.wholebody_s05.data.p01e_build build --old-a "$old_a" +  --config "$config" --lock "$lock" --output "$run/products" --session "$session"
date --iso-8601=seconds > "$step/finished_at.txt"
