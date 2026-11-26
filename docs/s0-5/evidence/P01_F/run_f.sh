#!/usr/bin/env bash
set -eu
task=$1
number=$2
case "$number" in ""|*[!0-9]*) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
base=/home/ubuntu/dexgym_data/s0-5/p01/runs
root="$base/P01-F-20261126-b22f8be"
archive="$base/P01-E-E1-20261126-3f90652"
old_a="$base/P01-A-20261126-09c1c9e-r03/registry"
tools=docs/s0-5/evidence/P01_F
test -s "$root/frozen.txt"
tail -n +3 "$root/frozen.txt" | sha256sum -c -
out="$root/$task-$number"
mkdir -m 700 -- "$out"
case "$task" in
  archive) args=("$tools/scan_archive.py" "$archive" "$old_a" "$out");;
  raw) args=("$tools/reconstruct.py" "$archive" "$old_a" "$root/archive-$3" "$out");;
  tests) args=(-m unittest discover -s tests/s0_5/unit -p 'test_p01*.py' -v);;
  *) exit 2 ;;
esac
git rev-parse HEAD > "$out/git_head.txt"
printf '%q ' "${args[@]}" > "$out/run.command"
status=0
runtime=(env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 bash --noprofile --norc)
/usr/bin/time -v -o "$out/run.time" "${runtime[@]}" -c '
    source /opt/ros/jazzy/setup.bash &&
    source "$HOME/teleop_ws/install/local_setup.bash" &&
    source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
    export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
    exec "$PWD/.venv/bin/python" -P "$@"
  ' p01f "${args[@]}" > "$out/run.stdout" 2> "$out/run.stderr" || status=$?
printf '%s\n' "$status" > "$out/run.exit"
exit "$status"
