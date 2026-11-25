#!/usr/bin/env bash
# Execute the bounded review probe in the pinned ROS and parser environment.
set -u
probe=$(realpath -- "$1") || exit 1
output=$2
mkdir -m 700 -- "$output" || exit 1
result=0
for section in environment samples historical; do
  env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 bash --noprofile --norc -c '
    source /opt/ros/jazzy/setup.bash &&
    source "$HOME/teleop_ws/install/local_setup.bash" &&
    source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
    cd "$HOME/dexgym_worktrees/s0-5" &&
    export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
    exec "$PWD/.venv/bin/python" -P "$1" "$2"
  ' preflight "$probe" "$section" > "$output/$section.json" 2> "$output/$section.stderr.txt"
  code=$?
  printf '%s\n' "$code" > "$output/$section.exit.txt"
  printf 'RUN %s exit=%s bytes=%s\n' "$section" "$code" "$(wc -c < "$output/$section.json")"
  if [ "$code" -ne 0 ]; then
    result=1
    cat "$output/$section.stderr.txt" >&2
  fi
done
exit "$result"
