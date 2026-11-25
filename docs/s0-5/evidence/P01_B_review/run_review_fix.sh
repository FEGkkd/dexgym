#!/usr/bin/env bash
# Isolated P01-B review. The historical decode environment never loads the current overlay.
set -eu
run=$1
old_a=$2
old_b=$3
case "$run" in /home/ubuntu/dexgym_data/s0-5/p01/runs/P01-B-REVIEW-*) ;; *) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
test "$(git rev-parse HEAD)" = f32d6bbfc4f535a5c53db1025a484e48fc5f2ce3 || test "$(git rev-parse HEAD^)" = f32d6bbfc4f535a5c53db1025a484e48fc5f2ce3
mkdir -m 700 -- "$run"
teleop=/home/ubuntu/teleop_ws
collector=/home/ubuntu/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection
evidence=docs/s0-5/evidence/P01_B_review
clean_python() {
  local overlay=$1
  shift
  env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    bash --noprofile --norc -c '
    source /opt/ros/jazzy/setup.bash &&
    if [ "$1" = current ]; then
      source "$HOME/teleop_ws/install/local_setup.bash" &&
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash"
    else
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/ros2_ws/install/local_setup.bash"
    fi &&
    shift &&
    export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
    exec "$PWD/.venv/bin/python" -P "$@"
  ' p01-review "$overlay" "$@"
}
record() {
  local name=$1
  shift
  printf 'clean_python ' > "$run/$name.command"
  printf '%q ' "$@" >> "$run/$name.command"
  local code=0
  clean_python "$@" > "$run/$name.stdout" 2> "$run/$name.stderr" || code=$?
  printf '%s\n' "$code" > "$run/$name.exit"
  printf '%s exit=%s\n' "$name" "$code"
  return "$code"
}
record resource_preflight current docs/s0-5/evidence/P00_review_fix/readiness_probe.py environment
cp -- "$run/resource_preflight.stdout" "$run/resource_preflight.json"
record unit current -m unittest discover -s tests/s0_5/unit -p 'test_p01*.py' -v
tail -5 "$run/unit.stderr"
record legacy historical "$evidence/legacy_collection_review.py" --original "$old_b/products" \
  --historical-prefix "$collector/ros2_ws/install/dex_collection_interfaces" \
  --current-prefix "$collector/install/dex_collection_interfaces" --output "$run/legacy" \
  --teleop-repo "$teleop" --collector-repo "$collector"
record hmb current "$evidence/run_hmb_review.py" "$old_a" "$old_b" "$run"
cat "$run/legacy.stdout" "$run/hmb.stdout"
python3 -B - "$run" <<'PY'
from hashlib import file_digest
from pathlib import Path
import json,sys

root=Path(sys.argv[1])
def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()
stable=json.loads((root/'output_sha256.json').read_bytes())
if any(sha(root/name)!=value for name,value in stable.items()):
    raise ValueError('Review output mismatch')
refs=('review_receipt.json','output_sha256.json','hmb.stdout','hmb.stderr','hmb.exit')
with (root/'process_sha256.json').open('x',encoding='utf-8') as stream:
    json.dump({name:sha(root/name) for name in refs},stream,indent=2,sort_keys=True)
    stream.write('\n')
print(f'{len(stable)} stable output hashes verified')
PY
