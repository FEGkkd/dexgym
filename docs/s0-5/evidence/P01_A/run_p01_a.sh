#!/usr/bin/env bash
# Stage A only: run unit tests before read-only source inventory, then verify real outputs.
set -u
repo=/home/ubuntu/dexgym_worktrees/s0-5
run=$1
case "$run" in /home/ubuntu/dexgym_data/s0-5/p01/runs/P01-A-*) ;; *) exit 2 ;; esac
cd "$repo" || exit 1
clean_python() {
  env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 S05_P01A_REGISTRY="$run/registry" \
    bash --noprofile --norc -c '
      source /opt/ros/jazzy/setup.bash &&
      source "$HOME/teleop_ws/install/local_setup.bash" &&
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
      export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
      exec "$PWD/.venv/bin/python" -P "$@"
    ' p01-a "$@"
}
unit=( -m unittest discover -s tests/s0_5/unit -p test_p01a_registry.py -v )
integration=( -m unittest discover -s tests/s0_5/integration -p test_p01a_real_registry.py -v )
session="$run/attempt-$2"
mkdir -m 700 -- "$session" || exit 1
clean_python "${unit[@]}" > "$session/unit.txt" 2>&1
unit_exit=$?
printf 'P01-A unit exit=%s\n' "$unit_exit"
cat "$session/unit.txt"
/usr/bin/python3 - "$session" "$unit_exit" <<'PY_TEST'
import json,sys
from pathlib import Path
base=Path(sys.argv[1])
obj={'exit':int(sys.argv[2]),'suite':'P01-A registry unit','log':str(base/'unit.txt'),
     'stdout_stderr':(base/'unit.txt').read_text()}
(base/'unit_result.json').write_text(json.dumps(obj,sort_keys=True)+'\n')
PY_TEST
if [ "$unit_exit" -ne 0 ]; then exit "$unit_exit"; fi
cmd=( -m dexgym.tasks.direct.wholebody_s05.data.source_registry
      --config docs/s0-5/evidence/P01_A/host_registry_config.json
      --output "$run/registry" --resource "$run/resource_preflight.json"
      --test-results "$session/unit_result.json" )
printf '%q ' "${cmd[@]}" > "$session/registry-command.txt"
printf '\n' >> "$session/registry-command.txt"
clean_python "${cmd[@]}" > "$session/registry.stdout" 2> "$session/registry.stderr"
registry_exit=$?
printf '%s\n' "$registry_exit" > "$session/registry.exit"
printf 'P01-A registry exit=%s\n' "$registry_exit"
cat "$session/registry.stderr" "$session/registry.stdout"
if [ "$registry_exit" -ne 0 ]; then exit "$registry_exit"; fi
clean_python "${integration[@]}" > "$session/integration.txt" 2>&1
integration_exit=$?
cat "$session/integration.txt"
printf 'P01-A real integration exit=%s\n' "$integration_exit"
/usr/bin/python3 - "$session" "$integration_exit" "$run/test_results.json" <<'PY_INTEGRATION'
import json,sys
from pathlib import Path
base=Path(sys.argv[1])
doc={'unit_before_inventory':json.loads((base/'unit_result.json').read_bytes()),
     'registry_exit':int((base/'registry.exit').read_text()),
     'real_integration':{'exit':int(sys.argv[2]),'stdout_stderr':(base/'integration.txt').read_text(),
                         'log':str(base/'integration.txt')}}
Path(sys.argv[3]).write_text(json.dumps(doc,sort_keys=True,indent=2)+'\n')
PY_INTEGRATION
exit "$integration_exit"
