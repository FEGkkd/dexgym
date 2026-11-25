#!/usr/bin/env bash
# P01-B only. Unit regressions precede catalogue and real decode.
set -eu
run=$1
old=$2
attempt=$3
case "$run" in /home/ubuntu/dexgym_data/s0-5/p01/runs/P01-B-*) ;; *) exit 2 ;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
clean_python() {
  env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 S05_P01B_PRODUCTS="$run/$attempt/products" \
    bash --noprofile --norc -c '
      source /opt/ros/jazzy/setup.bash &&
      source "$HOME/teleop_ws/install/local_setup.bash" &&
      source "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash" &&
      export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}" &&
      exec "$PWD/.venv/bin/python" -P "$@"
    ' p01-b "$@"
}
dir="$run/$attempt"
mkdir -m 700 -- "$dir"
unit=0
clean_python -m unittest discover -s tests/s0_5/unit -p 'test_p01*.py' -v > "$dir/unit.log" 2>&1 || unit=$?
/usr/bin/python3 - "$dir" "$unit" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
(root/'unit.json').write_text(json.dumps({'exit':int(sys.argv[2]),'log':str(root/'unit.log'),
'stdout_stderr':(root/'unit.log').read_text()},sort_keys=True)+'\n')
PY
printf 'unit exit=%s\n' "$unit"
tail -5 "$dir/unit.log"
test "$unit" -eq 0
python3 -B docs/s0-5/evidence/P01_B/capture_timestamp_witness.py "$old/timing_source.json" > "$dir/timing_source.json"
cmd=( -m dexgym.tasks.direct.wholebody_s05.data.decode_audit
  --config docs/s0-5/evidence/P01_B/sample_plan.json
  --a-config docs/s0-5/evidence/P01_A/host_registry_config.json
  --a-registry "$old/registry" --a-receipt docs/s0-5/evidence/P01_A/review_receipt.json
  --resource "$run/resource_preflight.json" --unit "$dir/unit.json"
  --timing "$dir/timing_source.json" --output "$dir/products" )
printf '%q ' "${cmd[@]}" > "$dir/decode.command"
printf '\n' >> "$dir/decode.command"
code=0
clean_python "${cmd[@]}" > "$dir/decode.stdout" 2> "$dir/decode.stderr" || code=$?
printf '%s\n' "$code" > "$dir/decode.exit"
printf 'decode process exit=%s\n' "$code"
cat "$dir/decode.stdout"
test "$code" -eq 0
integration=0
clean_python -m unittest discover -s tests/s0_5/integration -p test_p01b_real_decode.py -v > "$dir/integration.log" 2>&1 || integration=$?
printf 'real integration exit=%s\n' "$integration"
tail -18 "$dir/integration.log"
/usr/bin/python3 - "$dir" "$integration" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
result={'unit':json.loads((root/'unit.json').read_bytes()),'decode_exit':int((root/'decode.exit').read_text()),
        'integration':{'exit':int(sys.argv[2]),'log':str(root/'integration.log'),
                       'stdout_stderr':(root/'integration.log').read_text()}}
(root/'test_results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY
exit "$integration"
