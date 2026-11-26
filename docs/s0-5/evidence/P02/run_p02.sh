#!/usr/bin/env bash
set -eu
mode=$1
seq=$2
case "$seq" in ""|*[!0-9]*) exit 2;; esac
cd /home/ubuntu/dexgym_worktrees/s0-5
run=/home/ubuntu/dexgym_data/s0-5/p02/runs/P02-20261126-54c2e06
evidence=docs/s0-5/evidence/P02
sha256sum -c "$run/frozen.sha256"
out="$run/$mode-$seq"
mkdir -m 700 -- "$out"
case "$mode" in
  model) cmd=("$evidence/audit_model.py" "$evidence/plan.json" "$out/model_report.json");;
  intent) cmd=("$evidence/run_study.py" "$evidence/plan.json" "$out" "$run/model-$3/model_report.json");;
  history) cmd=("$evidence/run_history.py" "$evidence/plan.json" "$out" "$run/model-$3/model_report.json");;
  tests) cmd=(-m unittest discover -s tests/s0_5/unit -p 'test_p02*.py' -v);;
  *) exit 2;;
esac
git rev-parse HEAD > "$out/git_head"
date -Is > "$out/started_at"
printf '%q ' "${cmd[@]}" > "$out/command"
rc=0
runtime=(env -i HOME=/home/ubuntu PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
         PYTHONNOUSERSITE=1 "PYTHONPATH=$PWD/source" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1)
/usr/bin/time -v -o "$out/resources.txt" "${runtime[@]}" "$PWD/.venv/bin/python" -P "${cmd[@]}" > "$out/stdout" 2> "$out/stderr" || rc=$?
printf '%s\n' "$rc" > "$out/exit"
date -Is > "$out/ended_at"
exit "$rc"
