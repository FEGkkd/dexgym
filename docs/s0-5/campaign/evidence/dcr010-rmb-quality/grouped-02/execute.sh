#!/usr/bin/env bash
set -u
umask 077
W=$1
P=$2
C=$3
PH=$4
G=$5
L=$6
K2=$7
DR="$C/runs/HOME-ANCHORED-DIAGNOSTIC-DCR009-01"
FR="$C/runs/TELEOP-SHARED-RIGID-START-HANDOFF-FULL-01"
MODEL="$W/source/dexgym/assets/marvin_wuji/hand2_2/generated/dual_arm_marvin.xml"
MODEL_RECEIPT="$W/docs/s0-5/campaign/evidence/dcr009-stage-two/model_frame_receipt.json"
BRIDGE=/dev/shm/s05-resume-490770e/teleop-native-02/obj/libs05_teleop.so
CONFIG="$PH/third_party/marvin_kinematics/config/ccs_m6_40.MvKDCfg"
WUJI="$PH/third_party/wuji_c_api/lib/libwuji_sdk_c.so"
finish() {
    local result=$?
    printf '%s\n' "$result" > "$L/exit.status"
    date --iso-8601=seconds > "$L/ended_at"
}
trap finish EXIT
date --iso-8601=seconds > "$L/started_at"
printf '%s\n' "$$" > "$L/runner.pid"
cd "$W" || exit
test "$(git rev-parse HEAD)" = "$K2" && test -z "$(git status --porcelain)" && test ! -e "$G" || exit 86
git rev-parse HEAD > "$L/source.commit"
git -C "$PH" rev-parse HEAD > "$L/native.commit"
sha256sum "$MODEL" "$MODEL_RECEIPT" "$BRIDGE" "$CONFIG" "$WUJI" "$DR/plan/pilot_plan.json" \
    "$DR/plan/diagnostic_runtime_frames.txt" "$FR/cohort.json" "$L/execute.sh" > "$L/input.sha256" || exit
ldd "$BRIDGE" > "$L/native.ldd" || exit
args=(env "PYTHONPATH=$W/source:$W/scripts/s0_5_campaign_teleop" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
      timeout 1500 "$P" -Bu "$W/scripts/s0_5_campaign_teleop/build_offline_rmb_diagnostic.py"
      --plan "$DR/plan/pilot_plan.json" --cohort "$FR/cohort.json" --model "$MODEL" --model-receipt "$MODEL_RECEIPT"
      --frames "$DR/plan/diagnostic_runtime_frames.txt" --bridge "$BRIDGE" --config "$CONFIG" --phase1 "$PH" --wuji "$WUJI"
      --out "$G" --root capture:b63bc72d2e8b2ee395e966f6 --group-admission --hold-proof-escalation)
printf '%q ' "${args[@]}" > "$L/command.txt"
printf '\n' >> "$L/command.txt"
"${args[@]}"
