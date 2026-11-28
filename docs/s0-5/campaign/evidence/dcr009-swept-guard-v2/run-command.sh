#!/usr/bin/env bash
set -e
repo=/home/ubuntu/dexgym_worktrees/s05-dcr009-diagnostic
campaign=/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e
runs="$campaign/runs"
diagnostic="$runs/HOME-ANCHORED-DIAGNOSTIC-DCR009-01"
history="$runs/HOME-ANCHORED-DIAGNOSTIC-DCR009-ROOT05-YAW-STAGED-HISTORY-01"
native="$diagnostic/native/install-supported"
full="$runs/TELEOP-SHARED-RIGID-START-HANDOFF-FULL-01"
legacy="$runs/TELEOP-AB-SHARED-RIGID-03-01/shared-frames/shared_rigid_transforms.txt"
plan="$diagnostic/plan/pilot_plan.json"
controller="$history/native/install/whole_body_teleop/lib/whole_body_teleop/whole_body_teleop_node"
driver="$native/whole_body_mujoco_driver/lib/whole_body_mujoco_driver/whole_body_mujoco_driver_node"
model="$native/whole_body_mujoco_driver/share/whole_body_mujoco_driver/model/dual_arm_marvin.xml"
signed="$repo/source/dexgym/assets/marvin_wuji/hand2_2/generated/dual_arm_marvin.xml"
models="$repo/docs/s0-5/campaign/evidence/dcr009-stage-two/model_frame_receipt.json"
axes="$repo/docs/s0-5/campaign/evidence/dcr009-stage-two/virtual_axis_receipt.json"
review="$repo/docs/s0-5/campaign/evidence/dcr009-swept-guard-v2/independent-technical-review.json"
experiment="$runs/HOME-ANCHORED-DIAGNOSTIC-DCR009-ROOT05-SWEPT-GUARD-V2-01"
cd "$repo"
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/teleop_ws/install/setup.bash
source "$native/local_setup.bash"
source "$history/native/install/local_setup.bash"
unset ROS_DISCOVERY_SERVER ROS_STATIC_PEERS CYCLONEDDS_URI FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE S05_DCR009_SEGMENTED_START_STATE S05_DCR009_FINGER_STAGE S05_DCR009_HISTORY_AUDIT
export PYTHONPATH="$repo/source:$repo/scripts/s0_5_campaign_teleop:$diagnostic/preflight/python_imports:$runs/HOME-ANCHORED-PILOT-DCR009-01/read_only_preflight/python_imports:$PYTHONPATH"
plan_sha=3c0268291bdb8916a9a520d3fa73a1340fffd7f9e8a7b7364a1ef9950f9a0004
controller_sha=9a4a318b11984a1659284cb3c38e8f382ff3de8539f14f495f6ca335e8016376
driver_sha=b226d4cbf4bbe7897a93c212c2e494430cbfe62da2aa97661c51e1fdd27fce59
printf '%s  %s\n' "$plan_sha" "$plan" "$controller_sha" "$controller" "$driver_sha" "$driver" | sha256sum -c
if [ "${1-}" = "--check-only" ]; then
    /usr/bin/python3 -B -c 'import mujoco, rclpy, gymnasium, mcap_ros2, run_dcr009_guarded_diagnostic, swept_live_guard_v2; print(mujoco.__version__)'
    ros2 pkg prefix whole_body_teleop
    ros2 pkg prefix whole_body_mujoco_driver
    exit 0
fi
/usr/bin/python3 -B -c 'import sys; from pathlib import Path; from run_dcr009_guarded_diagnostic import verify_sweep_v2_review; verify_sweep_v2_review(Path(sys.argv[1]))' "$review"
mkdir "$experiment"
install -m 0644 "$0" "$experiment/run-command.sh"
printf '%s\n' "$$" > "$experiment/launcher.pid"
set +e
timeout 210 /usr/bin/python3 -B scripts/s0_5_campaign_teleop/run_dcr009_guarded_diagnostic.py \
    --plan "$plan" --plan-sha256 "$plan_sha" --cohort "$full/cohort.json" --virtual-axes "$axes" \
    --legacy-frames "$legacy" --frames "$diagnostic/plan/diagnostic_runtime_frames.txt" \
    --home-reference "$full/home_reference.json" --model "$model" --signed-model "$signed" --model-receipt "$models" \
    --controller-sha256 "$controller_sha" --simulator-sha256 "$driver_sha" --python /usr/bin/python3 \
    --domain 213 --root-five-start-only segmented --align-start-yaw-to-home --wrist-first-fingers \
    --replay-root-five-history --sweep-v2-review "$review" --output "$experiment/run" > "$experiment/run.log" 2>&1
status=$?
printf '%s\n' "$status" > "$experiment/launcher.exit"
exit "$status"
