# Pinned Teleop source audit and D1 entry route

## Scope and source identity

Reviewed Teleop Git object `6d0b0b5b2601618f1b0052d93ef2403eea9b064a` through the GitHub repository. Local Windows checkout `30cf13bdc496c43b3016462cfa0a206334d1ff7f` was a comparison only. Before D1, pin the actual Ubuntu worktree commit, executable/library/model hashes, environment, selected YAML contents, and loaded ROS parameters. This work package supplies offline checks. It confers no D1 approval.

## Expected graph and admission

| Pinned source | Verified behavior and acceptance implications |
| --- | --- |
| [Composition launch](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_bringup/launch/sim.launch.py) and [simulation composition YAML](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_bringup/config/sim.yaml) | Simulation entry delegates to the generic composition in `real.launch.py`. The selected simulation YAML enables exactly the MuJoCo driver and Teleop coordinator, with Tracker and MANUS hardware input nodes disabled. Inspect resolved effective composition. |
| [Coordinator launch](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/launch/whole_body_teleop.launch.py) and [simulation overlay](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/config/sim.yaml) | Explicitly select and hash the approved simulation overlay. Verify effective `workflow.driver_prefix=/tianji_mujoco_driver`, `arm.driver_standby_service=/tianji_mujoco_driver/standby`, `head.enabled=false`, `enable_hands=true`, `arm.kinematics_backend=marvin` and `arm.allow_identity_backend=false`. A bare coordinator launch uses real-oriented defaults. |
| [Arm controller](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/teleop_controller_node.cpp#L2350) and [Marvin adapter](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/kinematics_adapter.cpp) | The arm subscriber uses `/vive/tracker/{left,right}_wrist` with `tianji_interfaces/msg/TrackerPose`. The controller processes live tracker targets, then uses feasible handoff, command shaping and publication guards. Report registration mode, latched START reference, online nominal positions, independent modifier and the final accepted actuator proposal as different trace fields. |
| [Coordinator construction](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/whole_body_teleop_node.cpp) and [hand retarget node](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_teleop/src/retarget_node.cpp) | The two `/manus_glove_{left,right}` subscribers use `manus_ros2_msgs/msg/ManusGlove`. The coordinator owns both hand retarget nodes and their command-generation guard. Hand freshness tests use the ROS header, so independently review the transport epoch alignment in addition to the MCAP file order. |
| [Wuji session](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/wuji_sdk_cpp/src/retarget_session.cpp) and [MuJoCo driver](https://github.com/FEGkkd/whole-body-teleop/blob/6d0b0b5b2601618f1b0052d93ef2403eea9b064a/src/whole_body_mujoco_driver/src/mujoco_driver_node.cpp#L40) | Wuji retargets 63 keypoint floats per hand in a live session and validates finite output. MuJoCo receives `/tianji/joint_command` and `/{left,right}_hand/command`. Its measured robot/contact/arm/hand feedback is authoritative for the run. Record configured safety deadlines, command acceptance and actual applied trajectories separately. |

T0 uses the pinned original controller with all learned residuals disabled. T1/T2 use the same coordinator code and configuration, with the sole treatment variable being the bounded Actor modifier. The integration seam is inside the coordinator after online nominal candidate generation and before the common safety and commit path. T1 is a bypass or exact zero residual at that seam. Reuse `commit_hand_command` for hand publishing and retain the arm's generation and freshness checks in `publish_command`. Use equivalent staged snapshots to coordinate the two existing chains. The residual process never owns a joint or hand command publisher. The default arm simulation control loop is 250 Hz and the learner uses a 50 Hz input/action contract. A deterministic adapter for the resulting 5-to-1 rate relation, qualified observations and one arm/hand generation per residual sample need explicit T1/T2 parity and timeout evidence. Old cached RMB and logged robot commands/feedback are audit inputs only.

Use a fresh ROS domain dedicated to each run and a preflight environment with localhost-only discovery, no static peers, no bridge or real device clients. Verify the observed graph before any replay or explicit workflow START. Capture `ros2 node list`, `ros2 topic info -v` for the three command streams and robot feedback topics, and the effective Teleop/driver parameters. Require exactly one coordinator writer per command stream and the simulator producer on feedback. Discovering an additional publisher, non-sim driver, unexpected workflow service or unresolved clock ownership blocks activation. Preserve original START/operator events as evidence and use reviewed workflow operations to reproduce verified events. Diagnostic or lifecycle status messages recorded in a bag have no replay eligibility. Unverified START, initial state or registration bounds the outcome to offline diagnostics.

## Reproducible offline checks on a healthy Ubuntu mount

Use the already frozen inventory command in the [MCAP preflight README](README.md) once for the two original recordings. The next step obtains the immutable human-input timing witness for each bag separately. It needs a review JSON externally bound to the evidence SHA. Required keys:

- `schema` equal to `s05.teleop_timing_witness.review.v1`
- `teleop_revision` equal to the pinned Git identity above
- `bag_sha256` equal to its frozen MCAP identity
- `clock` equal to `REVIEWED_RECORDED_LOG_TIME` and `clock_evidence_sha256` pinned to the independent epoch and input-availability analysis
- `tracker_basis` set to `RAW_VIVE` or `REGISTERED_INPUT` based on separate calibration evidence
- `channels` with exactly the four human topics, each binding its MCAP `channel_id`, exact ROS `schema_name`, `schema_encoding`, and `schema_sha256` from preflight

After those checks, run:

```sh
python -B scripts/s0_5_campaign_teleop/input_timing.py \
  --bag /home/ubuntu/teleop_ws/data/s0_sim_20261119_172809/s0_sim_20261119_172809_0.mcap \
  --review /path/qualified-recording-clock-review.json \
  --review-sha256 "$REVIEW_SHA" \
  --out /path/new-input-timing-witness.json
```

Repeat with the 181022 bag and its own review identity. The witness uses log-time deadlines relative to the first selected human record and original MCAP message index to break ties. It rejects selected clock reversal, unexpected duplicate channels, schema drift and absent required input. It emits counts and a canonical schedule digest. It does not open ROS, write a replay bag, infer START or produce a qualification. Only after G-D0, observed clean runtime and signed-asset checks and an authorized isolated graph can an actual runner consume the reviewed stream plus independently verified lifecycle events.
