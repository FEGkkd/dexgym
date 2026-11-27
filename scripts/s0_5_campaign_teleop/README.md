# Read-only Teleop MCAP preflight

Run on healthy storage with a Python environment containing `mcap` and its recording compression codecs.

```sh
python -B scripts/s0_5_campaign_teleop/preflight.py \
  --bag /home/ubuntu/teleop_ws/data/s0_sim_20261119_172809/s0_sim_20261119_172809_0.mcap \
  --bag /home/ubuntu/teleop_ws/data/s0_sim_20261117_181022/s0_sim_20261117_181022_0.mcap \
  --out /path/to/new-teleop-inventory.json
```

The report binds both recordings to the frozen SHA-256 values. The script inventories observed channels, embedded schemas and metadata, and MCAP log and publish stamps. Embedded payload interpretation, clock qualification, START and initial state, lineage, and the replay allowlist remain independent D1 work. Robot commands and historical feedback are audit material.

Run synthetic controls with `python -B -m unittest discover -s scripts/s0_5_campaign_teleop -v`.

## Nominal replay of the signed HMB sources

Use the existing `teleop_ws` MuJoCo bringup and a dedicated unused localhost ROS domain. Bind the native recordings from the frozen registry with `hmb_cohort.py --registry ... --verify-assets --out ...`. The runner accepts the resulting cohort and one root. It verifies input MCAP and embedded schema identity and publishes only the two MANUS and two wrist topics. Adjacent SampleObservation records determine timing and writer identity. The operator sequence starts from measured simulation Home, with an explicitly synthetic START before the original input interval. No hardware nodes, Policy or recorded robot outputs participate.

`run_nominal_cohort.py COHORT OUTPUT --domains 195 196 197 198` runs multiple roots with fresh actual simulation instances. Supply unused domains approved for the workspace and source ROS Jazzy and the checked existing Teleop install. The ROS Python environment also needs `mcap`, `mcap-ros2-support` and the applicable compression libraries. Each root retains raw trace, input identity, replay and launch logs, and exit status. An exit status records whether replay finished. Phase and tracking evidence must be read from actual telemetry.

Use `summarize_nominal_replay.py TRACE --out SUMMARY` for observed native commands, decisions and measured feedback in ACTIVE/TASK. `render_nominal_feedback.py TRACE LOADED_MJCF --output STEM` uses the named, same-stamp MuJoCo feedback to produce video and an issued-command CSV. The renderer uses the SHA-verified loaded MJCF, headless MuJoCo graphics and FFmpeg. The recording data stay on the data host.

Use `pair_nominal_hmb.py` to pair the existing pipeline with DCR-008 on the frozen HMB cutoffs. It preserves the full planned denominator and separates runtime measured joint positions from offline nominal commands. The existing controller holds one START per root. The offline candidate uses its own generation resets, mapped targets and additional admission checks. `check_nominal_fk.py` compares the two MuJoCo geometries at the same named joint values before interpreting Cartesian differences.

See the [native nominal replay report](../../docs/s0-5/reports/TELEOP_native_nominal_replay.md) for the actual simulator/runtime evidence and result scope.

## Shared rigid START reference candidate

Prepare the previously verified fixed transforms with `prepare_shared_rigid_frames.py`. The transform receipt binds the historical calibration, native Marvin FK bridge and loaded MuJoCo XML. Apply `prepare_shared_rigid_overlay.py PACKAGE_COPY --cohort-domains` to an isolated copy of the original controller source. Source this overlay after the existing simulation install. Set `S05_SHARED_RIGID_MAP` to the verified transform file only for the simulation replay. The cohort build accepts localhost ROS domains 206–213. It applies the same world rotation to both wrists and aligns their midpoint at START, while retaining the native per-side fixed frame conversions. The original install supplies the bringup and MuJoCo driver.

Pair each successful trace with `pair_nominal_hmb.py`. Then run `export_shared_rigid_candidate.py --cohort COHORT --root CAPTURE_ID --trace TRACE --pair PAIRED_DIRECTORY --out NEW_ROOT_DIRECTORY`. The candidate preserves one row per original planned tick. The three observed joint-command payloads are represented by named issued angles and a separate mask per command group. Actual MuJoCo angles come from coherent four-topic feedback. An unavailable quantity is encoded with a mask or a status and `NaN` as appropriate. The per-root manifest gives the order, shape, dtype and file hash. Backward differences restart at signed partitions and observed lifecycle boundaries.

Use `seal_shared_rigid_candidate.py --cohort COHORT --candidate CANDIDATE_DIRECTORY --out MANIFEST` to verify every root and create the full index. Use `summarize_shared_rigid_geometry.py` to pair native A and B on the unchanged signed 50 Hz grid and report wrist scores and live relative geometry. Candidate commands are publication observations. The reported arm tracking diagnostics and available 54-axis payloads have no authority to qualify the V1-A active gate. The historical HMB and frozen M1 targets remain the scoring sources.
