# Marvin M6 + WujiHand2 Beta 2: external asset validation

**Status (2026-11-04):** the complete 54 DoF articulation loads in Isaac Lab
3.0.0/Newton MJWarp, retains named control channels, and passes the source
MJCF versus Newton forward-kinematics acceptance target. Physical position
drives are finite but have material tracking and backend response differences.
The asset is suitable for defining an offline Arm S0 task interface; controller
and physical parameter calibration are still required before claiming accurate
fixed-TCP physical execution or real-world fidelity. No S0 task, PPO run or
robot connection was made.

The machine-readable results are in [validation_summary.json](validation_summary.json).
Local CSV, full JSON logs, converted USD and source meshes remain under ignored
`artifacts/marvin_wuji/` or the external source checkout.

## Provenance and conversion

- Authoritative input: `FEGkkd/whole-body-teleop`, branch
  `codex/whole-body-sim-phase1`, commit
  `5d4c345842f197bffeae7d7e08c4c3718230f9e0`,
  `src/whole_body_mujoco_driver/model/dual_arm_marvin.xml`.
- XML SHA256:
  `6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce`.
  The present local `4sensor` checkout has the same XML hash. The importer
  refuses another hash and records the hashes of all 70 referenced meshes.
- The source model uses Marvin's verified J6/J7 directions and TCP orientation;
  these were not edited. Four keyframes are `hardware_zero`, `home`,
  `measured_j5_zero`, and `forward_pose`. The Wuji source is the external
  `wuji-description` tag `v2026.9.11` (source README cites commit
  `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`). The XML is authoritative
  over `model/tools/build_mjcf.py` where drive gains differ.
- Conversion uses official Isaac Lab 3.0.0
  `isaaclab.sim.converters.MjcfConverter`, with `fix_base=True`,
  `merge_mesh=False`, `self_collision=True`, asset transformer and multi-physics
  conversion enabled. It runs without Isaac Sim/Kit. The initial import
  accidentally left `newton:selfCollisionEnabled=False`; contact testing found
  that problem and the final USD was regenerated with the flag enabled.
  The final root `dual_arm_marvin.usda` SHA256 is
  `4643b25b3ddc1412f83ef2f089bfc4fe6dd90ad2f36ef76efd9d48112ecd1ceb`;
  all 70 source mesh and 9 generated USD/payload hashes are in the machine
  summary. The conversion script also records exact importer wheel versions.
  Some generated paths may vary with importer versions or checkout location,
  so structural and physical checks remain necessary after conversion.
- The byte-verified layered USD (one root and eight payload files) is packaged in
  `source/dexgym/assets/marvin_wuji/usd/`, so a DexGym checkout can load the
  asset directly with `make_robot_cfg()`. Four source keyframes are packaged
  separately as `source_keyframes.json`; a regenerated file has the same
  SHA256 `3539c12d23ef6084a4fdfdd1a4f06f96cf14306b9c8a1fd1278370faa84bc351`.
  The raw MJCF/URDF/STL dependencies remain in the pinned
  `whole-body-teleop` source commit; the script can
  regenerate the checked-in USD and verify all hashes from that checkout.

To load the packaged articulation, call
`dexgym.assets.marvin_wuji.robot_cfg.make_robot_cfg()`; its default path is
`source/dexgym/assets/marvin_wuji/usd/dual_arm_marvin.usda`. To reproduce the
conversion and all validation gates from the pinned source model, run from the
DexGym root with the XML and meshes present:

```bash
uv sync --locked --group dev
bash scripts/assets/setup_mjcf_importer.sh
export DEXGYM_MARVIN_MJCF=/path/to/whole-body-teleop/src/whole_body_mujoco_driver/model/dual_arm_marvin.xml
uv run --no-sync python scripts/assets/import_marvin_wuji.py
USD=artifacts/marvin_wuji/generated/6e0d569e-self-collision/dual_arm_marvin/dual_arm_marvin.usda
uv run --no-sync python scripts/assets/validate_model_structure.py --usd "$USD" --output artifacts/marvin_wuji/structure.json
uv run --no-sync python scripts/assets/validate_fk_parity.py --usd "$USD" --samples 1000 --num-envs 16 --output artifacts/marvin_wuji/fk_parity.json
uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" --mode smoke --num-envs 16 --steps 200 --resets 3 --output artifacts/marvin_wuji/smoke16.json
uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" --mode channels --num-envs 16 --steps 150 --output artifacts/marvin_wuji/channels54.json
uv run --no-sync python scripts/assets/validate_dynamics.py --usd "$USD" --mode dynamics --num-envs 1 --steps 250 --output artifacts/marvin_wuji/dynamics.json
uv run --no-sync python scripts/assets/validate_wrist_rotation.py --usd "$USD" --output artifacts/marvin_wuji/wrist_rotation.json
uv run --no-sync python scripts/assets/validate_contacts.py --usd "$USD" --output artifacts/marvin_wuji/contacts.json
uv run --no-sync python scripts/assets/view_marvin_wuji.py --usd "$USD" --pose home --viz newton_gl --steps 1000
```

The independent viewer defaults to the packaged USD and keyframes, without
requiring the source checkout. It also accepts `--viz none` for a headless GPU
load check. On the final host `--viz newton_gl --steps 5` completed with all 54
joints, 74 bodies and finite state; the visualizer was opened on `DISPLAY=:0`.
The packaged path was rechecked with 1,000 FK poses and a 16-environment,
20-step GPU smoke; the FK maxima stayed at 0.000950 mm and 0.0000367°. A
clean wheel build contained exactly nine USD layers and excluded the local
raw-model directory. Reproduction logs are in the local ignored
`artifacts/marvin_wuji/` directory.

On the verified host the installed package's Python was used with
`PYTHONPATH=` and `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` because
the host Conda C++ runtime otherwise takes precedence. This is a host runtime
workaround, not an asset dependency. The test interpreter was Python 3.12.3;
installed versions were
MuJoCo 3.12.0, Isaac Lab 3.0.0, Newton 1.6.1, Warp 1.17.0, PyTorch 2.11.0.
The GPU was NVIDIA RTX 5090 with driver 595.84.

## Structure, frames and parameters

The source is `nq=nv=nu=54`: left arm J1–J7, right arm J1–J7, left hand 20,
right hand 20. `ctrl` order is 7/7/20/20; `qpos` order places each hand after
its arm. `joint_mapping.py` builds both mappings by joint name and validates
unique limited hinges and one actuator transmission per joint. The generated
USD has 54 revolute joints, 20 fixed joints and 74 rigid bodies, matching the
74 non-world MuJoCo bodies. Its articulation root is the fixed pedestal at
`/marvin_m6_dual_arm_wujihand2_beta2/Geometry/pedestal`. It is Z-up in meters.
The Isaac Lab `ArticulationCfg` uses all 54 imported USD position drives and
explicitly exposes the original MuJoCo `qpos` joint order.

The TCP frames are the named bodies `TCP_Link_L` and `TCP_Link_R`; the hand
mounts are part of the same named-body FK comparison. MuJoCo quaternions are
WXYZ and Isaac Lab 3.0 body poses are XYZW; `frames.py` converts explicitly and
uses sign-invariant geodesic angle. No array-order assumption is used for
joint mapping.

The importer preserves all 24 named MJCF sites, including `tcp_L` and
`tcp_R`, as USD geometry markers. It does **not** convert the source's 152
MuJoCo sensors into Isaac Lab sensors; these include joint, TCP, fingertip and
tactile force/torque streams and need explicit Isaac Lab sensor implementations
when those signals are required. It also skips six source cameras, two lights,
four keyframes and the procedural checker texture. The FK and dynamics scripts
read the source keyframes directly from MuJoCo; they are not USD presets.
The four keyframes are preserved separately in a checked-in JSON file for
standalone viewer use; they remain absent as USD timeline presets. Newton
collision and contact behavior is separately checked below. These conversion
losses are recorded in `validation_summary.json`.

The source has 87 collision-enabled geometries and 38 excluded pairs; the USD
has 87 collision geometries (54 via mesh instances) and 38 filtered pairs.
Home produces zero contacts in both backends. Deterministic legal poses found
hand/support and left/right arm contacts in both. Contact counts are not equal:
hand/support 83 MuJoCo versus 32 Newton; arm-cross 103 versus 31 total contacts.
This establishes collision detection, not matching contact manifolds or exact
geometry. Arm capsules/cylinders are simplified collision proxies, separate
from visual meshes. Newton's contact filtering, discretization and manifolds
are not assumed equivalent to MuJoCo's.

The structure validator found no nonpositive dynamic mass, invalid principal
inertia, missing/invalid drive, joint-limit discrepancy or imported
mass/inertia/Kp/Kv/force-limit mismatch against the XML. All 54 imported
`newton:armature` and `newton:friction` values match the source values;
`newton:damping` matches source damping multiplied by π/180, the official
converter's per-radian to per-degree authoring conversion. This was checked
against the installed converter implementation, not inferred from similar
numeric values. Three fixed, massless
frame markers (`center_mount` and the two `TCP_Link` bodies) have no authored
mass/inertia, matching their source role; these are not driven links. MuJoCo
source ranges:
dynamic masses up to 35.09 kg; joint damping 0–3, armature 0.0002–0.08,
friction loss 0–0.1, position Kp 0.18–440, Kv 0.0082–45, positive force
limits 0.2–108. The committed XML arm Kp vector is 400/400/440/340/200/120/85;
for example J6 has source damping 0.6, armature 0.015 and friction loss 0.1,
while the USD authors `newton:damping=0.010471975`, armature 0.015 and friction
0.1. This damping conversion does not establish equivalent solver response.
`build_mjcf.py` contains a different 800/800/500/400/220/140/100 vector.
The physical Marvin arms currently run the SDK's position-follow mode.
The real driver submits bounded joint position goals; the controller's inner
position-loop gains are not exposed by the recorded bag or active driver
configuration. The SDK's `OnSetJointKD_A/B` is documented for **torque-mode
joint impedance** and is not the position loop. The latest physical
`bag_no_camera_20261103_161044` contains commanded joints plus measured
position, velocity and SDK torque feedback, which can support response
identification after unit and delay checks but cannot uniquely reveal an
internal Kp/Kd. WujiHand2's real YAML instead configures per-joint Kp/Kd and
reads them back while disabled; those device gains still are not numerically
interchangeable with the MJCF drive gains. The active source drive and joint
parameters are enumerated in `calibration_status.json` and all are marked
**not calibrated against physical torque/trajectory response**.

The source time step is 0.002 s with gravity (0, 0, -9.81) m/s². Dynamic
comparison also uses 0.002 s; source XML gains are retained without blindly
retuning. The source driver has ideal dynamics bias compensation, which was
not injected into Newton or the standalone MuJoCo drive test. Source physical
inertia, friction, hand mount thickness/load and mechanical/software safety
limits are not yet established as real hardware measurements. Training safety
limits must be configured separately from source mechanical joint limits.

## Measured gates

| Gate | Result | Evidence |
| --- | --- | --- |
| A source | Pass | Pinned XML SHA256; 70 meshes hashed in local import manifest. |
| B structure | Pass | 54/54 joints and controls, 74 bodies, 87 collision geometries, 38 filters, named mappings and parameter checks. |
| C FK parity | Pass | 1,000 legal poses, 16 Newton environments, 74 named bodies each, zero rejected samples or failed TCP comparisons. |
| D motion/wrist | Partial | All 54 controls respond and kinematic fixed-TCP wrist path passes; physical tracking drift remains. |
| E dynamics | Partial | Finite static/step/ramp/sine response measured; backend and tracking discrepancies remain. |
| F platform | Pass | Five unit tests; both existing Gym tasks run 400 steps in 16 environments with 16 timeouts each. |

FK samples include all four source keyframes, J5/J6/J7 positive/negative
probes, near-limit arm configurations and random legal joint vectors. TCP
position error max/mean/P95 was **0.000950/0.000274/0.000597 mm**; TCP
orientation error was **0.0000367/0.0000127/0.0000222°**. The maximum error
among all named bodies was 0.00324 mm and 0.00300°. Acceptance thresholds were
1 mm and 0.1° on TCPs; failed comparisons: **0**. These are conversion and FK
errors, not real robot accuracy claims.

The 16-environment asset smoke performed three Home resets and 200 physical
steps per reset; no nonfinite state or abnormal jump appeared (maximum
0.01335 rad/step). An opposed-command test exercised every one of the 54
named channels across paired environments for 150 steps; minimum positive
versus negative response was 0.05685 rad, with zero unresponsive channels.
Left/right arm and hand commands were set independently in the target buffer.

For each arm, a damped-Jacobian MuJoCo IK path commanded ±14.324° world-Z TCP
rotation in 100 waypoints while maintaining the original TCP location. The
**kinematic** path had maximum position drift 0.000712 mm and orientation
error 0.0000373°, with no limit violation. Newton and MuJoCo then followed
the same 750-step joint target sequence using their physical position drives.
Newton final TCP position error was **22.27 mm left, 23.01 mm right**; MuJoCo
was **33.34 mm left, 33.07 mm right**. Newton final orientation error was
**4.46° left, 5.34° right**. Backend TCP position differences were 17.93 and
20.99 mm. J5/J6/J7 individual and combined step tests remained finite and
within limits, but several did not settle within 0.02 rad in the 0.5 s window.
Thus the fixed-TCP *kinematic model* is consistent, while the physical
execution target is **not yet achieved**. This does not indicate an IK-infeasible
goal; the commanded path itself passed the kinematic checks.

The dynamics CSV covers static hold, arm/hand steps, six single-wrist steps,
two combined-wrist steps, a ramp and a sine trace. No NaN/Inf or joint-limit
violation was observed. Selected final joint differences (Newton versus
MuJoCo) were 0.023 rad for static hold, 0.020–0.026 rad for arm steps,
0.062–0.066 rad for hand steps and up to **0.105 rad** for right J6. These
differences are reported rather than erased by arbitrary gain changes.
Step overshoot, 0.02 rad settling time when reached, tail tracking error,
per-step jump and velocity traces are in the local JSON/CSV and committed
machine summary. Contact solver behavior, drive/discretization differences
and uncalibrated source dynamics remain investigation items.

## Platform regression and next boundary

Actual commands run after final conversion:

```bash
uv lock --check
ruff check source/dexgym scripts/assets scripts/smoke_newton.py tests
ruff format --check source/dexgym scripts/assets scripts/smoke_newton.py tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/newton tests/assets
python scripts/smoke_newton.py --task DexGym-Joint-Tracking-Direct-v0 --steps 400
python scripts/smoke_newton.py --task DexGym-Joint-Tracking-Manager-v0 --steps 400
```

Both task smokes reported observation `(16, 4)`, action `(16, 1)`, finite
states and 16 timeouts. The asset work adds no Gym task and leaves their
registration unchanged. Before Arm S0 physical control experiments, define
explicit operational tolerances, calibrate drive/friction/inertia and hand
mount geometry, and rerun the fixed-TCP execution and collision regressions.
The verified FK and named joint interface can support S0 task design now;
they do not justify a claim of hardware-ready dynamics.
