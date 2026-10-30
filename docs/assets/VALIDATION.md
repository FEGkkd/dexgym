# WujiHand2 Beta / bottle foundation validation, 2026-10-30

## Reproduction context

- DexGym work branch `codex/wujihand2beta-assets-foundation`, based on local `main` commit `2c782b76a001c5039db77ce2133237eaf0fa7e6f`. Fetched `origin/main` was `ebba0f145120195b96c0d79f5944b40dd76f1444`; the local base already contained seven later commits. The original dirty main worktree was not modified.
- Isaac Sim 6.0.1, Isaac Lab 3.0 source checkout, NVIDIA RTX 5090, 2 ms physics step. The official Beta checkout was commit `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95` at `/tmp/wuji-description-v2026.9.11` for this run. For legacy Wuji tasks, `/home/ubuntu/wuji-description` supplied the older first-generation USD.
- All `*_VALIDATION_PASS` messages below were obtained from actual GPU runs. Headless mode was tested; GUI display and HUD were authored but **not tested on a display**. Report values are fixture observations, not calibrated hardware properties.
- Temporary test-only Python packages were installed under `/tmp/dexgym-isaac-deps` because this workstation's Isaac Sim Python lacked several Isaac Lab utilities (`lazy_loader`, `gymnasium`, `prettytable`, `wcwidth`, `hydra-core`). This directory is not a DexGym dependency or committed asset.

## Executed commands and observations

The commands below assume the full Isaac Lab modules are importable by the Isaac Sim Python. On this workstation, the run used `PYTHONPATH=/tmp/dexgym-isaac-deps:$REPO/source:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_physx` for new fixtures; the full test suite and legacy task also included `isaaclab_contrib`, `isaaclab_rl`, `isaaclab_tasks`, and the temporary pytest path. See the main asset document for portable `scripts/run_isaaclab.sh` commands.

| Command | Actual result |
| --- | --- |
| `inspect_wuji_hand2_beta.py` | All four native USDs, dependent USD layers/meshes and MJCF maps audited by a per-side tree hash; 20 active joints/side, five queryable tips/side; Beta differences in the main asset document. |
| `validate_wrist_6dof.py` | `WRIST_VALIDATION_PASS`; left XYZ commanded shifts `0.01126, 0.00964, 0.00796 m`, right XYZ local rotation responses `0.0475, 0.0105, 0.0576 rad`. Peak measured wrist force `40 N`, torque `2 N·m`, line speed `0.555 m/s`, angular speed `5.803 rad/s`. |
| `validate_bottle_thread.py` | `THREAD_VALIDATION_PASS`; short cases each 120 steps, 0.24 s. Low torque `0.01 N·m` yielded `+0.0482 rad` elastic angle; `+0.04 N·m` produced `+0.5495 rad`, `-0.04 N·m` produced `-0.5495 rad`. Tilted 30° in two environments: env 0 `+0.5495 rad`, env 1 `-0.0107 rad` unforced drift. Short-case maximum axial error approximately `0.05 mm`; cap speed `2.374 m/s` during free fall. A ground-supported free bottle run of 600 steps reached `18.815 rad` (2.99 turns), final pitch error `0.125 mm`, maximum error `0.183 mm`. Finite end stop requested up to `0.385 N·m`, then the actually applied coupling torque was capped at `0.08 N·m`. Bottle angular velocity responded to cap torque. |
| `validate_hand2_contact.py --variant beta1` | `HAND2_CONTACT_VALIDATION_PASS`; isolated left contact `33.90 N`, isolated right contact `13.21 N`, inactive hand `0 N`. |
| `validate_hand2_contact.py --variant beta2` | `HAND2_CONTACT_VALIDATION_PASS`; isolated left contact `140.84 N`, isolated right contact `152.56 N`, inactive hand `0 N`; both Beta 2 native USD sides loaded with 20 joints each. |
| `validate_hand2_fingers.py --variant beta1` | `HAND2_FINGER_VALIDATION_PASS`; all 20 joints on each side moved toward independently commanded 0.5 rad targets. Minimum displacement left `0.446 rad`, right `0.455 rad`; inactive hand joint deviation `0 rad`. |
| `validate_hand2_fingers.py --variant beta2` | `HAND2_FINGER_VALIDATION_PASS`; all 20 joints on each side moved toward independently commanded 0.5 rad targets. Minimum displacement left `0.447 rad`, right `0.456 rad`; inactive hand joint deviation `0 rad`. |
| `smoke_task.py --task DexterousHands-Wuji-Hand-Reorientation-Direct-v0 --num_envs 1 --steps 3 --headless` | `SMOKE_OK ... obs=(1, 207) critic=(1, 413) action=(1, 20)`. |
| `smoke_task.py --task DexterousHands-Shadow-Hand-Reorientation-Direct-v0 --num_envs 1 --steps 3 --headless` | `SMOKE_OK ... obs=(1, 84) critic=None action=(1, 20)`. |
| `python -m pytest -q tests` | After repairing a stale Allegro config fixture to create the URDF it actually selects and adding asset-integrity, generic critic-layout, wrist round-trip and thread-physics regression tests: `65 passed, 1 skipped`. The earlier run before that fixture correction was `58 passed, 1 skipped, 1 failed`; the test file was from the pre-existing code and was fixed without changing Allegro runtime behavior. |

`validate_bottle_thread.py` requires ±20 µm final pitch residual, at least 50 µm signed axial travel under ±0.04 N·m, less than 0.1 mm maximum transient constraint error, and at most 0.06 rad no/low torque drift in the short cases. The long case allows 0.3 mm final and 0.4 mm transient error because the free bottle rests on the ground and the axial drive is compliant. These are **engineering acceptance thresholds for this sample**, not measured cap specifications. The 600-step run verifies near-three-turn motion and a finite end stop, but cap separation and re-engagement remain unsupported. Every run checked finite root pose/velocity; no NaN or runaway velocity occurred in these fixtures. The initial contact probe moves the bottle assembly **once before stepping**; no root pose writes are made during the simulation loop.

## Gate disposition

| Gate | Status | Evidence and limit |
| --- | --- | --- |
| A compatibility | Pass for tested paths | Legacy Wuji and Shadow GPU smoke, full unit suite. Train/play execution with a checkpoint was not repeated in this stage; existing API and task registration remain. |
| B WujiHand2 Beta | Pass for tested assets | Both sides and both variants loaded with 20+20 joints, native USD, verified top-level and dependency-tree checksums, independent finger-drive motion and contact forces. Exact hardware revision remains unknown. |
| C floating 6DoF wrist | Pass for short fixture | All six commanded axes responded, physically bounded wrench, short run speeds finite, no per-step pose writes, separate left/right bodies. Hard angular-velocity enforcement for all possible collision loads and long-horizon tracking remain unverified. |
| D bottle and thread | Pass for engineering fixture | Separate free rigid bodies, geometry and pitch, signed 120-step torque response, tilted local axis, reaction, two-environment fixture, near-three-turn ground-supported run and finite end stop. Full cap detachment, real-bottle stiction calibration and grasped unscrewing are not implemented or verified. |
| E engineering | Pass for available headless checks | 1/2 environment headless physics, unit suite, generated source USD and docs. GUI display, large-scale vectorization, train/play with checkpoint, long-horizon contact and real CAD calibration were not tested. |

## Follow-up before RL task design

1. Identify the exact hardware revision and wrist mounting transform from serial number/BOM; measure real finger drive effort and pose limits. The official USD currently sets every finger drive `maxForce` to 2 N·m, which differs from several MJCF limits.
2. Replace the example bottle with measured CAD, thread profile, mass and inertia; measure actual static breakaway and running torque. Test threaded contact under grasp and across longer horizons. The unforced drift should be reduced before dense contact RL.
3. Implement cap exit and re-engagement state if full unscrewing is required. The current D6 travel limit keeps the cap connected.
4. Validate GUI markers/HUD, strict wrist speed enforcement under collision, multi-environment throughput and checkpointed train/play on the target workstation.
