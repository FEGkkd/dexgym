# SuperMimic main-branch integration validation

This release integrates the M4.1–M5 SuperMimic work into DexGym. The reusable
Model-Based RL Core is implemented and usable with task/robot adapters. The M5
SE(3) Reach & Hold policy has **not** been shown to converge reliably. The
original experiment directories and checkpoints were not modified.

## Packaged inputs

| Input | Repository location | SHA256 |
| --- | --- | --- |
| Marvin/Wuji source MJCF | `source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml` | `6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce` |
| Validated bimanual Wave | `source/dexgym/assets/supermimic/references/cmu_141_16_wave_bimanual.npz` | `b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f` |
| Procedural bimanual reference | `source/dexgym/assets/supermimic/references/procedural_bimanual_reach_v1.npz` | `ca4741f01734389dd62f9e64c66cdc9c237c90da6370cf76b13bcda89e143af2` |
| Wave Newton PD reset states | `source/dexgym/assets/supermimic/references/newton_pd_states.npz` | `8f8c6c466c75141db01d55ca0bb0ea4f668f36c7541111ffe52783eecd5f91de` |
| Selected M4.1 Actor (Git LFS) | `models/supermimic/m41_wave_selected_best.pt` | `fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f` |

The MJCF loads all 70 referenced STL meshes and all 54 joints from the
isolated worktree. The extracted wheel was also loaded independently by
MuJoCo, yielding 54 joints and 70 meshes. The wheel build contains the XML, 70 STL files, license
notices, Wave/Procedural NPZ files, PD reset states and provenance reports.
The checkpoint is a separate Git LFS object and requires `git lfs pull`.
The Wave preflight passed with packaged assets and reported 1,357 frames,
exact 50 Hz, and complete historical MuJoCo/Newton physical reports.

## Fresh verification

- Full default test collection: **208 passed, 3 skipped** using the verified
  Isaac Sim Python and project venv import order. SuperMimic subset: **201
  passed**. Modified Python files pass Ruff; wheel build passed.
- Bounded 16-environment Reach Core smoke: 192 real transitions, 24 World
  Model updates, 8 Actor updates, 64 Critic updates, zero nonfinite updates.
  Twelve-step in-process evaluation has no complete Episode and makes no
  success claim.
- Bounded 16-environment Tracking Core smoke: 192 real transitions, 24 World
  Model updates, zero nonfinite updates. The Model Trust Gate withheld Actor
  and Critic updates in this short run. This is correct gating, not a trained
  Tracking policy.
- Independent full M4.1 Wave Newton replay using the packaged checkpoint and
  reference: **1,356/1,356** control steps, no hard termination, no Newton
  contact control or physics steps, no illegal joint state. Left/right mean
  TCP position errors: **2.09/4.65 mm**; mean SO(3) errors:
  **0.295°/1.066°**. Maximum command jump: **0.04 rad**.

The Newton runtime on this host used the existing pinned overlay at
`/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides`. This is
a simulator/Python dependency overlay; it is not included in Git. The large
M5 pilot checkpoint and traces remain in the original experiment directory.
Those pilots produced two strict successes in 192 independent evaluation
Episodes, while the 8/32-step Model Trust Gate remained closed; see
[M5 4096-environment validation](m5_4096_net_hold_validation.md).

## Reproduction

Run from the repository root on the verified host. These commands were
executed with outputs directed to `/tmp`:

```bash
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/preflight.py --output /tmp/dexgym_supermimic_preflight.json
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python -c "import sys,numpy; sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages'); import mujoco,pytest; sys.exit(pytest.main(['-q','tests/newton','tests/assets','tests/supermimic']))"
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py smoke_core_m43 --config configs/supermimic/m43_reach_smoke.json --num-envs 16 --iterations 12 --eval-steps 12 --output /tmp/dexgym_publish_reach_smoke
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py smoke_core_m43 --config configs/supermimic/m43_tracking_smoke.json --num-envs 16 --iterations 12 --eval-steps 12 --output /tmp/dexgym_publish_tracking_smoke
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate --checkpoint models/supermimic/m41_wave_selected_best.pt --reference source/dexgym/assets/supermimic/references/cmu_141_16_wave_bimanual.npz --output /tmp/dexgym_publish_wave_legacy
```
