# Phase 1B Beta 2 fixed pregrasp round 0

Date: 2026-11-03. This is one Reset distribution stage of the unified
`DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0` task, not a
separate GraspHold or CapTurn policy. The same 52D actor controls both hands
throughout each Episode. The policy is trained from scratch with a learnable
Gaussian std initialized at 1.0.

The fixed training Reset is local hand-shape seed **66003** in the committed
Beta 2 PGS Mimic pregrasp bank (SHA-256
`6811aa8bd8f58454ddd4f1f1ae780c10a4cdf1532b03fdb353a830041a2a9d63`).
The frozen manifest assigns that seed to training. Its zero-action native
contact probe did not form valid bilateral contact; an active feasibility probe
did at the first 50 Hz action. The one-row training mode checks the manifest
and rejects reserved evaluation seeds. The three held-out local seeds remain
9103120, 9103176, and 9103231. All are one relative template family.

## Pre-training controls

| Control, 200 episodes from seed 66003 | Success | Left/right contact-step rate | Bottle-stable step rate | Max observed opening |
| --- | ---: | ---: | ---: | ---: |
| Zero action | 0/200 | 100% / 0.14% | 100% | 0.035 rad |
| Random Gaussian action, std 1.0 | 0/200 | 76.2% / 6.74% | 17.3% | 1.784 rad |

The random control shows that a large transient relative angle does not pass
the contact, bottle stability, pitch, and continuous Success gate. Its raw
action elements exceed `[-1,1]` 31.74% of the time. The zero control does not
obtain sustained right-cap contact. Neither control supports a learned-skill
claim. Raw JSON episode rows and logs are under the ignored
`logs/phase1b_beta2_fixed66003/` directory.

Native smoke passed 15 zero-action steps at 1 and 1024 environments with
finite 277D actor and 421D critic observations. At 1024 environments, measured
step throughput after scene initialization was 4955 transitions/s, maximum
constraint error 3.63 μm, and right contact rate zero at the final step.

## Controlled PPO baseline

Only the Reset sampler changes from the multi-seed Beta 2 Level A baseline.
The task physics, Success, reward, 52D action scales, and Gaussian actor
distribution remain unchanged. The initial run is 1024 environments,
seed 1101, 1000 requested PPO updates with 32 rollout steps per environment.
The requested count is 32,768,000 PPO transitions. The experiment is subject
to the remaining 12 GPU-hour budget; observed learning and action saturation
will determine whether to stop before 1000 updates.

```bash
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export PYTHONPATH=/home/ubuntu/dexgym
scripts/run_isaaclab.sh rsl_rl/base/train.py \
  --task DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0 \
  --num_envs 1024 --seed 1101 --max_iterations 1000 \
  --run_name beta2_fixed66003_gaussian_std1_seed1101_1024_1000iter \
  --hand_variant beta2 --fixed_pregrasp_seed 66003 --solver_type pgs \
  --init_action_std 1.0 --action_distribution gaussian \
  --unified_reward_profile baseline \
  --source_bank_snapshot docs/assets/beta2_phase1b_pregrasp_bank.json --viz none
```

Training checkpoint selection requires deterministic 200-episode evaluation
on the fixed training state and separately on the frozen held-out local seeds.
The fixed-state result can demonstrate acquisition from this initial pose;
only the latter tests within-template generalization. Beta 2 Level B and
original-distance natural pregrasps remain unqualified and cannot inherit a
near-contact score.
