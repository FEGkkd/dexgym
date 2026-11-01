# Phase 1 contact and signed-pitch follow-up (2026-11-01)

This record supersedes the earlier CapTurn30 success percentages in
`phase1-bimanual-rl-v0.md`. The branch is `codex/bimanual-bottlecap-rl-v0`,
based on the user-selected `8987c9c` baseline. The official native USD is
WujiHand2 Beta 1 at `wuji-description` commit
`c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`; its match to the physical
hand and this engineering bottle's match to a real bottle are unverified.
The existing Allegro/MANO/data worktree edits were not changed for this work.

## Physical diagnosis

The pre-existing success predicate allowed 1 mm error in cap travel for a
30-degree turn whose *entire theoretical travel is only 0.25 mm* at the
example 3 mm pitch. A deterministic per-episode audit of one 200-episode run
found 6 wrong-direction axial outcomes among 196 nominal successes, and 103
nominal successes with more than 0.5 mm pitch discrepancy. Thus the old
196/200 count is a diagnostic of bilateral contact and relative rotation,
not a valid 30-degree thread-coupled opening score.

The cap's 500 Hz load trace with the same Reset, action sequence and seed
showed that the learned hand-contact trajectory develops millimetre-scale
thread error within tens of milliseconds. At 500 N/m and eight solver
iterations, the first >1 mm error occurred at 22 ms, the first >2 mm at
82 ms, and the first >4 mm at 542 ms. During the first second, mean absolute
error was 3.726 mm, maximum 5.752 mm, and maximum requested constraint
force 4.369 N; the 8 N force cap did **not** clip. Zero action from the
same state had maximum error 0.469 mm. Increasing position iterations to
16 reduced lateral drift but did not solve axial compliance. A controlled
3000 N/m plus 16-iteration ablation reduced mean axial error to 0.89 mm,
still far above a credible 30-degree pitch tolerance, and did not produce
a robust policy. The present explicit screw wrench is the limiting physical
approximation under learned finger loading; blindly raising its force cap
cannot address this measured case.
At 500 N/m, a 0.05 mm error generates only 0.025 N of elastic axial
restoring force. A mere 0.5 N persistent axial load would require about
1 mm static spring deflection before damping and other constraints act.
The strict acceptance is therefore a test of the current model's physical
fidelity, not simply a PPO hyperparameter target.

The installed Isaac Lab 3.0 PhysX configuration exposes
`solve_articulation_contact_last`. A controlled ablation of the **same**
older CapTurn policy improved its loose-gate score from 6/200 to 38/200
stable starts at 500 N/m, though its mean thread error remained 2.917 mm.
The setting is exposed as a training/evaluation CLI option rather than
silently changing the baseline. A 1024-environment, two-step GPU smoke with
it enabled returned finite `(1024,277)` Actor and `(1024,421)` Critic
observations.

The current CapTurn predicate requires a bilateral suspended grasp, actual
relative opening of at least 30 degrees, positive cap travel along the
bottle's **local** screw axis, and five consecutive policy steps of
pitch-consistent motion. The allowed travel error is now 0.05 mm, or 20% of
the 0.25 mm theoretical 30-degree travel. The previous intermediate
0.25 mm tolerance remains available in old raw logs only. This 0.05 mm
threshold is an engineering simulation criterion, not a real-bottle
calibration. Positive progress reward uses the same signed travel and
pitch check. Its numeric test rejects wrong-sign travel, nearly zero
travel, and excessive travel at 30 degrees.

## Repeated PPO and independent physics replay

Both runs used 1024 environments, 500 Hz physics, 50 Hz policy, contact-last
solver order, the same single local composite stable Reset sample, and
separate CapTurn PPO seeds. Both descend from the same seed-7 GraspHold
checkpoint. The local composite bank uses an uncommitted user-owned Phase 0G
left sample; its exact copy and SHA are in each ignored run directory.
Consequently these 200-episode replays repeat **one training-family state**.
They do not establish held-out-state or cross-family generalization even
when the evaluator's `--held-out` switch selects the reserved pregrasp rows.

| Run | PPO updates / transitions | 0.25 mm diagnostic score | Final 0.05 mm signed-pitch score | Natural pregrasp |
|---|---:|---:|---:|---:|
| Seed 9, contact-last continuation | 25 / 819,200 | 103/200 | **2/200 (1%)** | 0/200 in prior evaluation |
| Seed 10, fresh CapTurn continuation | 50 / 1,638,400 | 121/200 | **2/200 (1%)** | **0/200** |
| Seed 10, strict-gate fine-tune from preceding row | 25 / 819,200 | Not evaluated | **3/200 (1.5%)** | Not rerun: stable physical gate still fails |

The two first-row checkpoints are respectively
`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_08-14-46_phase1_cap_contact_last_seed9/model_97.pt`
and
`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_08-28-29_phase1_cap_signed_pitch_contact_last_seed10/model_98.pt`.
The strict 200-row evaluation files are
`artifacts/phase1/cap_seed9_strict_pitch_rows200.json` and
`artifacts/phase1/cap_seed10_strict_pitch_rows200.json`; the strict fine-tune
rows are `artifacts/phase1/cap_seed10_strict_finetune_rows200.json`.

Seed 9's final PPO KL was 0.0176, PPO clip fraction 0.2700,
explained variance 0.9428, right contact rate 0.5452 and grouped std near
0.25. Seed 10's corresponding values were 0.0194, 0.2974, 0.9390,
0.4891 and near 0.25. No NaN/Inf occurred. The logged joint-target clip
fraction was corrected to divide by two hand commands per policy step;
earlier reports of this field were twice the actual per-hand fraction.
It was 0.3658 for seed 9 and 0.3809 for seed 10 at their final updates.
The raw normalized action saturation was below 0.1%; joint-target drift,
not sampled-action saturation, remains an issue.

The 200 natural pregrasp episodes of seed 10 all failed before bilateral
contact. Median episode length was 14 policy steps (about 0.28 s), mean
terminal bottle-centre height 0.118 m, right-cap contact ratio zero, and
left-bottle contact ratio 0.82%. Fifteen episodes reported actual ground
contact; the others reached the 0.14 m bottle-centre drop threshold first.
No thread constraint failure occurred. The world-relative right pregrasp
wrist orientation also differs substantially from the single trained stable
state. This is a measured Reset-distribution and contact-formation gap, not
evidence that the preshapes are geometrically invalid.

The strict-gate fine-tune deliberately held the bank, contact-last setting,
action scale and physical parameters fixed. Its last PPO update remained
finite (KL 0.0189, clip fraction 0.2967, explained variance 0.9569,
about 5,979 environment steps/s), and right-cap contact rose to 60.25% in
training. Yet its last rollout had **zero strict success bonus**. In 200
deterministic replays, strict success changed only from 2 to 3, while mean
final pitch-progress discrepancy was 2.093 mm and joint-target clipping
rose to 46.64%. This is not a meaningful policy improvement. A mixed
pregrasp fine-tune was therefore not started: it would make the source
distribution harder before the stable-state physical gate works. The
bounded experiment stopped on diagnostic evidence, not because the
25-million-transition budget was exhausted. Phase 1 cumulative PPO
interaction is **16,617,472 environment transitions**.

For one-environment live PhysX contact auditing, an optional evaluator trace
creates a separate 128-contact cap-to-right-hand view and converts manifold
points into cap-local XYZ. Across 10 seed-10 episodes, 82.1% of normal
contact load on one-step `cap_ok` frames was on the cylindrical sidewall
(`|z|<10.5 mm`, radial distance 14–20 mm); 16.3% was on the top.
Thumb, index and middle all carried sidewall load in most such episodes.
The index load was mainly on its proximal abduction collision body, so this
is a three-digit sidewall contact, not a claim that all three fingertips
pinched the cap. Maximum measured penetration on these `cap_ok` frames was
1.55 mm and peak point normal force 22.94 N. Across *all* frames, including
unsuccessful motion, penetration reached 6.03 mm. Batched training currently
does not reject that deep transient from exact per-point manifold distances;
this remains an unresolved physical-quality limit.

The local JSON traces, per-episode rows, PPO logs and checkpoints are under
ignored `artifacts/phase1/` and `logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/`.
Neither seed currently meets the ≥30% stable CapTurn target under the final
signed-pitch gate. Natural pregrasp GraspHold and end-to-end CapTurn remain
at zero. No model is presented as a successful general opening policy.

The next minimal physics experiment is a one-environment implicit D6
translational drive or equivalent solver constraint whose axial target is
updated from the cap's measured relative angle. It must keep error below
0.05 mm under a controlled 0.5 N local-axis load and an opposing cap-sidewall
grip, preserve equal-and-opposite bottle reaction, and work with a tilted
bottle. Only after that fixture passes should the implementation be proven
GPU-batchable without per-environment CPU writes. A separate curriculum
experiment should then build a geometry-screened bridge from the local
stable tripod to the current natural pregrasps, measuring contact formation
before the 0.28 s free-fall drop boundary. Neither experiment is claimed
complete here.

## Current gates

| Gate | Status | Measured evidence |
|---|---|---|
| DirectRLEnv, 52-D action, vectorized PPO | Pass | 1024-environment, two-seed PPO runs and saved checkpoints; prior 2048-environment throughput smoke. |
| Numerical learning health | Partial | Finite PPO and grouped std near 0.25; joint-target clipping remains 36–47%. |
| Exact cap-sidewall contact | Partial | Live 1-env PhysX points show sidewall-dominant load on candidate valid frames; the 1024-env sensor only supplies per-body forces. |
| Strict stable-state 30-degree turn | Fail | Two seeds 2/200 each; strict fine-tune 3/200, all on one repeated training-family state. |
| Natural pregrasp GraspHold / CapTurn | Fail | Earlier GraspHold 0/200 and seed-10 CapTurn 0/200, with 0.28 s median free fall. |
| Cross-family stable evaluation | Not Tested | No independently validated stable family exists. |
| Real bottle transfer and 4096-env target | Not Tested | Example geometry/torque lacks calibration; 2048-env filter allocation already approaches hardware limits. |

The final offline regression run returned **156 passed, 1 skipped, 2 failed**.
Both failures are the same user-owned Allegro missing-asset fixture and MANO
keypoint-order assertion seen before this follow-up. The focused Phase 1
numeric/contract suite returned **26 passed**. No new Phase 1 test failed.
The final contact-last 1024-environment GPU smoke returned finite Actor
`(1024,277)` and Critic `(1024,421)` tensors and completed two policy
steps with no termination, truncation or numerical exception. Its timed
two-step throughput was 4,395 environment steps/s; this brief smoke is not
a sustained throughput benchmark.
The final one-environment contact-output smoke wrote 1,791 cap/right-hand
points with finite region statistics; its sidewall load fraction was 82.97%
over that episode. The 10-episode audit above is the broader location sample.

## Reproduction

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task cap_turn_30 --policy learned \
  --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_08-28-29_phase1_cap_signed_pitch_contact_last_seed10/model_98.pt \
  --num-envs 64 --episodes 200 --seed 1005 --held-out --source stable \
  --stable-eval-bank artifacts/phase1/composite_power_tripod_local_bank.json \
  --solve-articulation-contact-last --headless
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task cap_turn_30 --policy learned \
  --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_08-28-29_phase1_cap_signed_pitch_contact_last_seed10/model_98.pt \
  --num-envs 64 --episodes 200 --seed 1005 --held-out --source pregrasp \
  --solve-articulation-contact-last --headless
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task cap_turn_30 --policy learned \
  --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_08-28-29_phase1_cap_signed_pitch_contact_last_seed10/model_98.pt \
  --num-envs 1 --episodes 10 --seed 1005 --held-out --source stable \
  --stable-eval-bank artifacts/phase1/composite_power_tripod_local_bank.json \
  --solve-articulation-contact-last \
  --contact-point-output artifacts/phase1/cap_right_contact_points.json --headless
scripts/run_isaaclab.sh scripts/rsl_rl/train.py \
  --task DexterousHands-WujiHand2-Bimanual-CapTurn30-Direct-v0 \
  --num_envs 1024 --seed 10 --max_iterations 25 \
  --stable_train_bank artifacts/phase1/composite_power_tripod_local_bank.json \
  --stable_reset_probability 1.0 \
  --source_bank_snapshot docs/assets/phase0g_left_image_power_v2_grasp_init_bank.json \
  --solve_articulation_contact_last --resume \
  --load_run 2026-11-01_08-28-29_phase1_cap_signed_pitch_contact_last_seed10 \
  --checkpoint model_98.pt --run_name phase1_cap_strict_pitch_finetune_seed10 --headless
scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_rl.py \
  --num-envs 1024 --steps 2 --solve-articulation-contact-last --headless
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/run_isaaclab.sh -m pytest -q tests
```

The local `artifacts/phase1/composite_power_tripod_local_bank.json` is
intentionally ignored because it depends on user-owned, uncommitted Phase 0G
data. Its lack of a committed independent replacement is a reproducibility
blocker. The strict fine-tune checkpoint is
`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_08-46-31_phase1_cap_strict_pitch_finetune_seed10/model_122.pt`.
