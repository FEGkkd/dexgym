# Phase 1 bimanual bottle RL, v0 research record

## Scope and reproducibility

The work branch is `codex/bimanual-bottlecap-rl-v0`, based on `8987c9c7a152850a7c7a653daaf00b92a2031064`. The user chose this baseline and deferred the right-hand `_start` bank. Existing Allegro, MANO, data and tracking edits in the shared working tree were not staged for this work.

The model is the **candidate** WujiHand2 Beta 1 left and right native USD at `wuji-description` commit `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`. It is not proven identical to the user's physical hand. Bottle dimensions and torque are engineering examples, not calibrated PCO1881 data.

Local GPU setup: RTX 5090 32 GiB, Isaac Sim 6.0.1, Isaac Lab 3.0, PhysX and RSL-RL. For all commands below:

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
```

`/tmp/wuji-description-v2026.9.11` is the local pinned official checkout on this workstation. On another workstation, point the variable to any checkout at the same commit. Training checkpoints and raw evaluation JSON are kept in ignored `logs/` and `artifacts/phase1/` paths. The source bank and experiment summary are tracked.

## Implementation

Two registered `DirectRLEnv` task IDs share `BimanualBottleDirectEnv`:

- `DexterousHands-WujiHand2-Bimanual-GraspHold-Direct-v0`
- `DexterousHands-WujiHand2-Bimanual-CapTurn30-Direct-v0`

The 52-D action is `[left_xyz(3), left_local_rotation_vector(3), left_fingers(20), right_xyz(3), right_local_rotation_vector(3), right_fingers(20)]`. Wrist and finger values are normalized and applied as **target increments**. Zero action holds the current targets. Wrist trajectory speeds are limited by the existing ideal pose controller. Finger joints retain native USD dynamics and 2 N·m limits. Physics runs at 500 Hz and the policy at 50 Hz. The bottle and cap remain free rigid bodies; this task never scripts cap torque or object poses during a physics step.

The contact monitor uses exact PhysX bottle/cap-to-hand body filters and diagonal per-environment gathering. It also checks bottle/cap contact with ground. `BatchedThreadCoupler` maintains continuous angle, stick/sliding state, axial displacement and equal-and-opposite external wrenches each physics substep; the cap's existing D6 joint stays engaged. The batch path has no per-environment CPU loop inside normal physics steps. Selective Reset still performs per-selected-environment USD joint enable writes at the episode boundary.

The Actor has 277 fields, including **oracle simulator object poses**. It has no contact force, friction force or joint effort. The 421-D Critic adds privileged force, joint torque, thread and live tip-site slip diagnostics. Exact contact point and tangential friction vectors are not available in this large-batch view, so the slip field is a tip-site velocity proxy and right-cap sidewall specificity cannot yet be certified from the force matrix alone. The complete field schema is in `phase1_bimanual_rl_schema.json`.

Reset distinguishes four stable grasp states from 32 geometry-qualified pregrasp states. Three stable rows and 24 pregrasp rows are train rows; one stable row and eight pregrasp rows are held out. All four stable rows belong to one relative grasp family, so stable held-out scores are **same-family**, not cross-family generalization. The deferred right `_start` bank is not used.

## Evidence and known limits

The focused offline tests passed and the two-environment GPU smoke passed with `(2,277)` Actor and `(2,421)` Critic observations. The baseline full suite before this branch was 121 passed, one skipped and two failures in user-owned Allegro/MANO work. The final suite was **142 passed, one skipped, the same two failures**. No Phase 1 test failed.

| Round | Task, seed, envs | PPO updates / transitions | Deterministic held-out result | Diagnosis |
|---|---:|---:|---|---|
| 1 | GraspHold, 0, 64 | 50 / 102,400 | Stable 0/200, pregrasp 0/200; these rows were not yet separated from training | The learned controller was worse than zero action on stable starts. |
| 2 | GraspHold, 1, 1024 | 29 / 950,272 | Stable 0/200, pregrasp 0/200 | Left contact reward stalled; training stopped rather than extending unchanged. |
| 3 | GraspHold, 2, 1024 | 50 / 1,638,400 | Stable 0/200, pregrasp 0/200 | Left-first reward improved left contact but lost right tripod contact. |
| 4 | GraspHold, 2, 1024, resumed from round 3 | 50 / 1,638,400 | Stable 0/200, pregrasp 0/200 | Balanced reward improved right contact on stable starts, but no continuous bilateral hold. |

Round 4 checkpoint: `logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_03-57-47_phase1_round4_balanced_contact/model_98.pt`. Its grouped Gaussian std remained finite near 0.25. Raw action saturation in deterministic evaluation was near zero, while joint target clipping was high (~46% on stable and ~54% on pregrasp starts). At stable starts, per-step strict left and right contact rates were ~9% and ~17%; at pregrasp starts, ~84% and ~0.08%. Neither set reached the consecutive 0.5 s hold condition. The actual right pregrasp contact deficit is the strongest observed learning blocker.

The CapTurn30 zero-action counterfactual ran 200 held-out stable episodes: 0% success, 0% breakaway, maximum positive opening 0.00534 rad. Its mean final absolute axial constraint error was 0.463 mm. The Round 4 GraspHold policy sometimes rotates the cap much further but has mean final axial error 4.78 mm on stable starts and 1.69 mm on pregrasp starts; these values violate the CapTurn30 success tolerance of 1 mm. Relative rotation alone must not be interpreted as opening success.

| CapTurn round | Init and change | Stable held-out | Natural pregrasp held-out | Finding |
|---|---|---|---|---|
| 1, seed 5, 1024 envs, 25 updates / 819,200 transitions | Resumed GraspHold round 4; original progress reward | 0/200 | 0/200 | Stable mean final opening 1.32 rad but right strict contact 0.27% of steps and axial error 1.34 mm. Pregrasp mean opening 4.34 rad, right strict contact 0%. |
| 2, seed 5, 1024 envs, 25 updates / 819,200 transitions | Resumed CapTurn round 1; positive progress requires bilateral partial support and <1 mm pitch error | 0/200 | 0/200 | Stable right strict contact improved to 1.67% but failure rose to 40.5%; pregrasp failure rose to 74% and joint target clipping reached 98.8%. This checkpoint is not a successful policy improvement. |

The reward change was evidence-driven: in 64 stable episodes with the right hand deliberately kept off the cap, the round 1 policy had **zero right-cap contact** but could produce up to 0.203 rad positive relative rotation. The old progress term would reward such motion. With the guarded reward, a repeat 64-episode no-right-contact test recorded zero contact steps, **zero positive progress reward**, and zero successes even though transient cap/bottle relative motion remained. This removes that specific reward shortcut; it does not prove authentic cap-turn learning.

Further 64-episode ablations exposed another weaker shortcut: with only learned **left** actions, mean final relative rotation was 2.88 rad and positive progress reward averaged 0.0211 per step; with only learned **right** actions, mean final relative rotation was 0.53 rad and positive progress reward averaged 0.113 per step. Both scored 0/64 successes. A reverse-wrist command produced mean final opening -0.092 rad and mean progress reward -0.121, also 0/64. Random action scored 0/64 and 100% failure. In response, the **final code** only credits positive progress on physics steps meeting the full bilateral GraspHold predicate as well as <1 mm pitch error. Replay of the same left-only policy then reduced positive progress reward to 0.000068 per step (~300-fold); right-only dropped to zero. The small remaining left-only reward comes from rare transient frames that pass the actual bilateral grasp predicate. This final, stricter reward has passed counterfactual tests but has **not** undergone a new long PPO training round: the current stable bank cannot supply the required right tripod, so further training from it would not be a justified use of the budget.

Live PhysX fingertip marker geometry explains the contact bottleneck. In 16 randomized pregrasp resets, right thumb/index/middle tip sites averaged 3.1/6.5/7.6 mm radially outside the cap sidewall, all within the cap's axial height. These are safe near-starts, not formed pinches. In 16 stable-bank resets, the right index tip averaged **34.9 mm radially outside** and **30.8 mm above** the cap center, while thumb and middle tip markers were 17.4 and 21.6 mm radially outside. Those stable samples hold through other finger segments, not a right index-thumb-middle cap-sidewall tripod. Thus the stable-bank early curriculum is geometrically mismatched to the intended CapTurn grasp; merely increasing PPO updates is not a supported next step. Tip sites are marker positions, not actual collision points, so this is reach-geometry evidence rather than a full contact patch reconstruction.

The first 16-environment CapTurn PPO diagnostics smoke completed one update and exposed a high post-update KL of 0.256 and PPO clip fraction 0.807 for its small 512-transition rollout. At 1024 environments, final round 2 KL was 0.0168, clip fraction 0.265, explained variance 0.880 and ~6,180 environment steps/s during training. `BimanualDiagnosticPPO` retains upstream optimization and adds KL, PPO clip fraction, explained variance, actor mean, sampled action magnitude, entropy and six action-std groups to the normal RSL-RL logs. These healthy PPO update numbers did not translate into valid physical success.

The standalone throughput benchmark measured **8,868 environment steps/s at 1024** environments (100 timed policy steps, 8.78 GiB GPU used, 78 ms full reset) and **11,667 steps/s at 2048** environments (20 timed policy steps, 16.83 GiB GPU used, 101 ms full reset). The 2048 setup took about six minutes and peaked above 36 GiB host memory while constructing exact contact filters. Given the filter matrix's quadratic scaling, 4096 is not attempted on this 32 GiB GPU / 125 GiB host: an extrapolation of the observed 2048 allocation exceeds both budgets. This is a measured scalability limit, not a physics pass at 4096.

The expanded 16- and 1024-environment selective Reset checks each ran 100 resets with no physics step between peer snapshots and returned maximum peer-state error 0. They include wrist trajectory state, all thread state, contact history, reward counters and re-enabling an intentionally disabled D6 joint. The legacy Phase 0 contact-classification GPU script passed, as did the independent scalar bottle/thread fixture: full travel stopped near 18.807 rad with maximum pitch constraint error 0.183 mm. This scalar fixture result does **not** erase the 1–2 mm errors observed under RL hand contact.

The standard RSL-RL `play.py` loaded the final CapTurn checkpoint, executed five steps, and exported JIT and ONNX models under its ignored run directory. Future Phase 1 training now writes an exact source/bank snapshot and SHA-256 manifest into its run directory; previous rounds were made from evolving, uncommitted source, so their full source revisions cannot be reconstructed from Git alone. Their saved environment/agent configs, checkpoints, run logs and metrics remain available locally. This provenance gap limits exact replication of the early rounds.

## Commands

```bash
scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_rl.py --num-envs 2 --steps 2 --headless
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/run_isaaclab.sh -m pytest -q tests/test_batched_wrist_pose.py tests/test_batched_thread.py tests/test_batched_contacts.py tests/test_bimanual_rl_contract.py tests/test_bimanual_bank_split.py tests/test_bimanual_reward_terms.py tests/test_bimanual_contact_math.py tests/test_bimanual_ppo_diagnostics.py
scripts/run_isaaclab.sh scripts/environments/validate_bimanual_rl_reset.py --num-envs 16 --resets 100 --headless
scripts/run_isaaclab.sh scripts/environments/benchmark_bimanual_rl.py --num-envs 1024 --steps 100 --headless
scripts/run_isaaclab.sh scripts/environments/benchmark_bimanual_rl.py --num-envs 2048 --warmup-steps 2 --steps 20 --headless
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py --task grasp_hold --policy learned --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_03-57-47_phase1_round4_balanced_contact/model_98.pt --num-envs 64 --episodes 200 --seed 1002 --held-out --source pregrasp --headless
scripts/run_isaaclab.sh scripts/rsl_rl/play.py --task DexterousHands-WujiHand2-Bimanual-CapTurn30-Direct-v0 --num_envs 2 --steps 5 --load_run 2026-11-01_04-28-05_phase1_cap1024_supported_progress --checkpoint model_146.pt --headless
```

## Gate status

| Gate | Status | Evidence / remaining requirement |
|---|---|---|
| Direct environment and 52-D contract | Pass | Both task IDs register; 1/2/16/64/128/512/1024 GPU smoke ran before the latest diagnostics; 2-env smoke repeated after. |
| 1024-env PPO sampling, update, checkpoint | Pass | GraspHold rounds 2–4 and a saved checkpoint; CapTurn follow-up still underway. |
| 2048-env smoke/performance | Pass | 20 policy steps, 11,667 env steps/s, 16.83 GiB GPU; slow six-minute startup. |
| 4096-env target | Not Tested | 2048 startup already peaks >36 GiB host; quadratic filter allocation would exceed this 32 GiB GPU. |
| Independent GraspHold learning on geometric pregrasp | Fail | Round 4 held-out 0/200; right tripod contact almost absent. |
| Independent 30-degree opening | Fail | Both CapTurn rounds scored 0/200 stable and 0/200 natural pregrasp. Relative angle alone failed the contact/axial checks. |
| Learning health | Fail | 1024-env PPO finite and checkpointed; KL/clip/variance logged. Joint clipping remains high, especially 98.8% on final pregrasp evaluation, and genuine success did not improve. |
| No reward shortcut | Fail | Zero, random, no-right-contact, left-only, right-only and reverse controls ran. Final strict guard suppresses most unsupported positive progress, but true cap-sidewall contact position is not resolved in the batch view. |
| Legacy compatibility | Pass | 142 passed, one skipped; the same two pre-existing user-worktree Allegro/MANO failures remain. |

No policy is claimed to have learned a bilateral suspended grasp or 30-degree cap opening. Current model files are research checkpoints only.
