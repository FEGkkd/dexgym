# Phase 1 bimanual bottle RL, v0 research record

**Current research status: PARTIAL.** The latest
[strict-hold audit](phase1-strict-hold-audit.md) and
[Mimic grasp replay audit](phase1-mimic-grasp-replay-audit.md) supersede earlier
loose-gate GraspHold and CapTurn30 success percentages in this historical
record. The final Mimic strict-hold checkpoint scored 0/200 on held-out
pregrasp and 0/200 on the training-family evaluation; no stable Mimic bank
or learned CapTurn30 result exists. The
[budget audit](phase1-budget-audit.md) records 24,124,160 logged PPO
transitions, with other environment interactions uncounted.

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
| 2 | GraspHold, 1, 1024 | 32 / 1,048,576 logged | Stable 0/200, pregrasp 0/200 | Left contact reward stalled; training stopped rather than extending unchanged. The update count was corrected from TensorBoard records in the budget audit. |
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
| 1024-env PPO sampling, update, checkpoint | Pass | GraspHold rounds 2–4 and CapTurn rounds 1–2 saved checkpoints; all held-out success rates remain zero. |
| 2048-env smoke/performance | Pass | 20 policy steps, 11,667 env steps/s, 16.83 GiB GPU; slow six-minute startup. |
| 4096-env target | Not Tested | 2048 startup already peaks >36 GiB host; quadratic filter allocation would exceed this 32 GiB GPU. |
| Independent GraspHold learning on geometric pregrasp | Fail | Round 4 held-out 0/200; right tripod contact almost absent. |
| Independent 30-degree opening | Fail | Both CapTurn rounds scored 0/200 stable and 0/200 natural pregrasp. Relative angle alone failed the contact/axial checks. |
| Learning health | Fail | 1024-env PPO finite and checkpointed; KL/clip/variance logged. Joint clipping remains high, especially 98.8% on final pregrasp evaluation, and genuine success did not improve. |
| No reward shortcut | Fail | Zero, random, no-right-contact, left-only, right-only and reverse controls ran. Final strict guard suppresses most unsupported positive progress, but true cap-sidewall contact position is not resolved in the batch view. |
| Legacy compatibility | Pass for Phase 1 code | Latest full suite: 146 passed, one skipped; the same two pre-existing user-worktree Allegro/MANO failures remain. |

No policy is claimed to have learned a bilateral suspended grasp or 30-degree cap opening. Current model files are research checkpoints only.

## Directed physical blocker follow-up (2026-11-01)

The Phase 0E right formation search optimized thumb and middle contact, while
its index finger collision surface was **15.4 mm** from the cap. A new
deterministic official Beta 1 MJCF proxy optimization, seed 0, moved only the
four index joints. The index distal collision surface could reach the cap
sidewall at **1.0 mm** clearance, with 5.64 mm bottle and 4.48 mm other-digit
clearance. An independently fitted open preshape had 4.00 mm cap, 8.18 mm
bottle and 2.33 mm inter-digit minimum clearances. These are *offline geometry
proposals*, never a validated PhysX grasp or a training bank entry. The new
files are `phase1_right_cap_tripod_refined45066.json` and
`phase1_right_cap_tripod_preshape45066.json`.

Official USD/PhysX then formed this candidate without support and measured a
1 s hold and indexed Reset replay. The old thumb-middle radial-opposition
criterion passed, with 0.036 mm bottle motion, 0.147 mm worst formation
penetration and 8.87 N worst formation force. **The requested tripod failed**:
right thumb and middle were loaded for 1.0 s each, but the index was never
loaded during the 1 s hold. Its raw contact manifold showed an approximately
0.20 mm gap and zero normal force. A 0.02 rad index PIP preload still had no
loaded index contact; 0.04 rad disrupted middle contact; a +0.5 mm right-wrist
X offset also failed. Their raw contact traces are retained locally as
`artifacts/phase1/right_tripod_{original,pip02,pip04,wrist_dx05}_physx.json.gz`
(ignored by Git). The unsupported-hold results were respectively old-criterion
Pass but tripod Fail, tripod Fail, tripod Fail, tripod Fail. No candidate was
promoted to the stable bank. The synthesis fixture and its fresh-process
validator/curator now carry a separate, explicit `require_right_tripod`
criterion so an old thumb-middle hold cannot be mislabeled a tripod success.

The index joints tracked their commanded positions to within about 0.3 mrad,
so drive saturation is not the observed cause. In the first unsupported hold
frame, loaded thumb and middle contact points were near -96° and +95° about
the cap's local axis. The unloaded index point was near the cap's negative-X
sector. During
preload, the cap shifted in positive X while the wrist remained fixed; the
index lost contact. Moving its target farther into the cap did not establish
a stable third contact. In 10,000 deterministic fixed-wrist index-joint
samples, 32 brought an index collision surface within 3 mm of the cap in the
opposite positive-X sector, but **none** preserved 2 mm inter-digit and 3 mm
bottle clearance. An explicit +30° sidewall-sector optimizer finished near
+130° instead; its unqualified result is saved as
`phase1_right_cap_tripod_sector30_unqualified.json`. This is evidence about
the present *fixed wrist pose and fixed thumb/middle targets*, not a proof
that the Beta 1 hand cannot make a tripod pinch. The follow-up below tests
closing order and a broader right-wrist/finger search. Any successful proposal
still needs official PhysX hold and fresh-process Reset replay before entering
the stable curriculum.

An isolated, bottle-on-ground, hands-away PhysX diagnostic added a cap-local
axial test force without changing the formal task. At 500 steps (2 ms each):

| Axial load | Axial stiffness | Max screw error | Final screw error |
| ---: | ---: | ---: | ---: |
| 0 N | 500 N/m | 0.130 mm | -0.119 mm |
| 0.5 N | 500 N/m | 0.841 mm | 0.831 mm |
| 0.5 N | 800 N/m | 0.536 mm | 0.519 mm |
| 0.5 N | 2000 N/m | 2.350 mm | 1.998 mm |

The 2000 N/m result rules out blindly stiffening the explicit wrench spring.
The exact cause of its worse error is not yet isolated; solver/contact
interaction and finite force limits remain candidates. The 500/800 N/m
comparison was also run with the **same** CapTurn Round 2 checkpoint,
held-out stable family, 64 environments, 200 episodes and seed 1004. At
500 N/m, success was 0/200 and mean final absolute screw error 1.238 mm;
at 800 N/m, success remained 0/200 and the error increased to 2.598 mm.
Changing the thread stiffness can alter the grasp and cap motion, so the
isolated static improvement does not transfer to this policy. The formal
task default remains **500 N/m**. Both paired evaluation JSON files are in
ignored `artifacts/phase1/`.

Reproduction with the pinned official checkout and Isaac Lab Python:

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
$ISAACLAB_PYTHON scripts/environments/optimize_right_cap_tripod.py \
  --proposal docs/assets/phase0e_right_cap_sidewall_preshape45066.json \
  --proposal_index 45066 --output /tmp/right_tripod_refined.json
$ISAACLAB_PYTHON scripts/environments/prepare_right_cap_preshape.py \
  --reach_scan /tmp/right_tripod_refined.json --output /tmp/right_tripod_open.json
scripts/run_isaaclab.sh scripts/environments/generate_bimanual_grasp_bank.py \
  --headless --left_bank docs/assets/phase0e_pose_left_grasp_init_bank.json \
  --left_sample_index 0 --right_preshape docs/assets/phase1_right_cap_tripod_preshape45066.json \
  --right_proposal_index 45066 --close_s 0.35 --settle_s 0.05 \
  --follow_cap --hand_hand --require_tripod --output /tmp/right_tripod_physx.json
scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_scene.py \
  --headless --bottle_center_z 0.1 --steps 500 --test_axial_force 0.5 \
  --axial_stiffness 800
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task cap_turn_30 --policy learned --checkpoint \
  logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_04-28-05_phase1_cap1024_supported_progress/model_146.pt \
  --num-envs 64 --episodes 200 --seed 1004 --held-out --source stable \
  --thread-axial-stiffness 800 --headless
```

This follow-up made no new PPO update because neither blocker produced a
physically validated task improvement. The learning gates remain **Fail**.
The full offline test suite after these changes was **146 passed, 1 skipped,
2 failed**; the two failures are the existing user-worktree Allegro asset
fixture and MANO keypoint-order tests, which were not modified here.
The standalone scene smoke now generates its diagnostic USD into a separate
temporary directory per run. A one-step Isaac/PhysX smoke returned finite
states and left the existing uncommitted `generated/parameters.json` diff
unchanged.

## Right tripod follow-up: closure order and wrist search

The previously unqualified +130° index-sector candidate was fitted to an open
preshape with 4.00 mm cap, 8.17 mm bottle and 2.33 mm inter-digit proxy
clearance. Native USD/PhysX formation had no support and kept initial
penetration shallow, but the index displaced the dynamic cap before a stable
three-digit pinch formed. The right-cap contact ratio during formation was
10.3%; in the subsequent 1 s hold the bottle moved 399 mm and the tripod
criterion failed. Delaying index closure until 70% of a slower 0.7 s finger
trajectory did not solve it: right-cap formation contact ratio was 7.6%,
hold displacement 398 mm and no loaded tripod occurred. Both results are
**Fail**, not validated Reset samples.

A second directed fixture restored the existing validated Phase 0E
thumb-middle hold and changed **only** the four index joint targets over
0.6 s. During formation the left-bottle and right-cap contact ratios were
100% and 99.7%, with minimum separation -0.034 mm and peak force 2.86 N.
When the index approached the cap, the cap shifted about 4 mm in X and the
middle contact disappeared. In the 1 s hold, loaded index contact lasted
0.036 s, right-cap contact lasted 20.6% of samples, and the bottle moved
153 mm. Thus this targeted addition also **failed**. Normal joint tracking,
modest forces and shallow penetration make a transient force-balance and
contact-layout failure more plausible than a drive or impact failure; the
precise contact-wrench feasibility has not been proven.

The new `optimize_right_cap_tripod_wrist.py` searches the official MJCF proxy
geometry over bounded right-wrist XYZ and world rotation vector, four index
joints and optionally six thumb/middle joints. Its acceptance requires three
sidewall-near points, a maximum cyclic contact-sector gap ≤170°, ≥3 mm
bottle and ≥2 mm inter-digit clearance, and no deep cap penetration. Three
fixed-thumb/middle seeds (0, 1, 2) found **no geometry-qualified proposal**:
the closest selected index points remained on the cap top plane at z=0.658 m,
about 3–4 mm away. One bounded thumb/middle joint search found an index point
at the requested +20° sector but still on that top plane; its middle contact
was 8.26 mm away. This is a failed search over the stated bounds, not proof
that every possible WujiHand2 Beta pose is unreachable. No resulting pose was
used for RL training or promoted to a bank.

Ignored raw evidence is retained locally at
`artifacts/phase1/sector130_{direct,delayed,index_join}.json.gz` and
`artifacts/phase1/right_tripod_wrist_seed{0,1,2}.json`, plus
`right_tripod_wrist_fingers_seed0.json`. The three uncompressed PhysX result
SHA-256 values are, in order,
`323dd521b13c7355832fe3117fbabbe60fd3945f66d243922407b17135a03552`,
`268e937a9da8347ba56154986da647cd8fbdf24b967ff0bee8a33810dabe5a08`,
and `c39d8f7d52d09072bd2a7eab964e8d1d861ba5e74d9dcb975fff5fa16efe9b51`.

Reproduction of the most informative dynamic ablation:

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/environments/validate_right_index_join.py \
  --headless --bimanual_bank docs/assets/phase0e_bimanual_grasp_init_bank.json \
  --sample_index 0 \
  --right_proposal docs/assets/phase1_right_cap_tripod_sector30_unqualified.json \
  --right_proposal_index 45066 --close_s 0.6 \
  --output /tmp/right_index_join_physx.json
$ISAACLAB_PYTHON scripts/environments/optimize_right_cap_tripod_wrist.py \
  --proposal docs/assets/phase0e_right_cap_sidewall_preshape45066.json \
  --proposal_index 45066 --adjust_opposing_digits --seed 0 --maxiter 120 \
  --output /tmp/right_tripod_wrist_fingers_seed0.json
```

The next minimal experiment is a force-aware contact-layout search around
the cap's *actual USD collision surface*, then a gradual physical formation
run. The bank remains four same-family thumb-middle states; there is still
no independently validated right tripod state. More PPO on the current bank
would not address this measured geometry mismatch.
The latest complete offline suite returned **149 passed, 1 skipped, 2 failed**;
the two failures remain the same unmodified Allegro/MANO user-worktree tests.

## Follow-up: physically validated tripod and controlled PPO comparison

The preceding statement that there is no right tripod state describes the
earlier search only. A later official Beta 1 proxy search and native PhysX
formation produced a new result. The bounded proxy proposal had a 168.318°
maximum contact-sector gap, -0.101 mm cap penetration, 3.903 mm bottle
clearance and 2.335 mm interdigit clearance. These values are *proxy geometry*
measurements, not the PhysX validation. The 8 mm requested preshape clearance
resulted in 5.924 mm proxy cap clearance and 2% transient right-cap contact
in the native open-pose diagnostic. The earlier 4 mm open preshape gave 65%
contact in that diagnostic, so it was not a clean open start.

The native PhysX sequence first formed a thumb-index pinch. Moving only the
right middle MCP flexion target by +0.09 rad then yielded an unsupported 1 s
hold with right thumb/index/middle contact ratios 1.0/1.0/0.826. Independent
Reset replay yielded 1.0/0.996/0.84, 0.574 mm bottle displacement and no
cross-object hand contact. This state is saved in
`phase1_right_tripod_bimanual_grasp_init_bank.json`; the preceding
thumb-index state is saved separately in
`phase1_thumb_index_bimanual_grasp_init_bank.json`. Both were curated only
after native physical hold and fresh-process Reset replay. A world-frame
15 mm cap/bottle XY offset initially looked like thread radial drift; it was
caused by a tilted bottle. The bottle-local thread-axis error was 0.047 mm
near the hand and 0.007 mm in the far-hand control. The cap-bottle native
contact force was zero in that open-pose diagnostic.

The new right tripod alone did not satisfy the RL hold gate because its
Phase 0E left source has the thumb plus only one other loaded digit, while
the task requires at least two long fingers. Targeted left-index joint
perturbations either failed to load that digit or disrupted the right
tripod. The user-owned, **uncommitted** Phase 0G left power-grasp bank
contains a separately validated five-digit left state using the same
official asset and bottle configuration. A rigid bottle-relative reframe
combined that left hand state with the right tripod state. Native unsupported
1 s hold and independent Reset replay passed: left thumb/index/middle/ring/
pinky ratios 1.0/0.988/1.0/1.0/1.0; right thumb/index/middle
1.0/1.0/0.830; bottle displacement 0.018 mm; minimum contact separation
-0.063 mm; peak contact 17 N; no cross contact; bottle-local thread-axis
error 0.039 mm. This is physically measured, but the composite bank is kept
in ignored `artifacts/phase1/composite_power_tripod_local_bank.json`
because one source file is user-owned and uncommitted. Its training run logs
snapshot and hash that source. It cannot yet be reproduced from the remote
repository alone, and it supplies only one relative grasp family.

The formal DirectRLEnv 200-episode zero-action control from this local
composite state succeeded **91/200 = 45.5%**. The previous GraspHold Round 4
policy and the older thumb-index-only state each scored 0/200 under the
tripod task predicate. This confirms the old bank/task topology mismatch.
Training from the composite state is a same-family experiment, never a
held-out grasp-family result.

| Composite GraspHold comparison | Seed, envs, updates, transitions | 200-episode stable-start deterministic result | 200-episode held-out natural pregrasp | Diagnostic |
|---|---|---:|---:|---|
| Zero action | 1002, 64 eval envs | 91/200, 45.5% | not measured in this comparison | Physical stable-start reference, not learned. |
| Mean per-digit contact reward | 7, 1024, 50, 1,638,400 | 67/200, 33.5% | 0/200, 100% early failure | Training reward 176.49; right thumb contact 15.65% in final rollout, strict right tripod 8.83%, joint target clipping 70.75%. Reward favored index and middle while thumb was nearly absent earlier in training. |
| Balanced right-digit contact reward | 7, 1024, 50, 1,638,400 | 108/200, 54% | 0/200, 100% early failure | The only major code change was the bounded right-contact aggregation: 0.2 mean + 0.8 geometric mean of saturated thumb/index/middle scores. Final rollout right thumb 15.65%, strict tripod 8.42%, joint target clipping 69.29%. Stable deterministic joint clipping remained 83.44%. |

Both PPO rounds used 500 Hz physics, 50 Hz policy, a 0.85 stable Reset
probability, the same local composite bank, seed 7 and 50 updates. The
second model achieved 54% on its *training-family* stable start, above the
45.5% zero-action reference. It did not improve the held-out natural
pregrasp: neither correct-hand contact formed before the free bottle dropped.
Final second-round PPO diagnostics were 6,030 environment steps/s,
approximate KL 0.0181, clip fraction 0.2857, explained variance 0.8849 and
six action-std groups near 0.25. The action samples themselves rarely hit
[-1, 1], but their integrated **joint targets** frequently reached limits.
This target drift is a distinct issue from Gaussian std or raw action
clipping. The 54% same-family result does not meet the requested ≥60%
held-out natural-pregrasp learning target.

The second PPO checkpoint is
`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_07-00-23_phase1_grasp_composite_balanced_seed7/model_49.pt`.
The first is in the corresponding
`2026-11-01_06-47-54_phase1_grasp_composite_local_seed7/model_49.pt` run.
Raw traces and 200-episode JSON summaries are in ignored `artifacts/phase1/`.
The complete offline suite after this code change was **153 passed, 1
skipped, 2 failed**, with the same user-owned Allegro asset fixture and MANO
ordering failures. Focused Phase 1 tests were **22 passed**. No new Phase 1
offline failure appeared.

The composite curation and evaluation commands on this workstation are:

```bash
scripts/run_isaaclab.sh scripts/environments/validate_composite_grasp.py --help
scripts/run_isaaclab.sh scripts/environments/curate_composite_grasp_bank.py --help
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task grasp_hold --policy learned \
  --right-cap-contact-gate body_only \
  --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_07-00-23_phase1_grasp_composite_balanced_seed7/model_49.pt \
  --num-envs 64 --episodes 200 --seed 1002 --held-out --source stable \
  --stable-eval-bank artifacts/phase1/composite_power_tripod_local_bank.json \
  --headless
```

The current task gate remains **Fail** for held-out natural-pregrasp grasp
learning, and CapTurn30 has no new success claim from this GraspHold result.

### CapTurn30 from the local tripod state

A bounded CapTurn30 run resumed the balanced GraspHold checkpoint with seed 9,
1024 environments, 25 PPO updates and 819,200 further transitions. It used
only the local composite stable bank at Reset. The final checkpoint is
`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_07-08-41_phase1_cap_composite_tripod_seed9/model_73.pt`.
Its final rollout had strict right contact rate 51.5%, joint target clipping
76.5% and mean positive progress reward 0.0068 per step. A training success
reward alone was not used as evidence.

| CapTurn30 independent evaluation, seed 1004 | Episodes | True 30° successes | Final mean opening | Final mean absolute thread error | Finding |
|---|---:|---:|---:|---:|---|
| Learned policy, local stable start, 500 N/m axial stiffness | 200 | 6 (3%) | 2.282 rad | 2.430 mm | Rare successes passed the full contact/pitch predicate; most large-angle turns did not. Mean strict right contact during the rollout was 58.8%. |
| Zero action, same local stable start | 200 | 0 | 0.00386 rad | 0.359 mm | No breakaway in any episode. |
| Learned policy, natural pregrasp | 200 | 0 | -0.0219 rad | 0.0136 mm | 100% failed before correct bilateral contact. |
| Learned policy, right hand moved away at Reset | 64 | 0 | 2.827 rad | 0.160 mm | Zero right contact steps and zero positive progress reward despite unrelated relative rotation. |
| Learned policy, same stable start, **800 N/m** axial stiffness | 200 | 4 (2%) | 2.500 rad | 2.240 mm | Stiffness reduced mean error slightly but did not improve success. |

The 500 to 800 N/m change was an evaluation-only controlled ablation, not
a retrained policy or a recalibrated real thread. The 6/200 result is a
limited *same-family* success under the implemented PhysX success predicate,
not evidence of a robust opening strategy or real-bottle transfer. Exact
contact-point sidewall semantics remain unavailable in this batched sensor
view. No cross-family stable evaluation has yet been made.

### Correction to the GraspHold acceptance rule

The GraspHold 54% and the half finger-increment ablation's apparent 100%
above used the **original contact/suspension predicate**, which did not
check cap/bottle thread pitch error. Under half-scale finger increments,
joint target clipping before the first success was only 6.1%, and both
hands sustained contact, but final absolute thread error averaged
**4.25 mm**. Those episodes are **not physically valid assembly holds**.
The environment now requires <1 mm thread constraint error continuously
for the 0.5 s GraspHold success window, consistent with CapTurn30. An
independent 200-episode replay of that same half-scale checkpoint under
the stricter rule scored **1/200 = 0.5%**; final mean absolute thread error
was 4.04 mm, and the longer episodes accumulated 57.2% joint target
clipping. This revised result supersedes the apparent 100% as a physical
GraspHold claim. The old 54% score is retained only as an explicitly
contact-only diagnostic from the earlier rule and has not yet been
recomputed under the corrected rule.

This is a concrete physics/learning blocker: the current 500 N/m axial
thread spring and 8 N total constraint-force limit allow several
millimetres of pitch mismatch under learned hand loading. Merely increasing
axial stiffness to 800 N/m lowered the CapTurn evaluation's final mean
error from 2.43 to 2.24 mm and lowered success from 3% to 2%, so stiffness
alone at that level did not fix it. The scalar spring fixture passes with
small error under its own load; that does not validate the coupled
hand-loaded situation. Future policy claims must use the corrected gate.

### Final controlled GraspHold rounds under the corrected rule

The strict rule was used for two new 1024-environment, seed-7, 50-update
rounds, both with 0.025 rad/action finger increments and the same 0.85
stable Reset probability. They each collected 1,638,400 transitions.
The second changed only one reward term: beyond 1 mm of thread error,
a bounded penalty approaches -1 per policy step. It is zero at and below
1 mm. This was chosen to make the measured 3–4 mm violations comparable
to the bounded contact reward, and it was verified by a numeric unit test.

| Corrected-rule evaluation, same local stable family | 200-episode physical GraspHold | Mean final absolute thread error | Mean right correct contact over rollout | Joint target clipping | Interpretation |
|---|---:|---:|---:|---:|---|
| Zero action | **91/200, 45.5%** | 0.353 mm | 10.54% | 0% | The bank itself can hold under unchanged commands. |
| Policy trained with corrected success rule, no new error penalty | **0/200** | 3.626 mm | 10.57% | 70.68% | High contact reward, no sustained constraint-valid hold. |
| Pre-correction balanced checkpoint with 0.025 rad/action inference ablation, corrected rule | **1/200, 0.5%** | 4.043 mm | 28.65% | 57.20% | Smaller motion preserved more contact, but not pitch consistency. |
| Policy retrained with bounded thread-error penalty | **0/200** | 0.645 mm | **0%** | 66.68% | It learned to avoid the cap rather than satisfy contact and thread consistency together. |

The final penalty-round checkpoint is
`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_07-37-01_phase1_grasp_thread_penalty_seed7/model_49.pt`.
Its final PPO rollout had mean reward 77.15, strict grasp rate 0.13%,
right tripod rate 0.16%, joint target clipping 29.59%, mean thread penalty
-0.0345 per step and **no success bonus**. Deterministic evaluation on
200 held-out natural pregrasp episodes was **0/200** with 100% early
failure before bilateral contact. Thus reducing the thread error alone
did not yield a useful policy. The older 54% result and apparent 100%
half-scale result were generated by the superseded contact-only success
rule and must never be presented as physically valid grasp success.

The final full offline suite after this correction returned **155 passed,
one skipped, two failed**. The failures are the pre-existing user-owned
Allegro asset fixture and MANO keypoint-order tests. The targeted reward
suite returned 13 passed. GPU evaluations above ran with 64 parallel
environments and 200 independent episodes each. No NaN, global Reset leak
or new test failure was observed in these rounds.

Across the prior six Phase 1 PPO rounds and the five follow-up PPO rounds,
the recorded total is **13,340,672 environment transitions**. The local
composite source is one grasp family and includes an uncommitted user-owned
left-grasp bank, snapshotted with SHA in the ignored run directories. There
is no independently validated, committed cross-family stable evaluation
bank. The supported phase status is **PARTIAL**: DirectRLEnv, vectorized
physics, PPO, checkpoints and counterfactual evaluation work, and rare
same-family CapTurn30 success (6/200) was measured before the stricter
GraspHold correction. The requested ≥60% held-out natural-pregrasp
GraspHold, ≥30% stable CapTurn30 and ≥10% natural-pregrasp CapTurn30
targets all **Fail**. No checkpoint is suitable as a successful policy.

Next minimal physics experiment: with the official USD and the validated
composite Reset, hold fixed finger targets and vary one wrist/contact load
at a time. Log cap/bottle local axial error, D6 motion, thread spring force
and its 8 N clipping, contact positions and normal/tangential forces at
500 Hz; repeat at 500 and 800 N/m and with a measured safe force limit.
Require <1 mm pitch error under the same load before new PPO training.
Then generate a committed independent stable grasp family and a reachable
near-contact curriculum to replace the current natural pregrasp starts,
which all fail before either hand forms contact. These are measurements
to perform, not claims that either change will automatically solve learning.

Reproduction of the final negative result requires the local composite
bank and the snapshotted user source in the run directory:

```bash
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task grasp_hold --policy learned \
  --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_07-37-01_phase1_grasp_thread_penalty_seed7/model_49.pt \
  --num-envs 64 --episodes 200 --seed 1002 --held-out --source stable \
  --stable-eval-bank artifacts/phase1/composite_power_tripod_local_bank.json \
  --finger-increment-scale 0.025 --headless
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task grasp_hold --policy zero --num-envs 64 --episodes 200 --seed 1002 \
  --right-cap-contact-gate body_only \
  --held-out --source stable \
  --stable-eval-bank artifacts/phase1/composite_power_tripod_local_bank.json \
  --finger-increment-scale 0.025 --headless
```

The subsequent [true cap-sidewall contact and PPO report](phase1-sidewall-manifold-round.md)
supersedes body-only cap-contact claims. Its exact-manifold mode is now the
default for new bimanual task runs; the flags above preserve this historical
round's measurement behavior.
