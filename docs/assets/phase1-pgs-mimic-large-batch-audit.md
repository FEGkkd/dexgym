# Phase 1 PGS Mimic grasp and large-batch PPO audit

Date: 2026-11-01. Branch: `codex/bimanual-bottlecap-rl-v0`. The user raised
the cumulative transition limit from 25 million to **100 million**. The
approximately 12 GPU-hour limit remains in force. Raw traces, checkpoints,
TensorBoard events and full console logs live in ignored `artifacts/phase1/`
and `logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/`.

## Physical finding and the new bank

The TGS Mimic screw reported about 3.53 rad/s during a right-hand tripod
hold while the measured screw angle changed at about -0.009 rad/s. The error
was load dependent, appeared with right thumb/middle contact, and did not
appear in the no-hand screw fixture. Increasing screw drive resistance or
making the Mimic compliant did not resolve it. A controlled PGS replay
reduced the reported-versus-angle-difference RMS to about 0.102 rad/s. PGS
is therefore an explicit *experimental* solver choice for the new bank;
it does not establish fully validated cap-turn dynamics.

The captured TGS candidate could not be admitted: a fresh PGS-only Reset
lost the middle-finger sidewall contact. Closing the right middle PIP target
by 0.03 rad through its normal drive, allowing the physics to settle, and
capturing the actual state produced a same-solver candidate. It was then
replayed in a fresh process under gravity without object support:

| Measurement | Independent PGS Reset replay |
|---|---:|
| Duration | 1.0 s |
| Left bottle contact / right cap sidewall tripod | 100% / 100% |
| Left thumb-versus-other net-force opposition | 100% of policy steps |
| Left-cap / right-bottle / hand-hand cross contact | 0% / 0% / 0% |
| Initial / worst reported separation | -0.373 / -0.383 mm |
| Maximum left / right net body force | 8.43 / 0.545 N |
| Bottle translation / rotation from Reset | 0.067 mm / 0.00138 rad |
| Maximum Mimic pitch constraint error | 0.000529 mm |
| Left-only ablation bottle drift, with right hand moved away at Reset | 0.192 mm |

The result is committed as the **one-row, one-relative-grasp-family**
[`phase1_mimic_pgs_grasp_init_bank.json`](phase1_mimic_pgs_grasp_init_bank.json).
It carries the official Beta 1 asset hashes, joint order, bottle parameters,
solver mode, actual articulation states, and SHA-256 digests of the ignored
source/replay traces. The curator refuses to mark a state validated without
independent full and left-only replays. This is a stable **Reset candidate**
for controlled curriculum experiments, not a held-out grasp family or a
demonstration of learned control.

The three small candidate/replay JSON files are committed in
[`phase1_mimic_evidence/`](phase1_mimic_evidence/); the curator reproduces
the bank **byte for byte** from them. Full console logs remain ignored.

```bash
scripts/run_isaaclab.sh scripts/environments/curate_mimic_grasp_bank.py \
  --candidate-result docs/assets/phase1_mimic_evidence/row9_pgs_middle_pip_003.json \
  --replay-result docs/assets/phase1_mimic_evidence/row9_pgs_final_hold_replay.json \
  --left-only-result docs/assets/phase1_mimic_evidence/row9_pgs_final_left_only.json \
  --output /tmp/phase1_mimic_recurated_bank.json
cmp docs/assets/phase1_mimic_pgs_grasp_init_bank.json \
    /tmp/phase1_mimic_recurated_bank.json
```

## Previous small-batch counterfactuals

The 64-env CapTurn30 run resumed the earlier unsuccessful GraspHold policy
for 50 PPO updates (102,400 transitions). Its 200-episode deterministic
same-family evaluation yielded **0% CapTurn30 success, 100% episode
failure**, and no sustained final bilateral contact. A 64-episode action
scale ablation reduced the failure rate to 3.1% by changing the wrist
translation increment from 2 to 0.5 mm and finger increment from 0.05 to
0.01 rad, but right-cap contact and CapTurn success remained zero.

The 64-env GraspHold continuation from the old checkpoint reached higher
training reward but only **1/200** deterministic hold successes. A fresh
64-env GraspHold policy reached training reward near 95 yet scored **0/200**
on the stable family. The matched zero-action policy scored **200/200**.
Thus these policies degraded a stable initial grasp. The old dense reward
could accumulate for all 250 policy steps, while true success terminated
after 25 steps with a bonus of only 5; waiting out the episode was more
profitable than reaching the success gate.

The dense contact and holding terms are now integrated using the 0.02 s
policy step; the once-only success bonus is 25. With scores bounded by 1,
five seconds of maximal dense contact and holding can yield at most 14 for
GraspHold or 4.5 for CapTurn30. The numerical invariant has a unit test.
This changes the training objective only; it does not relax physical success.

## 1024 and 2048 environment measurements

The aligned-reward 1024-env GraspHold baseline ran **100 PPO updates / 3,276,800
transitions** with std 0.25. It had no true success-reward events in any
update. Right-cap contact rate rose from 5.3% to 32.8%, but the full grasp
rate ended at 1.7%. Final approximate KL was 0.0183, PPO clip fraction
0.285, action clip fraction 0.0125, explained variance 0.956, and reported
throughput 5126 environment steps/s. These are training diagnostics, not
an independent policy success result.

| PGS smoke | Result |
|---|---:|
| 1, 2, 16 environments | Reset, 30 finite contact steps each; pass |
| 1024 environments | Reset, cap cache error 0, 5 finite contact steps; 5059 env steps/s during stepping |
| 2048 environments | Reset, cap cache error 0, 5 finite contact steps; 9895 env steps/s during stepping |

At 2048, right tripod contact on the identical bank was 39% at the first
step and 82% by step 5, showing a significant first-step contact-cache
warm-up effect. Scene initialization took approximately seven minutes.

A 2048-env, 50-step physical action-noise counterfactual held the same
bank, wrist/finger scales and seed, changing only normalized Gaussian std:

| Std | Any strict hold success within 50 steps | Right tripod contact steps |
|---:|---:|---:|
| 0 | 79.64% | 82.09% |
| 0.05 | 0.93% | 17.04% |
| 0.25 | 0% | 7.19% |

The runs were sequential in one process, so PhysX contact caches may differ
after each Reset. The large loss of right-cap contact with exploration is
nevertheless consistent with the 1024 PPO log. The user subsequently chose
**initial Gaussian std 1.0** for the next PPO run and explicitly rejected
lowering it as a workaround for divergence. A planned std 0.05 trainer was
interrupted before its first update. The completed 2048-env baseline uses
std 1.0; the Phase 1 actor configuration now defaults to 1.0. The installed
RSL-RL Gaussian distribution uses a learnable scalar-space standard
deviation. Std remains learnable, with no clamp or fixed-std change.

The first 2048-env std-1.0 GraspHold run completed **200 PPO updates and
13,107,200 logged transitions** at seed 42. The final update reported
9,998 env steps/s, 19.7 GB GPU memory during training, approximate KL
0.0131, PPO clip fraction 0.2074, explained variance 0.8697, mean raw
element saturation 27.35%, and any-element action clipping 97.81%.
The six final grouped standard deviations (left wrist translation,
rotation, fingers; right wrist translation, rotation, fingers) were
0.853, 0.714, 0.863; 0.857, 0.801, 0.864. These values changed through
PPO learning; no manual std reduction was applied.

Right cap-sidewall tripod contact rose to 97.83% of sampled training
steps, yet bottle stability was only 10.82%, strict instantaneous grasp
1.29%, mean cap spin magnitude 1.218 rad/s, and the success bonus was
never triggered in any logged PPO update. Mean training reward rose to
2.68, showing that dense correct-contact reward can increase without
meeting the continuous suspended-hold criterion.

The final `model_199.pt` was evaluated deterministically on 200 episodes
from the **same one-row grasp family**, with 64 evaluation environments,
PGS Mimic, seed 1001, and the training action scales. Results were
**0/200 GraspHold success, 0% failure, 99.74% right correct-contact steps,
4.90% bottle-stable steps, 0.497 rad/s mean cap spin**. The deterministic
policy had 14.9% joint target clipping. The matched zero-action baseline
on the same bank and seed scored **200/200 success**, with 100% bottle
stability and mean cap spin 0.093 rad/s. This is a policy learning failure,
not a missing viable physical Reset on that family. No cross-family
generalization is established.

The next controlled run changes only the GraspHold reward's *motion*
component: a bounded time-integrated log penalty on actual bottle linear
speed, angular speed and screw spin, scaled by the unchanged strict hold
limits (0.05 m/s, 0.5 rad/s, 0.2 rad/s). This term is zero for CapTurn30
and changes no success, collision or PhysX condition. It starts a fresh
2048-env, seed-42 policy with std 1.0 for 300 updates. The first smoke
exposed an omitted reward accumulator key; after fixing it, a 16-env
five-step zero-action physics smoke passed with finite rewards and
observations. The failed first smoke is retained as
`artifacts/phase1/grasp_pgs_motion_16env_smoke.log`.

This ablation was **stopped early after 107 logged PPO updates /
7,012,352 transitions** because right-cap correct contact fell to
essentially zero from updates 50 through 106 and no success event occurred.
The independently evaluated `model_100.pt` scored **0/200** on the same
stable family, with **0% right correct-contact steps**, 78.75% bottle-stable
steps and 0.039 rad/s cap spin. The no-contact behavior reduced the
unconditional motion cost; it did not solve bilateral grasping. The proposed
300-update run was not completed. Its source snapshot, checkpoint and full
log remain under the ignored training directory. This is a rejected reward
ablation; its physical success criteria were never relaxed.

The next controlled reward revision removes that penalty and leaves the
original contact term intact. It adds a **nonnegative bilateral-contact
stability bonus**: `2*dt*sqrt(left_score*right_score) /
(1 + linear_speed/0.05 + angular_speed/0.5 + abs(cap_spin)/0.2)`.
The bonus is zero when either target contact score is zero, so releasing
the cap cannot earn it. Its five-second maximum is 10; together with the
original maximal contact/hold return 14, dense reward remains below the
success bonus 25. A numerical unit test and 16-env physical smoke passed.
A fresh 2048-env, seed-42, std-1.0 round completed **300 PPO updates /
19,660,800 transitions** as
`grasp_pgs_contact_stability_2048_std1_seed42_300iter`. The last update
reported 10,275 env steps/s, KL 0.0115, PPO clip fraction 0.1733,
explained variance 0.4324, 17.38% raw action element saturation, 31.42%
joint target clipping and nonzero real success events. Grouped std values
(left translation, rotation, fingers; right translation, rotation, fingers)
were 0.700, 0.205, 0.707; 0.703, 0.266, 0.704. The distribution started
at 1.0; these values were produced by PPO updates. Final training-step
right-cap contact was 97.89%, bottle stability 96.48%, strict grasp 58.81%,
and mean cap spin 0.284 rad/s.

Independent deterministic assessment of `model_299.pt` on 200 episodes
from the **same stable Reset row/family** scored **200/200 GraspHold success**,
zero drop/constraint failure, 96.59% bottle-stable steps and a full 25-step
stability streak. This is a real physically checked hold in the current
simulation, but the matched zero-action baseline already scored 200/200
on that row. The policy is therefore proven to preserve this initial grasp,
not to form a new grasp from a natural start or generalize to another grasp
family. Moreover, 73% of evaluation episodes registered thread breakaway,
the mean maximum opening excursion was 0.255 rad, joint target clipping
was 34.08%, and all-three fingertip-site sidewall proximity occurred on
only 0.16% of sampled steps despite valid thumb/index/middle body contacts.
These are important physical and topology limitations.

The fixed 200-episode held-out **geometric pregrasp** assessment scored
**0/200**, with 100% episode failure and no sustained left-bottle or
right-cap contact. The successful same-family stable evaluation cannot
substitute for the requested 60% natural-pregrasp target. The untrained
CapTurn30 transfer baseline from the GraspHold checkpoint also scored
**0/200**, despite 68.5% breakaway occurrence and a maximum transient
opening of 0.714 rad in the batch; correct 30-degree opening plus
contact/pitch/stability dwell was never met.

Offline geometry comparison of the 32 natural pregrasp rows to the sole
stable row (each wrist expressed relative to its own bottle) found left
wrist separations of **21.3–29.1 mm** and right separations of
**20.1–24.3 mm**. Their 20-joint position vectors differ by about
0.80–1.01 rad (left) and 0.96–1.04 rad (right) in L2 norm. With the
controlled 0.5 mm/action wrist scale, even 14 saturated policy commands
move only 7 mm per axis. An unsupported bottle starting near 0.5 m has
an ideal gravity free-fall time of about 0.271 s to the task's 0.14 m
failure height, before accounting for hand/object collisions. These
numbers explain why a stable-bank-only policy with small action increments
cannot plausibly recover the current natural pregrasp; they do not prove
that every other valid pregrasp topology is unreachable. A curriculum
must separately validate closer geometry or use a safe larger approach
command scale before making any natural-pregrasp learning claim.

After this round the logged PPO transition audit read **67,488,512 /
100,000,000** (evaluations and interrupted rollouts omitted). The next
round transfers the GraspHold actor, critic and normalizers to CapTurn30
for 300 updates at 2048 environments. To honor the requested initial std,
the scalar Gaussian std parameter is set back to 1.0 *after* loading the
0.651 source value, and only its Adam moments are cleared. The transfer
metadata is saved as `phase1_std_transfer.json` beside the new checkpoint.

## CapTurn30 2048-environment seed-42 round

The transferred CapTurn30 run completed **300 PPO updates / 19,660,800
transitions** at 2048 environments, seed 42, using PGS Mimic and the
one-row stable bank. The source GraspHold checkpoint's scalar Gaussian
std was 0.651; `--reset_loaded_action_std --init_action_std 1.0`
restored it to **1.0 before the first CapTurn update**, leaving the
transferred actor mean, critic and observation normalizers intact. The
std parameter stayed learnable. No lower fixed std, clamp or reduced
entropy setting was used to hide action saturation. The final update
reported 10,177 environment steps/s, about 19.5 GB GPU memory, KL
0.0126, PPO clip fraction 0.1934, explained variance 0.9450, and grouped
std (left translation/rotation/fingers; right translation/rotation/fingers)
of 0.779/0.595/0.979; 0.753/0.671/0.956. Final mean std was 0.91.

The training signal contained true CapTurn success bonuses, but raw
action-element saturation increased to **52.5%** by the final update.
The maximum opening reported in the training batch reached 6.74 rad;
the run did not report a constraint-failure reward, but this excursion
must be treated as a training-distribution warning rather than an
acceptable 30-degree terminal behavior.

Independent deterministic, seed-1001, 200-episode evaluations on the
**same stable Reset row and grasp family** compared three checkpoints:

| CapTurn checkpoint | Success | Mean final opening | Mean signed axial travel | Raw action saturation |
|---|---:|---:|---:|---:|
| `model_400.pt` | 200/200 | 0.831 rad | +0.402 mm | 23.0% |
| `model_500.pt` | 200/200 | 0.903 rad | +0.431 mm | 29.6% |
| `model_598.pt` | 200/200 | 1.083 rad | +0.519 mm | 36.6% |

For the earliest successful checkpoint, `model_400.pt`, all 200 runs
recorded thread breakaway and positive axial travel, with zero bottle
ground contact and zero thread-constraint failure. Mean absolute pitch
progress error was 0.0047 mm; left-bottle and right-cap correct contact
covered 100% and 92.3% of sampled steps. The model was selected for
further counterfactuals because it had less angular overshoot and action
saturation than the later checkpoints. Its 200-episode result is **not**
independent grasp-family generalization: the bank contains one row. The
right thumb/index/middle *body* contact gate was active 92.3% of steps,
but the stricter simultaneous fingertip-*site* proximity rate was 0%;
contact-point and wrench traces still need individual inspection before
asserting a human-like tripod topology.

The same `model_400.pt` scored **0/200** from the held-out natural
geometric pregrasp set, with 100% episode failure and no sustained
left-bottle or right-cap contact. This confirms that the learned
stable-family turn skill has not solved the full pregrasp-to-turn task.

Matched 200-episode PGS Mimic counterfactuals at seed 1001:

| Action condition | CapTurn success | Physical interpretation |
|---|---:|---|
| Zero actions | 0/200 | No breakaway; mean maximum opening 0.00039 rad. |
| Gaussian random actions, normalized std 1.0 | 1/200 | Mean right-cap correct-contact rate 1.6%; mean maximum opening 0.662 rad does not satisfy sustained contact and pitch success. |
| Learned policy, right hand placed away and right actions zeroed | 0/200 | Right-cap contact count 0; no opening progress. |
| Only learned left actions | 1/200 | Passive right-hand contact sometimes survives; one rare success cannot be attributed to active right control. |
| Only learned right actions | 0/200 | Left contact remained, but right-cap contact was lost. |
| Scripted opposite-direction right-wrist action, no external cap torque | 0/200 | Mean final signed opening -2.352 rad; mean maximum positive opening only 0.00061 rad. |

The first no-right-contact attempt **failed its assertion** because the
script moved the right hand away only at Reset, then let the policy move
it back. The evaluator now zeros right actions for that ablation, and
the corrected run verified zero right-cap contact steps. Both the failed
attempt and corrected trace remain in ignored `artifacts/phase1/`.
An older random-action evaluator used a hard-coded std 0.25 and scored
1/200; the matched counterfactual above uses the now-configurable
`--random-action-std 1.0`, matching the PPO initialization requested by
the user. Its raw actions were never used as an external cap torque.
These controls support real bilateral contact-driven progress in this
simulator; they do not prove robust pregrasp learning or hardware transfer.

The second 2048-environment CapTurn round used seed 43 and the **same
GraspHold source checkpoint**. Its transfer record again confirms std
0.651 -> **1.0** before the first update. It completed **150 PPO updates /
9,830,400 transitions**, about 10,222 environment steps/s at the final
update, KL 0.0142, PPO clip fraction 0.1652 and mean learned std 0.96.
The final training right-cap correct-contact rate was 94.0%; raw action
element saturation was 48.3%. No std reduction was manually imposed.

The same 200-episode deterministic, seed-1001, **same-row** evaluation
gave 200/200 CapTurn30 successes at both seed-43 checkpoints. At
`model_400.pt`, mean final opening was 0.839 rad, signed travel +0.405 mm,
mean absolute pitch error 0.0047 mm, right correct-contact rate 94.3%,
and raw action saturation 24.0%. At `model_448.pt`, corresponding values
were 0.926 rad, +0.443 mm, 0.0013 mm, 98.0% and 27.3%. The early
checkpoint again has less overshoot and saturation. **Two PPO seeds
reproduced the same-family turn result**, although both reused one
GraspHold seed and the identical stable training/evaluation Reset row.
Seed 43 also scored **0/200**, with 100% episode failure and no correct
contact, from natural geometric pregrasp.

A one-environment PhysX contact-point trace of the seed-42 `model_400.pt`
success found 77 cap/right-hand manifold points over 12 policy steps:
35 right-thumb-distal, 22 right-index-distal and 20
right-middle-distal. **99.5% of time-summed normal contact load was on
the cap sidewall**, versus 0.5% on top. The largest total normal force
in one policy step was 19.5 N, the largest individual point force 9.1 N,
and the deepest reported separation **-2.22 mm**. The one-episode
opening was 0.821 rad with +0.393 mm signed travel and no ground or
thread-constraint failure. The -2.22 mm contact depth slightly exceeds
the earlier 1-2 mm diagnostic target, so collision/contact-offset
calibration remains open. The time-summed 191 N in the raw trace is
not an instantaneous force.

The final logged PPO audit is **96,979,712 / 100,000,000 transitions**.
Deterministic evaluations, smokes, physical probes and unlogged
interrupted rollouts are omitted, so 3,020,288 is only an optimistic
upper bound on unused transitions. No additional PPO run was justified
inside that narrow remainder while the natural-pregrasp success rate
remained zero and action saturation was rising.

## Gate status

| Phase 1 criterion | Status | Evidence / limit |
|---|---|---|
| Task registration, 52-D action, reset/step, finite physics, 1024+ batch PPO and checkpoints | Pass | Earlier 1/2/16/1024/2048 smoke, two 2048-env CapTurn runs, checkpoint load in independent evaluator. |
| GraspHold learned from geometric pregrasp | Fail | Seed-42 GraspHold checkpoint 0/200 pregrasp; same-family stable 200/200, matched zero-action also 200/200. |
| CapTurn30 from validated stable Reset | Pass, same family only | Seed-42 and seed-43 checkpoints each 200/200, physical thread pitch and bilateral contact checked. Training and evaluation use the same single Reset row. |
| End-to-end CapTurn30 from natural pregrasp | Fail | Both CapTurn seed-42 and seed-43 0/200, 100% failures, no correct contact before falling. |
| Contact driven opening, not passive screw drift | Pass in tested family | Matched zero and verified no-right-contact ablations 0/200; one-env PhysX cap-sidewall contact trace captured. |
| Cross-family grasp and cap-turn generalization | Not Tested | Only one validated PGS Mimic stable grasp family exists. |
| 4096 environment throughput | Not Tested | 2048 was the largest exercised training batch. |
| Learning health throughout training | Fail | No NaN/KL explosion, but 52.5% final seed-42 raw element saturation, 6.74 rad transient maximum opening in training, and no natural-pregrasp success. |
| Original Wuji task compatibility | Pass for checked tests | 179 passing offline tests; the two remaining failures are existing user Allegro/MANO work, outside this branch's staged files. |

The available evidence supports a **same-row, two-seed 30-degree turn
skill**, not the full Phase 1 end-to-end learning target. In particular,
the evaluator's `--held-out` flag separates its execution from training,
but the same one-row bank was used in both; it must not be described as
a held-out grasp family or a held-out Reset state.

## Reproduction commands

Run from the repository root with `WUJI_DESCRIPTION_ROOT` pointing to
the pinned `wuji-description` checkout. The exact source hashes and
configurations of both 2048-env runs are in their ignored
`phase1_training_manifest.json`, `params/` and `source_snapshot/` files.

```bash
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/run_isaaclab.sh -m pytest -q tests
scripts/run_isaaclab.sh scripts/environments/audit_phase1_budget.py \
  --budget 100000000 --output artifacts/phase1/phase1_budget_audit_100m_final.json

scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task cap_turn_30 --policy learned \
  --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_19-04-52_capturn_pgs_from_grasp_2048_std1_seed42_300iter/model_400.pt \
  --source stable --stable-eval-bank docs/assets/phase1_mimic_pgs_grasp_init_bank.json \
  --held-out --thread-backend mimic --solver-type pgs \
  --right-cap-contact-gate manifold_sidewall --num-envs 64 --episodes 200 \
  --seed 1001 --finger-increment-scale 0.01 --wrist-translation-scale 0.0005 \
  --headless --output artifacts/phase1/reproduced_capturn_seed42_model400.json

# Local desktop GUI; omit --steps to keep playing until the window is closed.
scripts/run_isaaclab.sh scripts/rsl_rl/play.py \
  --task DexterousHands-WujiHand2-Bimanual-CapTurn30-Direct-v0 \
  --num_envs 1 --seed 1001 \
  --load_run 2026-11-01_19-04-52_capturn_pgs_from_grasp_2048_std1_seed42_300iter \
  --checkpoint model_400.pt --thread_backend mimic \
  --stable_bank docs/assets/phase1_mimic_pgs_grasp_init_bank.json \
  --solver_type pgs --right_cap_contact_gate manifold_sidewall \
  --finger_increment_scale 0.01 --wrist_translation_scale 0.0005 \
  --realtime --print_status_every 25 --viz kit
```

Isaac Lab 3.0 defaults to headless when `--viz` is omitted. The generic
play entrypoint now applies inference mode to the actor output, which is
required before Isaac Lab converts the CUDA action tensor for PhysX.
It accepts the validated stable bank and matching PGS/action configuration
without changing the original single-hand play defaults.

## Regression and scope

Full offline suite after the CapTurn std-transfer change: **179 passed, 1 skipped,
2 failed**. The two failures
are in pre-existing user Allegro/MANO work: a test creates the unfixed
Allegro URDF while the current asset adapter expects the `_fixed` filename;
another test expects a MANO ring keypoint from joint 7 while the current
converter returns joint 10. Neither file was changed or staged here.
Focused Mimic bank, reward, bimanual and original Wuji tests passed. No
Allegro/MANO/data content was added to the Phase 1 commit.

Current limits: only one stable grasp family, no proven pregrasp-to-grasp
policy, no validated cross-family CapTurn30 policy, no 4096-env run, and PGS screw
reported angular speed still differs from finite-difference angle speed.
Unsuccessful large-batch PPO does not authorize relaxing contact, pitch or
success thresholds.
