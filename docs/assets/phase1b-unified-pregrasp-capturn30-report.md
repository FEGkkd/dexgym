# Phase 1B: unified pregrasp to CapTurn30

Status: **PARTIAL — the two-seed near-contact PPO target passed with a
single-policy warm-start; a bounded-action ablation removed action overflow
on A/B, while physical velocity consistency and distant natural pregrasps
remain unresolved.** This report records actual
outcomes only. Training checkpoints, TensorBoard events and
raw contact traces remain in ignored `logs/` and `artifacts/phase1b/`.

## Task contract

Registered ID:
`DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0`.
One 52D Actor (Gaussian baseline or tanh-Gaussian ablation) commands both 20-joint official WujiHand2 Beta 1
hands and both ideal pose-command wrists at 50 Hz. PhysX runs at 500 Hz.
The cap and bottle are dynamic under gravity. The task uses the PGS Mimic
screw at 3 mm pitch. Neither object receives a per-step pose write or
scripted cap torque. The new task has its own configuration, reward, strict
five-step Success, and frozen pregrasp bank. Old GraspHold and CapTurn task
IDs and checkpoints remain available. Finger and bottle mechanical values
are engineering candidates, not hardware-calibrated measurements.
Each 52D action is ordered as left wrist 6D, left fingers 20D, right wrist
6D, right fingers 20D. Nonfinite elements become zero, then each element
is clamped to [-1, 1]. One command maps to at most 0.5 mm per wrist
translation coordinate, 0.02 rad per wrist rotation coordinate and 0.01
rad per finger target increment. Finger targets accumulate from the Reset
joint targets and are clipped to the official joint limits. Wrist targets
accumulate from the Reset wrist poses and use a 0.6-m radial workspace,
0.1-m minimum height, 0.15-m/s linear and 1-rad/s angular trajectory
limits. The wrist is an ideal pose-command boundary; these bounds do not
turn it into a floating-wrist force controller.

Actor observations are 277D and include Oracle Object State for bottle and
cap poses/velocities; they exclude contact forces and internal screw state.
The 421D critic adds privileged contact and thread quantities. The precise
field schema is in the new task's `OBSERVATION_SCHEMA.md`.

## Reset bank and reachability

The 51-row frozen bank has SHA-256
`fcf06057a357e23528b22fc049f8c2607f195b4712f5b410525daf83b7d33193`.
It contains 37 near-contact A rows and 14 weak-contact B rows, all from
**one relative template family**. Train/evaluation local source seeds are
globally disjoint across levels in `phase1b_pregrasp_split.json`.
The split manifest SHA-256 is
`adf9dff639dfc147063e76977129de35e840acd586af56b0b955ad1175cee3d5`;
the fixed evaluation simulator seed is 9117. The globally held-out natural
source seeds are 1625, 1779, 1831, 1842, 1881 and 2261.
The A partition, frozen before PPO, is 29 train / 8 evaluation rows; B is
6 train / 8 evaluation rows. World transforms are not counted as a new
relative grasp family.

The source generator first rejected natural-derived candidates: a PGS
active probe reached valid bilateral contact before the bottle fell in
0/22 initial A candidates. Hybrid candidates use a partially opened
stable-derived left preshape and an opposed natural right tripod whose
offline MJCF Jacobian fit places thumb/index/middle near the cap sidewall.
In native 50/500 Hz PGS simulation, the 51 admitted rows reached correct
bilateral contact under ordinary finger-target actions in 1–8 policy steps
(mean 2). None reached full bilateral contact in the first 15 zero-action
probe steps. The active probe's minimum bottle height was 0.495 m, maximum
correct contact force 5.35 N, maximum cross-object hand force 0 N, maximum
pitch error 3 μm, and first-step reported signed separation −0.092 mm.
The zero probe showed substantial drop in some B states (minimum 0.128 m),
which is why an active, time-limited contact opportunity matters.

PGS Mimic still has a screw-velocity discrepancy: admitted active probes
showed up to 1.27 rad/s absolute bias between PhysX screw velocity and
pose-differenced angular velocity. The earlier Phase 1 contact trace
reported −2.22 mm signed separation; that number is a PhysX contact
distance and has **not** been proven equal to exact geometry penetration.
Neither issue is treated as solved by the new Reset bank.

### Distinct natural-left family reachability audit

After A/B PPO saturated, an additional geometry proposal search tried to
expand the Reset distribution while preserving the frozen A/B split and the
single-task physical rules. Five official left finger flexion joints and one
radial wrist offset were fitted per original natural source in MJCF to
approximately 1-mm local bottle gaps; the right hand came only from the
existing **training-partition** tripod rows. Six natural-left sources passed
the opposed-thumb, four-finger, self-collision and gap screen. The baseline
search produced 11 full bimanual proposals (SHA-256
`1a708a5299d4b88963944fe0e9cfb724e6f253ee5c1833c78f051630a5e44ad4`).
A controlled second search shifted the left wrist an additional 0.75,
1.5 or 2.25 mm inward and produced 16 safe MJCF proposals (SHA-256
`b7a87e0b6c1c44b40bcc7f8649137ba653026e0337472219a4691a68cb06d609`).
These are **proposals, not validated Reset rows**.

Native 50/500-Hz PGS Mimic tests then applied ordinary bounded wrist and
finger target commands for 15 policy steps (0.3 s), with a separate
zero-action replay. The baseline had 9/11 solver-compatible candidates,
the closer group 12/16, but **neither group established valid bilateral
contact in any candidate**; no row was admitted to the bank and no PPO
training consumed these proposals. In the closer group, the left thumb
touched the bottle in only 3/16 candidates, no right index finger touched
the cap, and the required thumb-plus-two-long-finger left contact never formed. The initial
left-thumb *link-origin radial offset* from the bottle proxy surface was
13.6–28.5 mm despite MJCF minimum collider gaps near 1 mm. Those are
different geometric quantities: a proximal segment may approach the bottle
while the thumb tip remains unable to oppose in the available freefall
window. The current original natural D challenge remains physically
unqualified and 0/200 in policy evaluation. After these two controlled
Reset changes failed the same native contact gate, further A-only PPO was
not justified; the next useful work is a native-USD, fingertip-aware
left/right geometry fit and verified support window, not a longer replay
of the already saturated A/B distribution.
An optional thumb-tip-aware offline fit generated eight additional MJCF
proposals (SHA-256
`5f7597400a9c9cb7a9548913219bda685c4322615f2cfc4afe65f8fc4a5d819e`).
It reduced the initial left-thumb link-origin radial offset to about 2 mm.
All eight were solver-compatible in the native PGS probe, yet **0/8**
established valid bilateral contact by 0.3 s; only one obtained any left
thumb/bottle contact, and none obtained right-index/cap contact. All eight
failed by policy step 14. Even a 2-mm wrist-only approach takes at least
four 0.5-mm commands (0.08 s), during which an unsupported bottle falls
about 31 mm in free fall; finger dynamics can change that window but did
not rescue these measured candidates. They are excluded from every PPO
training and success claim. This third negative probe strengthens the
decision to redesign the native contact geometry and support window before
spending more PPO transitions on distant families.

## Tests and baselines

1/2/16/1024/2048-environment Reset/Step smoke probes passed with finite
277D actor and 421D critic observations. The 1024-env 20-step physics probe
reached ~8,001 transitions/s and the 2048-env probe reached ~11,168
transitions/s, measured without PPO updates. At the final zero-action step
the 2048-env left/right contact rates were 96.0% / 0%. All 27 focused
Phase 1B numerical and registration tests passed on the final candidate.
The broader suite yielded 206 passed, 1 skipped and 2 failures
in preexisting Allegro temporary-asset and MANO keypoint-order tests; these
belong to separate user work and have not been modified here.
Final counterfactual verification found and fixed an evaluator dispatch bug:
when a checkpoint was supplied, the labels `zero` and `random` previously
used Actor inference. The repaired dispatcher has a numerical regression
test and its final-checkpoint baselines were rerun; earlier no-checkpoint
zero/random baselines were unaffected.

On the eight frozen A evaluation rows, all with 200 episodes:

| Actor/policy | Success | Additional observation |
|---|---:|---|
| Zero action | 0/200 | Bottle stays stable; occasional right contact, no opening. |
| Gaussian random | 0/200 | 95.5% failure; isolated 2.29 rad maximum opening never passed physical Success. |
| From-scratch baseline, new update 25 | 0/200 | Left contact persists, right contact absent; ~75 mm mean bottle drift. |
| From-scratch baseline, new update 100 | 0/200 | Right contact absent; ~53 mm mean bottle drift. |
| From-scratch contact-coverage reward, update 50 | 0/200 | No stable bilateral support. |
| From-scratch contact-coverage reward, update 100 | 0/200 | No stable bilateral support. |
| Old stable-state CapTurn Actor, direct transfer without new PPO | 9/200 | 4.5%; 91% episode failure; one actor only. |
| Same direct-transfer Actor, right hand moved away | 0/200 | 100% failure. |
| Same direct-transfer Actor, left hand moved away | 0/200 | 100% failure. |
| Warm-start PPO, seed 1101, 51 new updates (`model_450.pt`) | 200/200 | Near-contact A; 23.9% raw deterministic action elements out of range. |
| Same checkpoint, held-out weak-contact B | 129/200 | 64.5%; 31.1% raw action overflow. |
| Warm-start PPO, seed 1101, 101 new updates (`model_500.pt`) | 200/200 | Near-contact A; 28.0% raw action overflow. |
| Same checkpoint, held-out weak-contact B | 200/200 | 26.0% raw action overflow. |
| Warm-start PPO, seed 1101, 201 new updates (`model_600.pt`) | 200/200 | A saturated; raw action overflow worsened to 39.5%. |
| Warm-start PPO, seed 1101, held-out original distant D | 0/200 | Evaluated `model_450.pt`; 100% failure. |
| Warm-start PPO, seed 2202, 2048 env, 51 new updates (`model_450.pt`) | 200/200 | Near-contact A; 26.2% raw deterministic action overflow. |
| Same seed-2202 checkpoint, held-out B | 148/200 | 74%; all eight held-out local rows represented. |
| Warm-start PPO, seed 2202, 2048 env, 101 new updates (`model_500.pt`) | 200/200 | Near-contact A; 31.9% raw action overflow. |
| Same seed-2202 checkpoint, held-out B | 200/200 | 31.1% raw action overflow. |
| Warm-start PPO, seed 2202, 201 new updates (`model_600.pt`) | 200/200 A, 200/200 B | No success gain; action overflow 44.0% A / 44.5% B. |
| Selected seed-2202 `model_500.pt`, held-out original distant D | 0/200 | 100% failure; 22.5% ground contact; 60.3% raw action overflow. |
| Tanh direct transfer before new PPO, held-out A | 9/200 | Same 4.5% as Gaussian direct transfer; action overflow 0. |
| Tanh PPO, seed 2202, 2048 env, 51 new updates (`model_450.pt`) | 200/200 A, 200/200 B | B improves over matched Gaussian 148/200; action overflow 0. |
| Same tanh checkpoint, original distant D | 0/200 | All eight local rows represented; 32.5% ground contact. |
| Tanh PPO, seed 2202, 2048 env, 101 new updates (`model_500.pt`) | 200/200 A, 200/200 B | Action overflow 0; B finger-target clipping 4.5%. |
| Tanh PPO, seed 2202, 2048 env, 201 new updates (`model_600.pt`) | 200/200 A, 200/200 B | Action overflow and finger-target clipping both 0 on A/B. |
| Same tanh checkpoint, original distant D | 0/200 | 43.5% ground contact; no D learning claim. |
| Tanh PPO, seed 2202, 2048 env, 300 new updates (`model_699.pt`) | 200/200 A, 200/200 B | Zero raw action overflow, joint-target clipping and angle faults on both held-out sets. |
| Same final tanh checkpoint, original distant D | 0/200 | 45.5% ground contact; no D learning claim. |

The 9 direct-transfer successes ended in 14–30 policy steps. All had
continuous five-step bilateral support, correct objects in contact, 0.71–1.44
rad opening, 0.343–0.699 mm axial travel, <=10.7 μm screw constraint
error, <=17.8 mm bottle displacement and no angle-jump fault. Those angles
show appreciable overshoot beyond 30° and require separate diagnosis.
They are not evidence that a new PPO policy learned pregrasp control.

The previous CapTurn checkpoint still replayed 200/200 on its same single
stable Reset row after the stricter bottle-stability Success gate was added.
This is only a regression baseline, not a Phase 1B result.

The unified reward's contact scores saturate in [0, 1], approach is a
telescoping signed potential, and positive opening progress is credited
only under stable bilateral support and pitch consistency. Opening progress
is capped at 30°. At 50 Hz over the five-second episode, the maximum
non-turn dense return is 8 for the baseline profile and 22 for the
contact-coverage ablation; both are below the one-time 25-point Success
bonus. Numerical tests reject nonfinite inputs, angle jumps and progress
without correct support. The grasp anchor is captured once when a valid
grasp first appears within 20 mm of the Reset reference, then remains fixed
after later contact loss, so drift cannot be hidden by a moving reference.

The warm-start checkpoint filename uses the source runner's global
iteration number. Because the transferred source is at iteration 400, the
`model_450.pt`, `model_500.pt` and `model_600.pt` rows represent 51, 101
and 201 new updates respectively. The selected seed-1101 checkpoint is
`model_500.pt`: it is the earliest
sampled checkpoint with 200/200 on both A and B. The 200 A successes at
`model_450.pt` lasted 13–16 policy steps and reached 0.725–0.930 rad true
relative opening, 0.346–0.444 mm axial travel, <=16.7 μm screw constraint
error and 9.0–12.6 mm bottle translation. Every trajectory retained correct
bilateral contact and passed the five-step stability gate. Both A and B
evaluation cover eight local held-out reset rows; this is **within one
relative template family**, and D remains unsolved.

Selected-checkpoint counterfactuals on A: zero action, random Gaussian
action, no right hand, no left hand, no hands, and a pure reverse-wrist
action each scored 0/200. A separate test that reversed only the right-wrist
rotation component while retaining all other learned actions scored 3/200;
those cases are genuine positive cap rotations caused by the retained
commands, so this is not a pure reverse-action negative control. Numeric
tests reject missing contact, nonfinite angle and a 6.74-rad single-step
opening jump.
The `no_hands` mode places both hands out of contact and zeros all commands,
leaving the bottle/cap dynamics and Mimic constraint active.

An additional same-initial-pose control uses seed-2202 `model_500.pt` while
freezing one hand's 26 actions at zero without moving it away at Reset.
Freezing the **left** hand scored 0/200 A versus 200/200 for the full
policy; left contact still occurred in 93.7% of observed steps, but right
cap contact collapsed to 0.1% and stable bilateral support to 0.1%.
Freezing the **right** hand likewise scored 0/200, with only 0.7% right
cap contact and 0.4% stable bilateral support. These controls show that
both hands' active policy commands are necessary in this evaluated family;
simple presence of the left hand at Reset cannot explain success.
For the earlier bounded `model_500.pt`, freezing the left hand again
scores 0/200 A. Freezing the right hand while leaving it at its initial
pregrasp pose scores **7/200**, not zero: in those rare cases the moving
left hand and passive right-hand contact can produce a physically valid
relative turn. The formal no-right-contact control moves that hand away
and remains 0/200 for the bounded checkpoint; moving the left hand away
also scores 0/200. This residual passive-right mechanism is reported
rather than filtered out; the full bounded Actor's 200/200 versus 7/200
under frozen right commands shows a large active-control contribution.
The final selected bounded `model_699.pt` was tested again on the same
200 held-out A episodes. Zero action, random action, no right contact, no
left contact and no hands each scored **0/200**. The mixed reverse control,
which keeps the learned policy except for the right-wrist rotation command,
scored 3/200 and is not a pure reverse-only control. With the left or right
26D command frozen at zero but its pregrasp pose retained, each control
scored **1/200**. Both are genuine rare successes from passive contact,
and neither is removed from the denominator. The frozen-left success took
41 steps and reached 1.33 rad, more overshoot than a normal full-policy
success. Thus both active hands strongly improve performance, while the
task's success definition does not require each hand to move on every
individual successful trajectory.

A 500-Hz trace of one successful held-out A trajectory records 1,298
right-hand/cap manifold points, no contact-buffer overflow, maximum screw
constraint error 8.67 μm, left/right peak normal force 41.8/25.8 N and
minimum PhysX signed separation −1.842 mm. The true relative cap opening
was 0.832 rad; the bottle rotated 0.106 rad while the cap rotated 0.855
rad in world coordinates. PhysX screw velocity differed from adjacent-pose
angular speed by up to 6.60 rad/s (mean absolute 1.25 rad/s); integrated
PhysX velocity yielded 0.771 rad versus the 0.832-rad true pose delta.
The 20-ms window discrepancy peaked at 0.0213 rad. These discrepancies,
and the signed contact separation, remain physical-model concerns; signed
separation alone is not a verified geometric penetration depth.
The trace evaluator now computes angular speed directly from adjacent
actual cap/bottle poses, separately from finite differences of the PhysX
joint position. In the earlier 160-substep trace, the true pose delta was
0.83205766 rad and the joint-position delta 0.83205789 rad; their
instantaneous speed difference was at most 0.000234 rad/s. A second
130-substep successful trajectory gives 0.80842926 rad true pose delta,
0.80842977 rad joint-position delta and only 0.78372807 rad when the
reported PhysX joint velocity is integrated. The joint-velocity versus
true-pose speed error has a 5.75-rad/s maximum and 1.28-rad/s mean on
that second trajectory; its minimum signed separation was −1.028 mm.
The cap pose and joint position agree closely; the reported joint
velocity is the inconsistent signal. A unit test verifies that the
pose-based diagnostic ignores whole-bottle world rotation and unwraps
through ±π correctly.
The unified CapTurn Success gate uses the measured angle, axial travel,
correct bilateral contact, bottle pose/velocity stability and constraint
error; it does not use the inconsistent PhysX spin-velocity field. The
spin-velocity field remains relevant to the old GraspHold-only stability
metric and to physical-model health monitoring.
The cap collision is a twelve-segment cube shell plus a top cylinder; its
visible shell is a separate mesh. Transforming the 1,298 reported contact
points into the cap frame gives radial positions 15.44–17.69 mm around a
17.00-mm nominal outer radius (median 16.50 mm), and heights −2.14 to
8.70 mm within the cap's 12.5-mm half height. Signed separation ranges
from −1.842 to +0.414 mm (median −0.643 mm). The proxy geometry and contact
offsets therefore need a direct shape-level audit before those distances
can be interpreted as exact material overlap; no contact threshold was
relaxed to make the rollout pass. The generated cap USD authors no
`contactOffset` or `restOffset` attributes; the effective runtime PhysX
defaults and the hand collision geometry have not yet been measured.

## PPO experiment ledger

The task's std=1.0 Gaussian baseline, 1024 envs, 32 steps/env/update,
seed 1101, from scratch, ran 102 updates (~3.34 million new transitions)
before the explicit no-progress/action-saturation stopping rule was used.
The saved `model_100.pt` scored 0/200. PPO mean reward rose while right
contact and bottle stability fell to zero, and policy-mean elements outside
`[-1,1]` rose from 0 to ~2.5%.

The controlled `contact_coverage` reward profile changed only dense
contact/proximity/stability terms. Its 5-second no-turn return has a tested
upper bound below the +25 Success bonus. At the same seed, bank, actor,
action mapping and 1024-env PPO configuration, it ran ~102 updates; its
`model_100.pt` also scored 0/200, with ~3.5% policy-mean action overflow.
The first two launches of this profile did not complete a PPO update because
new profile-specific reward names were absent from the Episode accumulator;
the regression test and accumulator fix are now in the repository. They are
excluded from transition accounting.

The third experiment initializes the **same single Actor/Critic** from the
old CapTurn checkpoint and restores learnable Gaussian std to 1.0. It uses
the baseline reward, same seed and frozen A training split, and 1024 envs.
It completed 201 new updates (6,586,368 transitions) before early stopping:
A/B success was saturated by global iteration 500 while deterministic action overflow
rose from 28.0% there to 39.5% by global iteration 600. The old checkpoint
itself scored only 9/200 A, so the improvement is due to additional PPO
training on the new task. This is explicitly a warm-start result. The two
from-scratch controlled runs each consumed approximately 3.34 million
transitions and scored 0/200.

The fourth experiment uses the same old Actor warm-start, independent seed
2202 and **2048 parallel environments**. It completed 201 new PPO updates
(13,172,736 transitions), with valid PPO losses and approximately
9,500–10,000 transitions/s during later iterations. At 51 new updates it
already scored 200/200 A and 148/200 B; by 101 updates it scored 200/200
on both. At 201 updates A/B remained 200/200, while deterministic action
overflow rose from 31.9% to 44.0% on A and from 31.1% to 44.5% on B.
It was stopped by the explicit success-plateau/action-saturation rule at
201 updates despite a 300-update command limit. `model_500.pt` is the
selected second-seed checkpoint. Both seeds fail the original distant D
evaluation (0/200), so no distant-pregrasp generalization is claimed.

The first four valid PPO experiments added 26,443,776 transitions in this round:
3,342,336 in each from-scratch controlled run, 6,586,368
in the first warm-start run and 13,172,736 in the second. This is an
accounting figure, **not a transition cap**. The fifth run added 19,660,800,
bringing this round to **46,104,576 new PPO transitions**. Per the user's
latest instruction, there is no transition-count upper limit; the separate
GPU-time budget and evidence-based stopping rules still apply. The two failed
reward-profile launches performed zero PPO updates and are excluded. The second 2048-env
run's mean episode reward reached about 35.6, stable bilateral-support rate
about 96%, approximate KL about 0.018 and explained variance about 0.991
near update 185; the raw sampled-action out-of-range fraction was already
54.4%. These health metrics explain the early stop and the preference for
the earlier selected checkpoint.

A fifth, controlled 2048-env experiment isolated action
parameterization. It keeps seed 2202, frozen A training rows, baseline
reward, 52D action mapping, old single-Actor source checkpoint and
learnable latent Gaussian std 1.0, but transforms the Gaussian sample
through tanh. PPO log-probability includes the inverse transform and
Jacobian; KL is exact in latent space, and entropy uses a reparameterized
Monte Carlo estimate. Numerical tests verify action bounds, exact
log-probability, ratio cancellation and KL; a two-env native PPO update
completed with finite losses and zero raw action overflow. Before any new
PPO, direct transfer under this distribution scored 9/200 A, the same
4.5% as the unsquashed direct-transfer Actor, with zero action overflow.
That untrained transfer produced 177 angle-fault episodes and a 5.25-rad
maximum opening, all rejected unless the strict five-step physical Success
window was met. The 51- and 201-update bounded A evaluations have zero
angle-fault episodes and 200/200 Success.
At 51 new updates the bounded Actor scored 200/200 on both A and B, with
all eight held-out local rows represented and zero raw action overflow.
At the same seed and iteration, the Gaussian Actor scored 200/200 A and
148/200 B, with 27.6% deterministic action overflow on B. This is a
controlled early-checkpoint improvement in weak-contact success, beyond
the trivial action-bound effect. On B, finger-target clipping falls from
41.3% for Gaussian to 19.4% for tanh, and observed stable bilateral
support rises from 15.1% to 58.5%. Both methods retain zero wrist-workspace
clipping. The original distant D remains 0/200,
so the bounded distribution does not solve reachability at that distance.
At 201 new updates A/B remain 200/200 with zero raw-action overflow and
**zero finger-target clipping**. Observed bilateral support rises to 99.1%
on A and 82.1% on B. A's maximum opening falls from 1.016 to 0.903 rad;
B's mean maximum opening is 0.801 rad and worst episode 1.179 rad. The
original distant D still scores 0/200, with 43.5% ground contact.
The run finished normally after 300 new PPO updates (global iteration 699)
in 37 minutes of trainer wall time, at roughly 10,000 transitions/s during
most later iterations; observed peak GPU memory was 19.9 GiB of 32.6 GiB.
The final update had finite value loss 0.0902, approximate KL 0.0136,
clip fraction 0.2055, explained variance 0.9953 and no nonfinite action.
On held-out A/B, `model_699.pt` remains 200/200 with zero raw-action
overflow, zero finger-target clipping, zero wrist-workspace clipping,
zero angle faults, no ground contacts and 99.0%/82.0% observed bilateral
support. Its worst A/B opening is 0.943/0.999 rad; the 30-degree threshold
is 0.524 rad. D is still 0/200 with 45.5% ground contact. We select this
checkpoint over `model_600.pt` because success and target clipping are
equal while the worst B overshoot falls from 1.179 to 0.999 rad.
Across the final 200 A and 200 B successes, all eight held-out local rows
in each split are represented. Completion takes 10–16 A or 11–18 B policy
steps; true opening spans 0.647–0.943 rad on A and 0.701–0.999 rad on B,
with 0.313–0.450 mm and 0.339–0.478 mm axial travel respectively. Maximum
absolute screw constraint error is 13.4 μm on A and 9.1 μm on B, and
maximum bottle displacement from Reset is 10.7/11.8 mm. These are
successful but still moderately overshot 30-degree turns.
At 101 new updates, both Gaussian and tanh reach 200/200 on A/B, but
tanh has zero action overflow versus roughly 31% for Gaussian. Tanh
reduces A/B finger-target clipping from 5.0%/9.3% to 1.9%/4.5% and lowers
the observed maximum opening from 1.092/1.211 rad to 1.016/1.072 rad.
The B stable-support rate is similar (Gaussian 79.2%, tanh 77.5%). The
latent Gaussian std remains learnable; at the tanh checkpoint its six
group means range from 0.786 to 0.968, compared with 0.931 to 1.001 for
Gaussian. The comparison therefore isolates the full distribution change
and its learned consequences, not a manually fixed smaller std.
All 200 bounded B successes came from non-stable Reset rows and retained
correct left-bottle and right-cap contact through the successful five-step
window. They finished in 14–21 policy steps with 0.716–1.232 rad actual
opening, 0.347–0.592 mm axial travel, at most 7.75 μm screw constraint
error, 8.4–15.2 mm bottle displacement and no ground contact. The opening
range shows overshoot beyond 30° despite the progress reward cap; this
remains a control-quality issue rather than a false Success under the
current at-least-30° contract.
A 500-Hz trace of one bounded B success (`model_450.pt`) records 1,158
cap/right-hand contact points across 140 substeps with no buffer overflow,
maximum screw constraint error 3.78 μm, peak left/right normal force
28.9/19.6 N and minimum signed separation −1.845 mm. True pose opening
is 0.801052 rad versus 0.801052 rad from joint-position differences;
integrated PhysX joint velocity yields only 0.707974 rad. The previously
identified Mimic velocity and signed-distance concerns persist under
bounded actions.
A separate successful bounded B `model_500.pt` trace includes actual wrist
poses and all Actor commands at 500 Hz. Over 130 substeps the cap rotates
0.797 rad in world coordinates while the bottle rotates only 0.066 rad;
the real relative opening is 0.770 rad. The right wrist translates 3.33
mm and rotates 0.258 rad; its wrist-rotation action has mean absolute
normalized magnitude 0.808 and its finger actions 0.477. The left wrist
also adjusts (6.84 mm, 0.256 rad). There are 765 right-cap contact points,
with maximum point force 5.60 N and screw constraint error 6.40 μm.
Together with the right-action ablation, this supports active right-hand
contact driving the cap in the full policy; it does not imply every
successful initial condition excludes a passive-right route.
The selected final `model_699.pt` has its own successful held-out B trace
(`artifacts/phase1b/eval_tanh_model699_B_500hz_trace.json`). Across 120
PhysX substeps it records 962 right-cap manifold points and no buffer
overflow. The cap rotates 0.724 rad in world yaw while the bottle rotates
−0.014 rad; measured relative opening is 0.738309 rad and the Mimic joint
position changes by 0.738309 rad. Axial travel is 0.357 mm, consistent
with the 3-mm pitch to within roughly 5 μm. Maximum screw constraint
error is 6.30 μm and peak left/right total normal forces are 23.2/18.5 N.
The minimum signed separation is −1.539 mm and the PhysX joint-velocity
versus pose-differenced speed bias still reaches 7.46 rad/s (mean absolute
1.58 rad/s); integrating PhysX joint velocity gives only 0.684 rad.
Thus the selected checkpoint does not resolve either physical-model
diagnostic.
An independent shape-level audit of those 962 points uses the actual
`cap.usda` collision prims: twelve scaled cube walls and one top cylinder.
None authors a contact or rest offset. The contact-point signed distance
to the union of these external collision shapes ranges from −1.429 to
+0.237 mm, versus −1.539 to +0.421 mm for PhysX's reported pairwise
separation; their Pearson correlation is 0.784. The point with the most
negative PhysX separation lies 1.390 mm inside the cap collision proxy.
This is evidence of substantial proxy overlap at that point, but the
manifold point alone does not give the hand surface location, the effective
runtime contact offset, or exact material penetration. The reproducible
auditor is `scripts/environments/audit_phase1b_cap_contact_geometry.py`;
its raw JSON is
`artifacts/phase1b/eval_tanh_model699_B_cap_geometry_audit.json`.
The same selected Reset and PGS Mimic backend were also replayed after
moving both hands out of contact. Across its 140-step 500-Hz trace it had
zero right-cap manifold points and no Success; the reported joint velocity
matched adjacent true-pose differences to mean absolute 0.000036 rad/s
and maximum 0.000164 rad/s. The loaded successful trace differs by four
orders of magnitude. This localizes the inconsistency to the contact-loaded
articulation trajectory rather than a universal unit conversion or fixed
sample timing offset. Its trace is
`artifacts/phase1b/eval_tanh_model699_B_nohands_500hz_trace.json`.

### Controlled PGS position-iteration probe

The independent evaluator now accepts `--pgs-position-iterations {8,16,32}`.
It sets both `PhysxCfg` position-iteration bounds to the selected value;
the task and its trained checkpoint retain eight as the default. The
installed Isaac Lab `PhysxCfg` documentation states that these bounds
clamp each physics actor's requested solver count. This is a configuration
and trajectory comparison; it does not independently read back the
effective iteration count inside PhysX. The same held-out level-B local
Reset row (index 1), checkpoint `model_699.pt`, evaluation seed 9117,
50-Hz policy and 500-Hz simulation were used for one episode per setting.

| Configured position iterations | Success | Substeps | Pose opening (rad) | Integrated PhysX screw velocity (rad) | Velocity MAE vs pose difference (rad/s) | Maximum velocity bias (rad/s) | Right-cap manifold points | Minimum signed separation (mm) | Maximum constraint error (μm) |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | Yes | 120 | 0.738309 | 0.684361 | 1.577223 | 7.460209 | 962 | −1.539 | 6.300 |
| 16 | Yes | 110 | 0.690272 | 0.706008 | 1.650264 | 6.098259 | 870 | −1.201 | 6.854 |
| 32 | Yes | 120 | 0.788967 | 0.633545 | 1.319192 | 7.884022 | 940 | −1.543 | 9.567 |

Increasing the configured position iterations did not consistently remove
the loaded velocity disagreement or signed-separation concern in this
single-trajectory probe. It also changes the contact trajectory and
episode length, so the table is not a per-substep paired comparison.
No solver setting was adopted from this result. The ignored raw traces are
`artifacts/phase1b/eval_tanh_model699_B_500hz_trace.json`,
`artifacts/phase1b/eval_tanh_model699_B_pgs16_500hz_trace.json` and
`artifacts/phase1b/eval_tanh_model699_B_pgs32_500hz_trace.json`.

### Velocity-state localization

A repeat of the same 120-substep successful B episode additionally logged
the bottle and cap world angular velocities and their difference projected
onto the bottle screw axis. This measured rigid-body relative speed matches
the PhysX Mimic screw speed to **0.00000066 rad/s mean absolute error** and
0.00000286 rad/s maximum error. Independently, the Mimic joint-position
difference matches the true relative-pose angle difference to 0.000098
rad/s mean absolute error. Thus both ways of reading velocity agree with
each other, and both ways of reading position agree with each other; the
disagreement is between reported instantaneous velocity and actual
per-step position change in the contact-loaded trajectory. The speed versus
position-difference error remains 1.577 rad/s MAE and integrates to a
0.05395-rad shortfall over this episode. Using previous-step velocity,
next-step velocity or trapezoidal endpoint integration increases MAE to
1.863, 1.999 or 1.649 rad/s, respectively. A one-substep sampling offset
therefore does not explain
the observed discrepancy. These measurements do not by themselves
identify which PhysX solver operation causes the difference. The no-hands
control above remains the unloaded comparison.

The raw trace and reproducible audit are
`artifacts/phase1b/eval_tanh_model699_B_500hz_velocity_layers_trace.json`
and `scripts/environments/audit_phase1b_velocity_layers.py`. The audit's
JSON summary is
`artifacts/phase1b/eval_tanh_model699_B_velocity_layers_audit.json`.

The seed-1101 warm-start training curve shows mean episode reward rising
from approximately 10 at the first new update to 34 by roughly 100 new updates, while
the measured bilateral stable-support rate rises to roughly 59% at that
point. At the final update it reaches roughly 94%. PPO approximate KL is
about 0.02 around global iteration 500, explained variance about 0.97, and the critic
value loss about 5.7. The six logged Gaussian std groups remain near 1.0
except the left-wrist rotation group, which decreases to about 0.90 by
global iteration 600. Around global iteration 500 approximately 29% of policy-mean elements and
47% of sampled raw action elements lie outside [-1, 1]; the environment
clamps them. Finger target clipping reaches about 19% at that update,
whereas wrist workspace clipping is zero. This is a significant action
parameterization limitation even though real held-out success improves;
simply lowering std is not a demonstrated solution. The first two
from-scratch runs showed rising reward without valid cap contact, so they
were stopped rather than extended blindly.
For a unit Gaussian centered at zero, 31.7% of scalar samples already fall
outside [-1, 1]; the learned nonzero means raise this further. The
deterministic policy's 28–40% overflow is separate evidence that the
actor has learned to rely on action clipping. Because the controller
accumulates clipped wrist and finger targets, many distinct raw Gaussian
actions produce the same command. A bounded distribution with a correct
PPO log-probability and ratio was tested in the fifth controlled run;
changing std alone would not have tested this mechanism.

The selected seed-2202 `model_500.pt` remained 200/200 in a repeat A
evaluation after the final telemetry edit. Measured over the full 20-ms
policy window, mean actual left/right wrist translation speeds were
0.031/0.018 m/s and angular speeds 0.999/0.999 rad/s. The bottle averaged
0.5004 m height, 0.0197 m/s linear speed and 0.223 rad/s angular speed;
the finger-target clip fraction was 4.4% and wrist workspace clipping was
zero. A previous telemetry implementation sampled only the last 2-ms
PhysX substep and falsely read 0 m/s linear wrist motion; the corrected
full-window computation has a numerical regression test and a native
simulation smoke. This telemetry change did not alter actions or rewards.

The diagnostic figures show all five completed PPO runs. The warm-start
x-axis uses the source runner's global iteration 400–699, whereas the
from-scratch x-axis begins at zero. Short throughput dips during the
warm-start runs coincide with concurrent independent evaluations.

![PPO reward, contacts, action overflow, KL and clipping](phase1b_training_curves.png)

![Six action-group Gaussian standard deviations](phase1b_std_curves.png)

![Every reward term](phase1b_reward_terms.png)

![PPO loss, saturation and throughput diagnostics](phase1b_ppo_health.png)

## Reproduction and raw artifacts

Run from the repository root with the project Isaac Lab launcher and
`WUJI_DESCRIPTION_ROOT` pointing to the official WujiHand2 description.
The exact old checkpoint path below is the initial single-policy transfer.
The frozen bank and split manifest are checked into `docs/assets/`.

```bash
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export PYTHONPATH=/home/ubuntu/dexgym
OLD=logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_19-04-52_capturn_pgs_from_grasp_2048_std1_seed42_300iter/model_400.pt
scripts/run_isaaclab.sh scripts/rsl_rl/train.py \
  --task DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0 \
  --num_envs 1024 --seed 1101 --max_iterations 400 \
  --run_name phase1b_seed1101_A_legacy_actor_warmstart_1024 \
  --init_checkpoint "$OLD" --reset_loaded_action_std --init_action_std 1.0 \
  --unified_reward_profile baseline --source_bank_snapshot docs/assets/phase1b_pregrasp_bank.json \
  --source_bank_snapshot docs/assets/phase1_mimic_pgs_grasp_init_bank.json \
  --viz none --device cuda:0
scripts/run_isaaclab.sh scripts/rsl_rl/train.py \
  --task DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0 \
  --num_envs 2048 --seed 2202 --max_iterations 300 \
  --run_name phase1b_seed2202_A_legacy_actor_warmstart_2048 \
  --init_checkpoint "$OLD" --reset_loaded_action_std --init_action_std 1.0 \
  --unified_reward_profile baseline --source_bank_snapshot docs/assets/phase1b_pregrasp_bank.json \
  --source_bank_snapshot docs/assets/phase1_mimic_pgs_grasp_init_bank.json \
  --viz none --device cuda:0
scripts/run_isaaclab.sh scripts/rsl_rl/train.py \
  --task DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0 \
  --num_envs 2048 --seed 2202 --max_iterations 300 \
  --run_name phase1b_seed2202_A_tanh_2048 \
  --init_checkpoint "$OLD" --reset_loaded_action_std --init_action_std 1.0 \
  --action_distribution tanh --unified_reward_profile baseline \
  --source_bank_snapshot docs/assets/phase1b_pregrasp_bank.json \
  --source_bank_snapshot docs/assets/phase1_mimic_pgs_grasp_init_bank.json \
  --viz none --device cuda:0
```

For a fixed held-out evaluation, use the selected model path in
`--checkpoint` and change `--level` between A, B and D. The evaluator
writes both aggregate and episode-level JSON, and `--policy` also accepts
`zero`, `random`, `no_right`, `no_left`, `no_hands`, and `reverse`.
`freeze_left` and `freeze_right` retain the same Reset pose and mask only
that hand's policy action.

```bash
scripts/run_isaaclab.sh scripts/environments/evaluate_phase1b.py \
  --checkpoint logs/rsl_rl/dexteroushands_pregrasp_capturn30_phase1b/2026-11-02_00-42-31_phase1b_seed2202_A_tanh_2048/model_699.pt \
  --action-distribution tanh --policy learned --level A \
  --episodes 200 --num-envs 64 --seed 9117 \
  --output artifacts/phase1b/eval_tanh_model699_A.json \
  --rows-output artifacts/phase1b/eval_tanh_model699_A_rows.json
scripts/run_isaaclab.sh scripts/environments/audit_phase1b_cap_contact_geometry.py \
  --trace artifacts/phase1b/eval_tanh_model699_B_500hz_trace.json \
  --output artifacts/phase1b/eval_tanh_model699_B_cap_geometry_audit.json
scripts/run_isaaclab.sh scripts/environments/evaluate_phase1b.py \
  --checkpoint logs/rsl_rl/dexteroushands_pregrasp_capturn30_phase1b/2026-11-02_00-42-31_phase1b_seed2202_A_tanh_2048/model_699.pt \
  --action-distribution tanh --policy learned --level B \
  --episodes 1 --num-envs 1 --seed 9117 --viz none \
  --physics-trace-output artifacts/phase1b/eval_tanh_model699_B_500hz_velocity_layers_trace.json \
  --output artifacts/phase1b/eval_tanh_model699_B_velocity_layers.json
python scripts/environments/audit_phase1b_velocity_layers.py \
  --trace artifacts/phase1b/eval_tanh_model699_B_500hz_velocity_layers_trace.json \
  --output artifacts/phase1b/eval_tanh_model699_B_velocity_layers_audit.json
scripts/run_isaaclab.sh scripts/environments/propose_phase1b_natural_left_hybrids.py \
  --output artifacts/phase1b/natural_left_hybrid_base_candidates.json \
  --report artifacts/phase1b/natural_left_hybrid_base_proposal_report.json
scripts/run_isaaclab.sh scripts/environments/propose_phase1b_natural_left_hybrids.py \
  --additional-left-inward-m 0.00075 0.0015 0.00225 \
  --output artifacts/phase1b/natural_left_hybrid_closer_candidates.json \
  --report artifacts/phase1b/natural_left_hybrid_closer_proposal_report.json
scripts/run_isaaclab.sh scripts/environments/validate_phase1b_pregrasp_pgs.py \
  --native-bank artifacts/phase1b/natural_left_hybrid_closer_candidates.json \
  --output artifacts/phase1b/natural_left_hybrid_closer_pgs_probe.json \
  --count 16 --steps 15 --probe-fingers fit_approach --probe-order active_first
scripts/run_isaaclab.sh scripts/environments/propose_phase1b_natural_left_hybrids.py \
  --thumb-tip-target-m 0.002 \
  --output artifacts/phase1b/natural_left_thumb_tip_candidates.json \
  --report artifacts/phase1b/natural_left_thumb_tip_proposal_report.json
scripts/run_isaaclab.sh scripts/environments/validate_phase1b_pregrasp_pgs.py \
  --native-bank artifacts/phase1b/natural_left_thumb_tip_candidates.json \
  --output artifacts/phase1b/natural_left_thumb_tip_pgs_probe.json \
  --count 8 --steps 15 --probe-fingers fit_approach --probe-order active_first
```

Raw PPO consoles, TensorBoard events and checkpoints are under
`logs/rsl_rl/dexteroushands_pregrasp_capturn30_phase1b/` and
`artifacts/phase1b/`. In particular, the selected first-seed checkpoint
is `2026-11-01_23-20-26_phase1b_seed1101_A_legacy_actor_warmstart_1024/model_500.pt`.
The selected bounded-action checkpoint is
`2026-11-02_00-42-31_phase1b_seed2202_A_tanh_2048/model_699.pt`.
Its SHA-256 is `9cd2c2d9a1b4716272935f466d0fa2262894454ec2ff2a01695aebd0c4f5a0d5`.
Its trainer console is `artifacts/phase1b/train_seed2202_A_tanh_2048_console.log`.
The 500-Hz successful trace is
`artifacts/phase1b/eval_warmstart_model450_A_500hz_trace_with_pose.json`.
The direct-transfer, from-scratch, reward-ablation, selected-checkpoint
and counterfactual evaluation JSON files all use the `eval_*.json` naming
scheme in that same artifact directory. Checkpoints, event files and trace
JSON are ignored by Git because they are large; the code, frozen reset
bank and experiment report are committed.

## Gate status

| Gate | Status | Evidence or remaining work |
|---|---|---|
| A — task | Pass | Registration, 52D action, 1/2/16/1024/2048 Reset/Step, real PPO updates and deterministic replay. |
| B — physical | Partial | Dynamic contacts, gravity, screw Reset and counterfactuals validated; PhysX screw-speed bias and deep signed contact distance remain unresolved. |
| C — learnable pregrasp | Pass, one family | A/B 51 active-reachable candidates; zero-action probes cannot establish bilateral support; B policy succeeds. Wider family coverage remains future work. |
| D — 1024+ PPO | Pass | Actual 1024-env 201-update, 2048-env 201-update and 2048-env 300-update PPO runs, checkpoints and deterministic replay. |
| E — held-out learning | Pass, warm-start | 9/200 before new PPO versus 200/200 A/B after 101 new updates; zero/random and no-contact controls 0/200. From-scratch PPO failed. |
| F — two-seed >=20% | Pass, warm-start | Independent seeds 1101 and 2202 each reached 200/200 A and 200/200 B after 101 new updates on frozen held-out cases. Both fail D at 0/200. |
| G — health | Partial | No NaN/Inf or PPO explosion; the bounded 300-update run has zero deterministic action overflow and zero target clipping on A/B. The Gaussian comparator reaches 39–44% late overflow, and PGS screw-speed bias remains unresolved. |
| H — regressions | Partial | Old stable CapTurn 200/200 and Phase 1B tests pass; complete suite has two unrelated, preexisting Allegro/MANO failures. |
