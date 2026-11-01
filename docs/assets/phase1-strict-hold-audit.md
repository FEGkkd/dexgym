# Phase 1 strict suspended-hold audit — 2026-11-01

Branch `codex/bimanual-bottlecap-rl-v0` builds on the user-selected
`8987c9c` baseline. These experiments use the engineering bottle and the
official WujiHand2 Beta 1 *candidate* USD; neither the bottle torque nor
hand hardware match has been measured on the user's hardware. The right
`_start` template remains excluded.

## Why the earlier apparent success was invalid

The deterministic 200-episode training-family evaluation of
`mimic_sidewall_tripod_approach_seed41_50iter/model_49.pt` originally
reported one GraspHold success. A pre-auto-Reset capture of that episode is
saved at `artifacts/phase1/mimic_row5_success_pre_reset_candidate.json`.
It had genuine right thumb/index/middle cap-sidewall forces of
13.38/15.66/2.23 N, but the bottle had moved **28.94 mm** from Reset, its
linear and angular speeds were about **0.076 m/s** and **2.63 rad/s**, and
the cap was spinning at **4.25 rad/s**. The old success gate admitted a
moving assembly because it checked a broad suspended workspace and bottle
linear speed below 0.2 m/s. The captured state is diagnostic only. It is
not a validated stable-grasp Reset sample or evidence of cap-turn learning.

`bottle_hold_stability` now requires a continuous 0.5 s bilateral hold
window with bottle drift below 20 mm from the window anchor, tilt below
10 degrees, linear speed below 0.05 m/s, angular speed below 0.5 rad/s,
and cap spin below 0.2 rad/s. The limits are engineering acceptance values,
not hardware calibration. The cap-turn task uses the same bottle-motion
limits but permits actual cap spin. Contact topology, thread constraint,
and gravity-on suspended-space checks still apply. Reset independently
clears the per-environment hold anchor and streak counters. The evaluation
tool can optionally capture the complete pre-Reset state of a real success;
it labels that capture as requiring independent Reset replay.

Re-evaluating the earlier checkpoint with the strict gate gave **0/200**
training-family successes. No new stable Mimic Reset bank was created.

## Controlled PPO round and independent evaluation

The strict-gate PPO run used 1024 environments, seed 41, natural pregrasp
Reset only, the Mimic thread backend, actual PhysX cap-sidewall manifold
contact, and the tripod-sidewall proximity reward. It completed **25 PPO
updates and 819,200 transitions** in about 115 s at a final sampling rate
of 6,899 environment steps/s. The checkpoint and exact source snapshot are
in the ignored directory:

`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_14-03-44_mimic_strict_hold_seed41_25iter/`

The final sampled right correct-contact rate was **0.11%**, bottle-stable
rate **1.97%**, and strict bilateral grasp rate **0%**. The policy's grouped
action standard deviations remained 0.2482–0.2512. Approximate KL was
0.0163, clip fraction 0.2161, explained variance 0.7504, action clipping
1.41%, and joint target clipping 6.71%. No NaN/Inf or training exception
was observed. These optimizer metrics do not imply a learned grasp.

The fixed 200-episode held-out pregrasp evaluation scored **0/200**;
the separate 200-episode training-family evaluation also scored **0/200**.
Both used deterministic actions, seed 1001, 64 parallel environments, and
the final checkpoint. The held-out rows are reserved from the same
pregrasp template family, so this is not a cross-family claim. The held-out
evaluation had zero correct right-sidewall contact steps; the
training-family evaluation had some contact but no sustained strict hold.
The exact JSON and stdout are retained under `artifacts/phase1/` as
`mimic_strict_hold_seed41_25iter_{heldout,trainfamily}_200.*`.

**Source-version note:** The training source snapshot uses `_stable_steps`
to reset the bottle hold anchor. Shortly after launch, a separate
`_hold_motion_steps` counter was added so cap-turn's allowed cap spin cannot
continually refresh the bottle-motion anchor. The two deterministic
evaluations used this final counter. The stricter physical acceptance
thresholds were identical, but the evaluation environment is not bytewise
the same as the training snapshot. The snapshot and checkpoint are retained
for exact training reproduction. No conclusion of success depends on this
change, and no extra PPO run was attributed to the final code.

## Verification and remaining blocker

`bottle_hold_stability` has an isolated regression test using the captured
false-positive state and independent drift, tilt, linear speed, angular
speed and cap-spin tests. The offline bimanual suite passed **48/48** and
the Isaac Lab task contract passed **3/3**. Mimic selective Reset passed
100 operations each at 2 and 16 environments with maximum instantaneous
peer-state error **0.0**, including the new hold anchor and motion counter.
The full repository result was **168 passed, 1 skipped, 2 failed**. The two
failures are in concurrently user-edited Allegro/MANO code:
`test_allegro_hand_track_config.py` expects a different temporary URDF
filename, and the untracked `test_checkpoint1_helpers.py` expects a
different MANO joint mapping. Their tracebacks are recorded in
`artifacts/phase1/strict_hold_full_pytest.log`; neither area was edited or
included in this Phase 1 change.
The final code also completed a 1024-environment, three-step Mimic smoke
with finite observations and rewards, zero cap-cache reset error and
7,416 environment steps/s in the measured stepping interval. This short
zero-action run checks throughput and state consistency, not grasp success.

The cumulative logged PPO count is **24,124,160** transitions out of the
25-million limit, as corrected by the [full event-log budget audit](phase1-budget-audit.md).
Evaluation, short probes, and interrupted unlogged rollouts add further
interactions; the remaining total budget is therefore less than 875,840.
Additional identical PPO updates have no evidence-based justification.
The immediate blocker is the lack of a Mimic-compatible dynamically stable
grasp Reset family: the existing legacy stable bank violates the hard-pitch
Mimic geometry by roughly 1.2–1.26 mm, while the natural pregrasp policy
almost never forms a valid right cap-sidewall tripod. CapTurn30 was not
trained on a stable Mimic start and no 30-degree learned opening success is
claimed. The next small physical experiment should use contact feedback
to close the right index and middle onto the cap's interior sidewall,
measure cap spin and bottle drift, then independently Reset and replay any
settled bilateral state before using it as a stable-bank curriculum start.
The resulting targeted experiment and failed fresh-process replay are
documented in [the Mimic grasp replay audit](phase1-mimic-grasp-replay-audit.md).

| Gate | Status | Evidence |
|---|---|---|
| 1024-env PPO and checkpoint | Pass | 25 updates, 819,200 transitions, saved `model_24.pt` and source snapshot. |
| Correct cap-sidewall contact | Pass at one env; partial at 1024 | Exact manifold classifier was previously cross-checked point by point; this round trained at 1024 without invalid-buffer samples in the final iteration. |
| Held-out natural-pregrasp GraspHold >=60% | Fail | 0/200 deterministic episodes. |
| CapTurn30 from stable Mimic states >=30% | Not Tested | No dynamically replayed stable Mimic bank. |
| End-to-end pregrasp CapTurn30 >=10% | Not Trained | Bilateral contact did not stabilize. |
| 4096-env target | Not Tested | Existing quadratic contact-filter allocation remains a scaling issue. |
| Selective Reset and focused tests | Pass | 2/16 environments, 100 resets each, zero peer-state error; 48 offline plus 3 Isaac contract tests. |

The saved `params/agent.yaml`, `params/env.yaml`, source snapshot and
training manifest record the exact run configuration. With the official
assets at the stated root, rerun the experiment and evaluation using:

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/rsl_rl/train.py \
  --task DexterousHands-WujiHand2-Bimanual-GraspHold-Direct-v0 \
  --num_envs 1024 --max_iterations 25 --stable_reset_probability 0 \
  --thread_backend mimic --right_cap_contact_gate manifold_sidewall \
  --right_contact_balance_weight 0.2 \
  --right_cap_proximity_mode sidewall_tripod --seed 41 \
  --run_name mimic_strict_hold_seed41_25iter --headless
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task grasp_hold --policy learned --source pregrasp \
  --thread-backend mimic --right-cap-contact-gate manifold_sidewall \
  --right-contact-balance-weight 0.2 \
  --right-cap-proximity-mode sidewall_tripod --held-out \
  --num-envs 64 --episodes 200 --seed 1001 --headless \
  --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_14-03-44_mimic_strict_hold_seed41_25iter/model_24.pt
```

Remove `--held-out` to reproduce the training-family evaluation. The
available checkpoint corresponds to the saved source snapshot; the final
working-tree version has the hold-anchor refinement documented above.
