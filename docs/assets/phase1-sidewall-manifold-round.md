# Phase 1 cap-sidewall contact and controlled PPO rounds — 2026-11-01

> **Correction after live-state capture:** The one training-family episode
> previously reported as a GraspHold success was a false *stable hold* under
> the old motion criterion. Its bottle had moved 28.9 mm and was rotating at
> about 2.63 rad/s; the cap spin was 4.25 rad/s. It had correct contact
> topology, but the object was not settled. With motion-qualified hold
> acceptance, the same checkpoint scored **0/200**. See
> [the strict-hold audit](phase1-strict-hold-audit.md).

Branch: `codex/bimanual-bottlecap-rl-v0`; user-selected baseline: `8987c9c`.
The pre-round code was `104b2fe`. These are engineering-bottle experiments
with WujiHand2 Beta 1 candidate USD, Isaac Sim 6.0.1, Isaac Lab 3.0 and an
RTX 5090. The bottle geometry and 0.020 N m example breakaway torque are not
measurements of the user's hardware. The unconfirmed right `_start` template
is not used.

## Contact topology diagnosis

The old right-cap body force matrix counts any finger-cap collision. A
one-environment independent PhysX point view showed that bank row 3's right
index collided at the **bottom rim** rather than the sidewall: over the last
0.5 s, body-level tripod contact was 25/26 steps, index sidewall contact was
0/26, and thumb/middle sidewall contact was 26/26 and 25/26. A fingertip-site
gate rejected this case, but a second controlled row-9 probe revealed that
sites cannot reliably classify collision points: without right-wrist descent,
the index site was inside the axial sidewall band while its true sidewall
contact was absent in the final 0.5 s; the middle site was above the band
while its collision body actually contacted the sidewall. Row 9's actual
thumb/index/middle sidewall contact occurred in 40/9/40 of 50 policy steps,
with all three together in only 4/50. This is a transient scripted contact,
not a stable Reset state or a learned policy.

The new optional PhysX manifold view filters the cap against right thumb,
index and middle bodies. A GPU batched reduction reads each contact point,
rotates it into the live cap frame (`xyzw`), and counts normal force only when
`|z_local| <= cap_height/2 - 1 mm` and the point lies within 4 mm of the
example cap radius. It selects only the diagonal environment/filter block,
so a force from another environment cannot be credited. The count/start
buffers are converted from the installed PhysX/Warp `uint32` CUDA format
before PyTorch indexing. A saturated global buffer or more than 16 points
for one pair invalidates that environment's right contact; incomplete data
cannot earn success. `body_only` and `site_sidewall_proxy` remain selectable
for historical comparison, while `manifold_sidewall` is the new default.

In the one-environment row-9 replay, the batched classifier and independent
point-by-point PhysX loop agreed on **every one of 50 steps** for each of the
three digits: thumb 40, index 9, middle 40; no overflow. The body-only tripod
flag was true in 37/50 steps, whereas the exact sidewall tripod was true in
4/50. The exact mode also completed 1-, 16- and 1024-environment smoke tests
with finite 277-D Actor and 421-D Critic observations. The 1024-environment,
three-step smoke ran at 7,382 environment steps/s; it was a zero-action
throughput check, not a grasp test. During the 1024-environment PPO run, last
sampling throughput was 6,190 environment steps/s and observed GPU memory
was about 9.8 GiB. The last sampled invalid-point fraction was 0.0001; those
steps received no right-contact credit. Scaling the existing quadratic
sensor/filter matrix to 4096 environments remains untested and was already
projected to exceed this workstation's memory budget.

## PPO and deterministic evaluation

All four rows below use GraspHold, seed 41, 1024 environments, 500 Hz physics,
50 Hz policy, hard-pitch experimental Mimic thread, and natural geometric
pregrasp Reset only. The held-out set contains eight reserved rows from the
same pregrasp family, repeated to 200 episodes. It is not cross-family
generalization. Training-family evaluation contains 200 episodes drawn from
the other 24 rows. All evaluations use deterministic policy actions and seed
1001. The reward terms changed between rounds, so mean training rewards are
not directly comparable.

| Controlled change | Updates / PPO transitions | Last sampled right contact / bilateral grasp | Held-out GraspHold | Training-family GraspHold |
|---|---:|---:|---:|---:|
| Site proxy gate, original balanced contact reward | 25 / 819,200 | 0.11% / 0.03% | 0/200 | 0/200 |
| Partial correct-site contact reward made denser (`balanced_weight=0.2`) | 50 / 1,638,400 | 1.60% / 1.02% | 0/200 | 0/200 |
| Also target only tripod sites near the interior cap sidewall | 50 / 1,638,400 | 2.13% / 1.63% | 0/200 | 1/200 under the old motion gate; later invalidated |
| Replace site proxy with actual PhysX sidewall points | 50 / 1,638,400 | 2.10% / 1.37% | **0/200** | **0/200** |

The third round's **one** training-family episode also passed the *old*
GraspHold gate when re-evaluated with exact manifold contact, twice at the
same seed. It came from pregrasp bank index 5 and had real thumb/index/middle
sidewall contact, but the subsequent pre-Reset state capture revealed that
the bottle and cap were moving rapidly. The old criterion only checked
contact, a broad workspace and bottle linear speed below 0.2 m/s. It was
therefore a **false stable-hold success**, not a validated grasp or bank
entry. Its transient cap angle is not a CapTurn30 result either. The new
motion-qualified gate scores this checkpoint 0/200; no stable Mimic bank was
created.

The final exact-contact PPO run logged KL 0.0157, clip fraction 0.2349,
explained variance 0.8323, grouped action standard deviations
0.2498–0.2526, action clipping 3.10%, joint-target clipping 16.66%, and
no NaN/Inf. Its sampled mean reward was −8.83 and last right body-level
tripod rate was 5.08%, compared with 2.10% true sidewall tripod. On the
held-out set, true right tripod and bilateral grasp occurred in 0% of policy
steps; 100% of episodes failed. These measurements support a contact-topology
and reset-distribution bottleneck, not a PPO optimizer failure.

The recorded cumulative **PPO** interaction after this round is
23,171,072 environment transitions out of the 25-million research budget.
Evaluation and short physical probes add smaller amounts. Further identical
PPO extension is paused so the remaining budget can be used only after a
Mimic-compatible, dynamically replayable stable grasp state exists.

## Reproduction and retained artifacts

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_rl.py \
  --num-envs 1024 --steps 3 --thread-backend mimic \
  --right-cap-contact-gate manifold_sidewall --headless
scripts/run_isaaclab.sh scripts/environments/probe_pregrasp_contact_jacobian.py \
  --sample-index 9 --steps 50 --maintain-contact --contact-points \
  --right-cap-contact-gate manifold_sidewall --right-wrist-down-steps 0 \
  --right-index-mcp-flex --right-index-abd-target-min=-0.2 \
  --right-middle-mcp-abd --right-middle-abd-target-min=-0.2 \
  --right-middle-pip-target-max 1.6 --headless
scripts/run_isaaclab.sh scripts/rsl_rl/train.py \
  --task DexterousHands-WujiHand2-Bimanual-GraspHold-Direct-v0 \
  --num_envs 1024 --max_iterations 50 --stable_reset_probability 0 \
  --thread_backend mimic --right_cap_contact_gate manifold_sidewall \
  --right_contact_balance_weight 0.2 \
  --right_cap_proximity_mode sidewall_tripod --seed 41 \
  --run_name mimic_manifold_tripod_seed41_50iter --headless
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task grasp_hold --policy learned --source pregrasp \
  --thread-backend mimic --right-cap-contact-gate manifold_sidewall \
  --right-contact-balance-weight 0.2 \
  --right-cap-proximity-mode sidewall_tripod --held-out \
  --num-envs 64 --episodes 200 --seed 1001 --headless \
  --checkpoint logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_13-33-31_mimic_manifold_tripod_seed41_50iter/model_49.pt
```

The final checkpoint is at the path above. The earlier loose-gate
false-positive checkpoint is
`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_13-13-50_mimic_sidewall_tripod_approach_seed41_50iter/model_49.pt`.
Large checkpoints, training stdout, per-episode JSON and contact traces are
ignored under `logs/` and `artifacts/phase1/`; the source code and this
summary are the Git deliverable. The exact success replay rows are in
`artifacts/phase1/mimic_site_checkpoint_exactgate_trainfamily_replay_200_rows.json`
and its pre-Reset physical state is in
`artifacts/phase1/mimic_row5_success_pre_reset_candidate.json`.

Focused contact/reward tests passed 19/19; all offline bimanual tests passed
84/84 after the exact-manifold addition. The full repository test run after
that addition returned **166 passed, 1 skipped, 2 failed**. Both failures
are in concurrently user-edited Allegro/MANO test areas:
`test_allegro_hand_track_config.py` and untracked
`test_checkpoint1_helpers.py`. They were not modified or included in this
Phase 1 commit. The local `scripts/run_isaaclab.sh` can return zero after a
Python exception, so Isaac smoke and training logs were checked for explicit
completion markers and tracebacks rather than using shell exit alone.

## Gate status and next experiment

| Gate | Status | Evidence / limit |
|---|---|---|
| 1024-environment DirectRLEnv, PPO, checkpoint | Pass for Mimic pregrasp mode | 50 updates, 1.64M transitions and saved model; no nonfinite state. |
| Correct right cap sidewall classification | Pass at 1 env; partial at 1024 | Exact one-env classifier agreed with independent points for 50 steps; 1024 smoke and PPO ran with 0.01% invalid points, but a full 1024-env per-pair contact audit was not performed. |
| Held-out natural-pregrasp GraspHold ≥60% | Fail | 0/200 for all four rounds. |
| CapTurn30 stable-start ≥30% | Not Tested on Mimic | Existing four legacy stable states violate the hard-pitch Mimic Reset geometry by about 1.2–1.26 mm; no compatible stable bank. |
| CapTurn30 natural-pregrasp ≥10% | Not Trained in this round | True bilateral sidewall grip is too sparse for a controlled turn curriculum. |
| 4096-environment target | Not Tested | Existing quadratic filter allocation is the measured scaling blocker. |
| Old Wuji single-hand tasks and repository suite | Partial | Old-task tests passed within the full run; two unrelated user-area tests failed as listed above. |

The next physical experiment is a contact-feedback closing fixture that
keeps right index and middle on the interior sidewall while reducing cap
spin and bottle drift. Capture an actually settled state, then independently
Reset and replay a gravity-on hold. Only physically replayed states may
enter a Mimic stable bank and justify CapTurn PPO. No right `_start`
curriculum, new RL algorithm, scripted cap torque or direct object-pose
writing was added in this round.
