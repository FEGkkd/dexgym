# Marvin Wave M4.1: alternating world-model training

This branch adds an alternating Newton → eight-step world-model → 32-step Actor
BPTT → TD(λ) Critic loop for the pinned bimanual Wave reference. The source
reference is `cmu_141_16_wave_bimanual.npz`, SHA256
`b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f`.
It retains the 54DoF Newton articulation and controls only the 14 arm joints.

## Algorithm and engineering choices

The loop follows [SuperMimic](https://arxiv.org/pdf/2610.28480), section 3.1
and Appendix D: each iteration collects real transitions, trains the model on
continuous eight-step physical windows, then backpropagates through a frozen
model over 32 imagined steps and trains the Critic on TD(λ) targets. A learned
termination head uses hard Newton safety terminations for BCE training. Actor
returns use survival `c[t+1] = c[t] * (1 - p_terminate[t+1])`; Critic targets use
the same probability. World-model parameters stay frozen only during Actor
backpropagation. The Newton collector stores actual pre-reset next states,
including terminal transitions, and never joins an episode across reset.

The fixed-base Marvin task omits the paper's floating-root reward terms. The
`paper_multilink` profile implements its four six-link exponential tracking
terms and action/limit penalties. `marvin_tcp_priority` retains the M4 reward
for a controlled algorithm comparison. Goal-latent tanh and observation noise
are explicit ablations; neither is silently imposed on old E11 checkpoints.
The optional behavior prior anchors Actor updates to the initialization
checkpoint and is reported separately from the Pure run. The prior contributes
a penalty in the imagined Actor loss; the physical Actor still receives only
proprioceptive history and future link goals.

Newton contact counts now use `Contact.worldid` per environment and include the
last physics substep before Isaac Lab calls `_get_dones()`. Safety terminations
are per environment. The authoritative TCP remains the pinned M3 site-based
reference and differentiable FK. Hand joints remain fixed at Home, with hand
contact disabled in the Newton asset.

## Reproducible commands

Use the process-local Isaac Python path, without editing the shared venv:

```bash
export PYTHONPATH="$PWD/source:/home/ubuntu/miniconda3/envs/manus_wuji_real/lib/python3.12/site-packages:/home/ubuntu/miniconda3/envs/teleop/lib/python3.12/site-packages:/tmp/marvin_wave_python_override"
/home/ubuntu/dexgym/.venv/bin/python scripts/supermimic/train_aligned.py \
  --num-envs 1024 --iterations 1000 --world-hidden 1024 \
  --world-batch 256 --imag-batch 32 --replay-steps 500 \
  --min-world-updates 20 --trust-check-interval 100 --action-noise 0.005 \
  --resume /home/ubuntu/dexgym_data/marvin_supermimic_wave_m4/e11_4096env_total4000/best.pt \
  --no-goal-latent-tanh --reward-profile marvin_tcp_priority \
  --observation-position-noise-rad 0 --observation-velocity-noise-rad-s 0 \
  --actor-lr 1e-7 --behavior-prior-weight 100 \
  --prior-checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m4/e11_4096env_total4000/best.pt \
  --checkpoint-interval 100 --final-eval \
  --output /home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/aligned_1024_1000
```

`latest.pt` is written atomically at each checkpoint interval. Replay is not
stored; `--resume` reconstructs it from new physical transitions and waits for
episode-safe windows and a held-out world-model trust check before Actor
updates. `candidate.pt` is a trial policy. `best.pt` is only written after a
complete Newton evaluation passes the recorded thresholds; every candidate
should also be independently evaluated with `evaluate_wave.py`.

## Early measured results

The 32-environment Pure smoke collected 6,400 physical transitions in 200
iterations, completed 84 world-model updates, and made **zero** Actor/Critic
updates. Randomly initialized actions repeatedly hit the task's 120 mm TCP
safety termination, leaving no held-out 32-step physical windows. This run is
an algorithm gate result, not a learned policy.

An unconstrained E11-initialized 128-environment smoke collected 15,360 real
transitions and completed 81 world-model and 72 Actor/Critic updates. Its
held-out TCP prediction error was 6.05 mm at eight steps and 7.96 mm at 32
steps, but independent Newton evaluation stopped at control step 24. This is
model-exploitation evidence; it must not be reported as tracking success.
The same evaluator reproduced the pinned E11 baseline for all 1,356 control
steps with zero Newton contacts and zero illegal joint states. The left/right
TCP mean position errors were 3.20/5.87 mm and mean orientation errors were
0.42°/1.28°.

A conservative 128-environment smoke used the E11 Actor as its fixed behavior
prior, Actor learning rate `1e-7`, and 120 alternating iterations. It made 81
world-model and 72 Actor/Critic updates, then completed all 1,356 Newton
control steps with zero contacts and illegal joint states. Left/right TCP mean
position errors were 3.65/5.77 mm; mean orientation errors were 0.48°/1.14°.
This small run establishes stability, not improvement over the E11 baseline.

The 1024-environment 1000-iteration run collected 1,024,000 Newton
transitions and made 961 world-model and 942 Actor/Critic updates in 916 seconds.
Its held-out world-model TCP position prediction errors were 3.40 mm at eight
steps and 4.17 mm at 32 steps. The full Newton evaluation completed all 1,356
control steps with zero contacts and illegal joint states. Left/right TCP mean
position errors were 2.46/4.42 mm and mean orientation errors were
0.34°/0.92°. This improves over E11 in the same evaluator. Its validated
checkpoint is `aligned_1024_1000/best.pt` in the external M4.1 data directory.

Provenance correction: this process launched with source commit `b310d23`.
The run's JSON `git_commit` field was read at shutdown, after monitoring code
was committed during training, so it reports the later repository HEAD. The
trainer now captures the source commit at startup for subsequent runs.

## Longer training comparison

All milestone policies used the same pinned reference, the same deterministic
zero-perturbation Newton evaluation, and completed 1,356 control steps with
zero Newton contacts and zero illegal joint states. Position values below are
millimeters; rotation values are SO(3) geodesic degrees.

| Checkpoint | Left position Mean/P95 | Right position Mean/P95 | Left rotation Mean/P95 | Right rotation Mean/P95 |
| --- | ---: | ---: | ---: | ---: |
| E11 | 3.20 / 13.86 | 5.87 / 18.00 | 0.42 / 1.65 | 1.28 / 3.68 |
| 500 | 2.82 / 13.40 | 5.79 / 20.30 | 0.38 / 1.56 | 1.16 / 2.92 |
| 1000 | 2.46 / 9.18 | 4.42 / 12.26 | 0.34 / 1.08 | 0.92 / 2.38 |
| 1500 | 2.03 / 6.15 | 4.71 / 9.72 | 0.29 / 0.76 | 1.05 / 2.83 |
| 2000 | 2.31 / 7.61 | 4.87 / 11.39 | 0.32 / 0.94 | 1.06 / 3.09 |

For checkpoint selection, average the eight ratios formed by both arms'
position Mean/25, position P95/50, rotation Mean/8, and rotation P95/15.
Only full, contact-free, legal-joint evaluations are eligible. The resulting
scores are E11 0.1960, 500 0.1887, 1000 0.1365, 1500 0.1242, and 2000
0.1384; lower is better. The 1500-iteration checkpoint has the best combined
score. In particular, 2000 iterations did not improve both arms uniformly.

The selected policy is the independently replayed 1500-iteration checkpoint,
copied to `/home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt`
with SHA256 `fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f`.
The external `selection.json` records the eligibility and score of every
candidate. With fixed-seed initial joint perturbations of ±0.03 and ±0.05 rad,
the selected policy completed all 1,356 Newton control steps with zero contacts
and illegal joint states. At ±0.05 rad, left/right TCP mean position errors
were 3.78/6.85 mm and mean orientation errors were 0.49°/1.46°.
The independent physical state record is in
`aligned_4096_cumulative2000/eval_1500/newton_policy_states.npz`; charts are in
`selected_visuals/`.

The first 1000 iterations used 1024 Newton environments, 1,024,000 physical
transitions, and 961/942/942 World Model/Actor/Critic updates. The next 1000
iterations resumed that Actor in 4096 Newton environments, added 4,096,000
physical transitions, and made 1922/1904/1904 updates. Thus the complete
M4.1 sequence used 5,120,000 physical transitions and 2883/2846/2846
updates. The second stage took 1781 seconds and sampled peak GPU memory was
21.3 GB. Its held-out world-model TCP position prediction errors were
3.26 mm at eight steps and 3.04 mm at 32 steps. The final 2000-iteration
checkpoint passed the physical quality gates but ranked below 1500 by the
same selection score.
