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
  --world-batch 128 --imag-batch 64 --replay-steps 1000 \
  --resume /home/ubuntu/dexgym_data/marvin_supermimic_wave_m4/e11_4096env_total4000/best.pt \
  --no-goal-latent-tanh --reward-profile marvin_tcp_priority \
  --observation-position-noise-rad 0 --observation-velocity-noise-rad-s 0 \
  --actor-lr 1e-7 --behavior-prior-weight 100 \
  --prior-checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m4/e11_4096env_total4000/best.pt \
  --checkpoint-interval 200 --final-eval \
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

Longer conservative runs and their independent Newton results are recorded in
the external M4.1 experiment directories. Training length alone does not
certify a policy.
