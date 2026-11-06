# M5 4096-environment net-speed hold pilot

This is a bounded Strict-Pure Model-Based RL pilot of the IK-free dual-TCP SE(3)
Reach & Hold task. It is **not** a successful task acceptance or a trained
multi-task policy. The model uses the existing SuperMimic Core, Newton MJWarp
physical collection, masked WM8, gated imagination, Actor BPTT, and TD(λ).
No IK, target joint supervision, teacher policy, Wave distillation, PPO, or
asset/controller changes were used.

## Hold-speed contract and safety audit

The original task version used the final 500 Hz Newton joint-velocity sample
for the `≤0.15 rad/s` hold condition. A zero-action physical audit found
approximately ±0.5 rad/s oscillation of both Joint7 actuators with a 98.7%
adjacent-substep sign reversal. The joint moves approximately 0.0013 rad/s
net over the policy interval, and the TCP jitter is below about 0.6 mm. The
instantaneous readout is a real substep motion, not a mislabeled Tensor.
The original hold criterion had no valid transitions in the 4096-environment
training Replay, even near the goal. Audits are saved in
`m5_velocity_audit_zero_9511/` and `m5_velocity_audit_zero_16env_9512/`.

Following the user's explicit choice, `policy_step_displacement_v2` defines
hold speed as `abs(q[t+1] - q[t]) / 0.02` for each of the 14 arm joints. It is
used by the hold reward, low-speed reward, velocity penalty, success mask, and
episode metrics. Physical state, Actor observation, World Model target, and
hard safety termination still use the real Newton joint velocity. Independent
evaluation also records the maximum absolute velocity over all ten 500 Hz
substeps. The controller, Kp/Kd, command interpolation, effort limits, and
robot asset are unchanged. The original `instantaneous_qd_v1` task contract
remains the default for existing checkpoints; v2 is a distinct, strictly
checked Task and physics contract. An old v1 checkpoint completed a new
single-environment, 200-step independent Newton evaluation with its original
metric in `m5_v1_checkpoint_compat_eval_52/`.

## Training and independent physical results

Each seed used 4096 Newton environments, 40 iterations, 655,360 real physical
transitions, 640 World Model updates, 34 Actor updates, 272 Critic updates,
zero teacher transitions, and zero nonfinite updates. The training-only goal
sampler mixed original Near goals with smaller FK-generated near-hold goals.
The independent evaluation used the *same original Near goal set* with seed
9501, 64 environments, 200 policy steps each, no exploration noise, and a
fresh Newton process. Every evaluation completed all 64 episodes without a
hard termination, contact event, or joint violation. The 500 Hz velocity peak
remained below the existing 5 rad/s hard safety threshold.

| Actor | Left/right mean TCP position | Left/right mean SO(3) | Full success | Ever reached both 6D targets | Longest valid hold | 500 Hz velocity peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed 55, random initialization | 19.91 / 20.08 mm | 3.81° / 3.91° | 0/64 | 0/64 | 0 | 0.620 rad/s |
| Seed 55, 40 iterations | 10.74 / 9.74 mm | 2.61° / 2.48° | 0/64 | 3/64 | 5 steps | 0.617 rad/s |
| Seed 56, 40 iterations | 9.14 / 10.02 mm | 2.43° / 2.40° | 1/64 | 2/64 | 25 steps | 0.621 rad/s |
| Seed 57, 40 iterations | 10.83 / 10.72 mm | 2.46° / 2.56° | 1/64 | 1/64 | 39 steps | 0.616 rad/s |

The two successful episodes are isolated. Seed 56's successful episode first
reached both SE(3) thresholds at step 60, first completed 25 valid hold steps
at step 160, and had no contact. Its final right TCP position error was
5.39 mm, outside the 5 mm target. Seed 57's successful episode first reached
both thresholds at step 38, first completed the hold at step 62, and had a
39-step maximum hold; its final left/right position errors were 11.21/6.29
mm. The task's success rule requires a complete safe episode, arrival by
step 125, and **at least one** continuous 25-step valid hold. It does not
require the final control step to be in tolerance. These results do not show
stable long-term holding or convergence. Medium and Wider Feasible curricula
were therefore not started.

All three training runs admitted only one-step imagination. Noisy held-out
normalized delta RMSE at horizons 1/8/32 was respectively 0.645/1.094/1.832
(seed 55), 0.636/1.123/1.758 (seed 56), and 0.631/1.104/1.869 (seed 57).
The trust gate correctly withheld 8/32-step Actor updates. For seed 55, an
independent Newton Actor trace gave TCP position P95 prediction error in mm:

| Comparison | 1 step L/R | 8 steps L/R | 32 steps L/R |
| --- | ---: | ---: | ---: |
| Action-conditioned open loop | 0.68 / 0.85 | 1.15 / 0.97 | 3.64 / 2.96 |
| Actor closed loop in Model vs Newton | 1.11 / 0.95 | 2.93 / 3.63 | 7.88 / 7.10 |

The seed 55 closed-loop 32-step TCP SO(3) P95 was 1.24°/1.47°. Its model
validation and real trace are in `m5_4096_net_hold_eval_40_55/`. The model's
closed-loop error near a 5 mm success tolerance is a concrete reason to keep
the 32-step gate closed. Actor update count alone is not convergence evidence.

The three complete 4096-environment loops took 46.5, 41.4, and 43.0 seconds.
Peak **PyTorch allocated** VRAM was 1.88 GiB per run; that number excludes
Isaac/Newton allocations and is not a total-device peak measurement.

## Verification and artifacts

- Full SuperMimic CPU/Torch suite: **199 passed** after the task change.
- New failing-first tests cover v1/v2 speed semantics, reward, 25-step episode
  accounting, and substep peak recording.
- 64-environment, 20-step Newton v2 Smoke: 1,280 transitions, zero hard
  terminations, zero command mismatch, and Environment/Task reward parity.
- Original M4.1 Wave regression on this branch previously completed
  **1356/1356** Newton steps with zero contacts and illegal joint states;
  source reports are `m5_wave_legacy/evaluation.json` and
  `m5_v3_wave_legacy/evaluation.json`. Its selected checkpoint SHA256 remains
  `fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f`.
  The Wave reference SHA256 remains
  `b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f`.
- New task code changes neither the legacy Actor nor the Marvin controller.
- The v2 config SHA256 is
  `61083000f330de552debca325a9e1cd76c68dd9f663ec330b3b9df769ed98082`.
- Seed 55/56/57 checkpoint SHA256 values are respectively
  `8b20c79484e7c3d8cede37afd82f9aee1365536f25d1399f8d412bad4e0314c8`,
  `085cef47c156dab190c16c42cdbfc4b7297a092226885f436a3cccc105df48e4`,
  and `67be02369c60623b2e5ec34e227dfb7f35f8a41904da04112dd35036d0504ec0`.
- Seed 55's checkpoint records preceding commit `75051e8` because the v2
  task changes were not yet committed when that pilot ran. The full readable
  v2 semantic manifest and its SHA256 are embedded in that checkpoint.
  Seeds 56 and 57 record committed source `1c1c0c3`.

The experiment root is
`/home/ubuntu/dexgym_data/marvin_supermimic_m5_se3_reach_hold/`. Each
`m5_4096_net_hold_pilot_{55,56,57}/` directory contains a model, full
iteration log, and report. Each `m5_4096_net_hold_eval_40_{55,56,57}/`
directory contains a separate 64-episode physics trace and report.
`m5_4096_net_hold_init_55/` and `m5_4096_net_hold_eval_init_55/` contain
the paired untrained Actor baseline. None of the earlier M4/M5 models,
references, or assets were overwritten.

## Reproduction commands

Run from this M5 worktree. Use **new output paths** on a rerun.

```bash
export PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source
export ISAAC_PYTHON=/home/ubuntu/isaacsim-6.0.1/bin/python
export M5_DATA=/home/ubuntu/dexgym_data/marvin_supermimic_m5_se3_reach_hold

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py smoke_se3_m5 --num-envs 64 --steps 20 --seed 55 --level near_hold_mix --hold-speed-mode policy_step_displacement_v2 --output "$M5_DATA/new_v2_smoke.json"
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_se3_m5 --config configs/supermimic/m5_se3_near_hold_mix_4096_v2.json --iterations 40 --seed 56 --output "$M5_DATA/new_v2_seed56"
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_se3_m5 --checkpoint "$M5_DATA/new_v2_seed56/model.pt" --num-envs 64 --seed 9501 --level near --output "$M5_DATA/new_v2_seed56_eval"
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py validate_se3_model_m5 --checkpoint "$M5_DATA/new_v2_seed56/model.pt" --physics-trace "$M5_DATA/new_v2_seed56_eval/physics_trace.pt" --closed-loop --output "$M5_DATA/new_v2_seed56_eval/model_validation.json"
$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py diagnose_se3_velocity_m5 --num-envs 16 --control-steps 30 --seed 9512 --output "$M5_DATA/new_zero_action_velocity_audit"
```

The task acceptance remains **open**: aggregate strict success is only 2/192
Near episodes, 8/32-step on-policy trust is not established, and no Medium,
Wider Feasible, contact, or real-robot task success was tested.
