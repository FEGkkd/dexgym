# Arm-only continuous dual-TCP SE(3) following

This task uses the existing SuperMimic Model-Based RL Core to follow a moving
dual-TCP pose target in real Newton MJWarp physics. It is a separate task from
static SE(3) Reach & Hold and from six-link Wave tracking. The pinned input is
the **procedural**, 50 Hz `procedural_bimanual_reach_v1.npz`, SHA256
`ca4741f01734389dd62f9e64c66cdc9c237c90da6370cf76b13bcda89e143af2`.
Training samples the 201-frame forward trajectory and its time reverse. The
TCP excursion from frame 0 is at most 57/81 mm and 9.1/12.7 degrees for the
left/right arms. A joint witness is used only for safe Newton reset and for
an explicitly labelled reference-servo feasibility diagnostic. The policy,
world model, critic, reward and success evaluation never read it.

## Arm-only contract

The complete 54-DoF Marvin/Wuji robot remains in Newton, with all 40 finger
joint *commands* set to Home every physics substep. The physical finger joints
can still move slightly under the servo and load. They are absent from the RL
state, action, observation, goal, critic and replay. A 16-environment smoke
measured zero hand-command deviation from Home and up to 0.090 rad physical
finger displacement; this is why the report distinguishes commanded and
physical Home.

The 16-environment, 40-step Newton integration smoke collected 640 real
transitions with zero unexpected safety termination and zero maximum
executed-command mismatch between the Robot Adapter and Newton. The
`CoreNewtonCollector` checked Task MDP reward, hard termination and arrival
time against the environment on each transition. A separate evaluation-mode
reset smoke confirmed that a requested 0.01 rad arm perturbation actually
changed the initial command by up to 0.00994 rad; a failing physical check
exposed and preceded the one-line reset fix.

| Interface | Definition |
| --- | --- |
| Physical/world-model state | 42 = arm `q14`, `qd14`, `qcmd14` |
| Actor action | 14 raw arm actions, mapped by `arm_joint_delta_v1` to bounded command increments |
| Learned world-model delta | 28 = arm `Δq14`, `Δqd14`; command update is analytic |
| Actor observation | 68 = arm `q14`, `qd14`, `qcmd14`, previous action14, dual TCP position error6, SO(3) log error6 |
| Actor goal | 10 future frames × 18 = two TCP positions and rotation 6D |
| Critic features | 222 = physical arm state42 plus ten future TCP goals180 |

At control time `t`, the actor sees targets `t+1…t+10`; Newton executes
`a_t`, and reward and termination evaluate the physical arrival state against
target `t+1`. A 200-step episode ends at frame 200. Safety termination covers
contacts, illegal arm states and dangerous speeds; a large tracking error is
not a safety failure. The task's full-episode success threshold requires each
arm's position mean ≤5 mm and P95 ≤10 mm, SO(3) mean ≤2° and P95 ≤4°,
complete clip playback and no safety termination. This success definition is
for **continuous tracking**, not the 25-step hold condition of the separate
static Reach & Hold task.

The training config is
[`continuous_se3_follow_4096_full32_v1.json`](../../configs/supermimic/continuous_se3_follow_4096_full32_v1.json).
It fixes WM autoregression at 8 transitions and trusted actor imagination at
32 steps. Each iteration collects 4×4,096 true Newton transitions, performs
32 World Model updates, one BPTT Actor update once trusted, and 5 epochs × 4
Critic minibatches. Training is strict-pure: no teacher-collected replay,
no IK, no joint target loss, no imitation and no PPO. Disjoint environment
lanes provide training data and deterministic/noisy model validation. The
Replay holds 256 transitions per environment; model and imagination windows
cannot cross an episode reset. The checkpoint includes Replay for explicit
Replay resume; model-only export uses fresh-physics resume semantics.

## Measured physical pilot

The zero-action baseline on 16 Newton environments had 0/16 strict
tracking successes; left/right mean position error was 26.96/26.50 mm and
mean SO(3) error was 4.21°/4.19°. The reference-servo diagnostic completed
16/16 episodes with 0.56/0.57 mm mean position error. This only establishes
that the pinned trajectory is physically feasible; it did **not** supply
training transitions or Actor supervision.

Seed 72's 400-iteration strict-pure checkpoint saw 6,553,600 real physics
transitions, 12,800 WM8 updates, 354 trusted BPTT32 Actor updates and 7,080
Critic minibatch updates. All three nonfinite-update counts were zero.
The held-out noisy model's normalized delta RMSE at horizons 1/8/32 was
0.402/0.622/0.811 at iteration 200, versus 1.062/3.404/9.758 for its
zero-delta baseline. These are action-conditioned model predictions, not
closed-loop Newton policy errors.

| Independent 200-step Newton evaluation | Episode success | Position mean L/R | SO(3) mean L/R | Safety failures |
| --- | --- | --- | --- | --- |
| Zero action, seed 9101, 16 env | 0/16 | 26.96 / 26.50 mm | 4.21° / 4.19° | 0 |
| Actor iter 100, seed 9101, 64 env | 0/64 | 10.81 / 6.75 mm | 3.51° / 2.58° | 0 |
| Actor iter 200, seed 9101, 64 env | 0/64 | 10.23 / 5.90 mm | 2.60° / 1.27° | 0 |
| Actor iter 400, seed 9101, 64 env | 64/64 | 2.15 / 1.48 mm | 0.49° / 0.80° | 0 |
| Actor iter 400, seed 9102, 0.01 rad reset perturb, 64 env | 64/64 | 2.05 / 1.65 mm | 0.58° / 0.77° | 0 |
| Actor iter 600, seed 9102, 0.01 rad reset perturb, 64 env | 64/64 | 1.09 / 1.11 mm | 0.27° / 0.28° | 0 |
| Actor iter 800, seed 9102, 0.01 rad reset perturb, 64 env | 64/64 | 0.91 / 0.90 mm | 0.23° / 0.26° | 0 |
| Actor iter 800, seed 9103, 0.02 rad reset perturb, 64 env | 64/64 | 1.16 / 1.11 mm | 0.30° / 0.30° | 0 |
| Actor iter 1000, seed 9102, 0.01 rad reset perturb, 64 env | 64/64 | 0.95 / 0.96 mm | 0.26° / 0.33° | 0 |

The planned maximum of 1,000 iterations completed in 1,888 s across ten
resumable segments. It collected 16,384,000 real transitions, performed
32,000 WM8 updates, 954 trusted 32-step Actor updates and 19,080 Critic
minibatch updates. Every accepted Actor update used horizon 32. The 21
recorded trust failures occurred before trust was established; none of the
three optimizer nonfinite counters increased. Peak PyTorch allocation was
3.74 GB; an `nvidia-smi` sample during the full loop showed 19.8 GB total
device allocation on the 32.6 GB RTX 5090. The 800-iteration checkpoint is
selected by independent physical mean position/orientation error; the final
1000-iteration weights and full Replay are retained for comparison.

At iteration 1000, held-out noisy actor data gave normalized 1/8/32-step
World Model delta RMSE of 0.354/0.571/0.702, versus a zero-delta baseline
of 1.026/3.326/9.499. On deterministic actor data, 32-step action-conditioned
prediction had TCP position P95 max 1.69 mm, SO(3) P95 max 0.338° and joint
velocity RMSE 0.034 rad/s. These are model prediction diagnostics, not
closed-loop actor-in-model versus actor-in-Newton error measurements.

All 64-environment evaluations contain repeated instances of **two** pinned
trajectory variants, 32 per variant, with independent physical environments.
Training samples reference start frames 0–100, while these complete-clip
evaluations start at frame 0. Evaluation seeds vary the physical reset offset,
not the target trajectory.
The perturbed run uses a different seed and a real initial arm joint offset;
the reset diagnostic measured up to 0.00994 rad offset against its reference
start. These results demonstrate learned physical control on the chosen task.
They do not establish accuracy on unseen trajectories or different payloads.
The training checkpoints' source commit is `72e96d3`; their independent
perturbed evaluation uses a subsequent fix that makes the configured reset
perturbation actually take effect. The policy weights and servo did not change.
The exported `fresh_physics` checkpoint passed strict contract loading in a
new Newton process. On the same seed-9102/0.01-rad protocol it completed
64/64 episodes with 0.91/0.91 mm mean position error and zero safety failures.

The original M4.1 Wave Actor was also replayed independently from its
protected source checkpoint and reference: 1,356/1,356 Newton control steps,
zero hard termination, contact control/physics steps and illegal joint
states. Its left/right mean TCP position error was 2.28/5.22 mm and mean
SO(3) error was 0.31°/1.14°. The original Wave checkpoint SHA256 remained
`fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f`.
The full regression report is an archived physics artifact, not a training
metric for the new task.

## Reproduce

Run from the repository root on the verified Isaac Sim 6/Newton host. The
runtime overlay is an existing simulator dependency, not a training asset.
Training and evaluation outputs below go to `/tmp`; the explicit export
command writes the selected model into Git LFS. Pull Git LFS assets before
running.

```bash
git lfs pull
export PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source
ISAAC_PYTHON=/home/ubuntu/isaacsim-6.0.1/bin/python

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py smoke_continuous_se3 \
  --num-envs 16 --steps 40 --output /tmp/continuous_se3_smoke.json

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_continuous_se3 \
  --config configs/supermimic/continuous_se3_follow_4096_full32_v1.json \
  --output /tmp/continuous_se3_follow_seed72_iter100

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py train_continuous_se3 \
  --resume /tmp/continuous_se3_follow_seed72_iter100/model.pt \
  --output /tmp/continuous_se3_follow_seed72_iter200

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_continuous_se3 \
  --checkpoint /tmp/continuous_se3_follow_seed72_iter800/model.pt \
  --num-envs 64 --seed 9102 --reset-perturb-rad 0.01 \
  --output /tmp/continuous_se3_follow_seed72_eval800_perturb

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py export_core_checkpoint \
  --checkpoint /tmp/continuous_se3_follow_seed72_iter800/model.pt \
  --output models/supermimic/continuous_se3_follow_seed72_iter800.pt

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate_continuous_se3 \
  --checkpoint models/supermimic/continuous_se3_follow_seed72_iter800.pt \
  --num-envs 64 --seed 9102 --reset-perturb-rad 0.01 \
  --output /tmp/continuous_se3_follow_export_eval800_perturb

$ISAAC_PYTHON scripts/supermimic/isaac_runtime.py evaluate \
  --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt \
  --reference /home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz \
  --output /tmp/continuous_se3_wave_legacy_regression

$ISAAC_PYTHON -c "import sys,numpy; sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages'); import mujoco,pytest; sys.exit(pytest.main(['-q','tests/newton','tests/assets','tests/supermimic']))"
```

The test command passed **221 tests with 3 skips**. Repeat the second training
command with each previous `model.pt` as `--resume` to complete each 100-step
segment through iteration 1000. It is a Replay resume with a fresh Newton
process and explicit trust revalidation, not a bitwise restore of simulator
state.

The full Replay checkpoint at selected iteration 800 is about 3.3 GB and is
kept as an external experiment artifact. An exported `fresh_physics` checkpoint is
small enough for Git LFS, but it does not restore the old physical environment
or Replay and must not be presented as a bitwise training resume.
The selected export is
[`continuous_se3_follow_seed72_iter800.pt`](../../models/supermimic/continuous_se3_follow_seed72_iter800.pt),
SHA256 `cd9eb16b1b1ce36ad32be2728865a2eb478ec0c507b195bb62ad10ad7849c689`.
The selected and final full Replay checkpoints, all per-iteration logs and
independent physics traces are archived without overwriting older experiments
under `/home/ubuntu/dexgym_data/marvin_supermimic_continuous_se3_follow_seed72_20261107/`.

This pilot used one Actor initialization seed and two target trajectories
(one path and its reversal). Independent physical seeds and start perturbations
show robustness on that restricted task, but no unseen trajectory, payload,
contact manipulation or real robot deployment has been validated.
