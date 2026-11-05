# Marvin Bimanual SuperMimic Wave v0

## Pinned reference and method

The only motion is `cmu_141_16_wave_bimanual.npz` at
`/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/`.
Its SHA256 is `b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f`.
It contains 1357 frames at exactly 50 Hz and drives both arms. The source MJCF SHA256 is
`6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce`;
the Newton USD SHA256 is `8ed2cb9a14a813022149aaa8a101159af86dab676aec7dfdde4a00a8d9698fa6`.
`scripts/supermimic/preflight.py` checks these hashes, the 54DoF joint names, 50 Hz timeline,
six link and both TCP poses against MuJoCo FK, the M3 replay reports, and the 16-environment acceptance.
All training checkpoints include the reference hash and fail to load against another version.

The algorithm follows the [SuperMimic paper](https://arxiv.org/abs/2610.28480): collect real
simulator transitions, train a differentiable world model, then optimize an Actor through imagined
rollouts by BPTT. The user-pinned [TrackerLab community reproduction](https://github.com/ZouYan93/trackerlab-supermimic)
commit `606aae160b88635c97eefb05a3e4e4591afd7632` was fetched and verified;
its root license is MIT. It supplies a useful architectural comparison, including 2048-wide
world-model layers and a 2048→1024→512→256 goal Actor. The separate
[mjlab reproduction](https://github.com/YHFone/supermimic-mjlab) is Apache-2.0.
These are community implementations, not the paper's official code.

Marvin differs from the floating-base humanoid setting: it is fixed-base 14DoF control inside a
full 54DoF articulation; the 40 finger joints receive fixed Home commands and contribute their
mass and inertia. Finger state, contact, and action are absent from Actor, Critic, and world model.
The Newton asset disables finger contact; hand contact safety is therefore outside this result.
Newton MJWarp under Isaac Lab 3.0 provides the real transitions. MuJoCo supplies independent
FK, joint-limit, and source-geometry contact checks. No PPO, real arm, or modified MJCF is used.

## Implemented contracts

- `DexGym-Marvin-SuperMimic-Wave-Direct-v0`: 500 Hz physics, 50 Hz policy, ten physical
  substeps per action, M3 drive profile, 54DoF physics, 14D arm position commands. The command
  moves toward `q_origin + action_scale * tanh(action)`, is clamped to joint limits and
  0.04 rad per policy step, and is interpolated over the ten substeps. The same analytic map
  runs inside imagination.
- Actor input: ten 56D proprioceptive frames (14 arm positions, 14 velocities, 14 previous
  actions, 14 executed position commands) and ten future 90D goal frames. Each goal has six
  real Marvin links with position, rotation-6D, linear velocity, and angular velocity.
  The Actor does not receive current `q_ref(t)` as a command.
- World model: 42D `[q_arm, qd_arm, q_cmd]` state, three 2048-wide layers by default,
  running normalization, state increments, and exact analytic command-state update. It trains
  on eight continuous real transitions without crossing episode resets.
- Actor/Critic: 2048→1024→512→256→32 goal encoder; 592→2048→1024→512→256→14
  deterministic Actor; privileged 256→128 Critic. A frozen model produces 32-step imagined
  rollouts, the differentiable Marvin FK computes all six tracked links, and the Actor receives
  BPTT gradients. The Critic learns TD(λ) targets with γ=0.99 and λ=0.95.
- A versioned observation/goal/action contract is in `marvin_wave_io_v1.schema.json`.
  Reference windows hold the final pose and zero velocity beyond the last frame.
- Checkpoints store network, optimizer, normalization, configuration, and reference hash.
  Replay is deliberately excluded; `--resume` recollects real Newton transitions.

The pinned Newton PD replay is used only for safe Actor initialization and finger physical
reset states. The initialization uses privileged reference joints during training, but the
deployed Actor receives only the inputs above. The final candidate must improve through the
world-model BPTT stage and be judged by a separate complete Newton physical replay; supervised
initialization is never counted as a SuperMimic success by itself.

## Reproduced experiments and current result

External experiment root: `/home/ubuntu/dexgym_data/marvin_supermimic_wave_m4/`.
The stable process-local Python path used for Isaac Lab 3.0 is recorded in the shell commands
that created the runs. The shared virtual environment and robot asset were not rewritten.

| Run | Real transitions | WM / Actor / Critic updates | WM TCP 8 / 32 steps, held-out env | Newton policy result |
| --- | ---: | ---: | ---: | --- |
| E1 smoke | 2,048 | 10 / 0 / 0 | 201 / 167 mm | no Actor |
| E2 safe teacher | 25,600 | 100 / 0 / 0 | 2.99 / 13.19 mm | no Actor |
| E3 first BPTT | 25,600 | 100 / 5 / 5 | 3.93 / 9.99 mm | full clip, but severe right-arm divergence |
| E3 phase-balanced warmup | 40,960 | 100 / 10 / 10 | 2.86 / 4.68 mm | BPTT Actor terminated at step 3 |
| E3, 512 env | 256,000 | 1,000 / 100 / 100 | 1.51 / 3.23 mm | terminated at step 27 |
| E3, 1024 env | 768,000 | 1,500 / 200 / 200 | 1.45 / 2.87 mm | full 1356-step clip, no contact or hard-limit failure |

The 1024-environment Actor's measured Newton errors were:

| TCP | Position mean / P95 / max (mm) | SO(3) mean / P95 / max (deg) |
| --- | --- | --- |
| Left | 7.77 / 26.34 / 60.61 | 0.98 / 3.05 / 7.63 |
| Right | 18.18 / 51.66 / 80.81 | 4.65 / 15.72 / 22.13 |

It reached the last frame without collisions or safety termination. Maximum physical arm speed
was 2.35 rad/s; maximum acceleration was 73.46 rad/s²; actuator torque saturation rate was
0.142%. During its 1024-environment collection, two off-reference samples triggered safe
episode termination, with no contact. The separate verified direct-reference PD baseline,
which has privileged `q_ref(t)` at every step, attained left/right mean position errors of
0.46/0.64 mm and mean orientation errors of 0.053°/0.085° with no collision. These are
different information conditions.

The 1024 Actor is a **candidate**: the right TCP P95 position 51.66 mm exceeds the 50 mm
research target, and P95 orientation 15.72° exceeds the 15° target. An earlier mean-only
checkpoint label was corrected; the original file is preserved as
`candidate_mean_only_legacy.pt`, and `candidate.pt`/`metrics.json` record the stricter failure.
No checkpoint is marked validated under the stricter mean, P95, safety, and complete-clip gate.

The 4096-environment complete-loop capacity test collected 524,288 real transitions at
48,457 transitions/s, then ran 100 eight-step world-model updates and five 32-step Actor/Critic
updates. Replay used 1.74 GB; PyTorch's peak allocation was 2.43 GB, while `nvidia-smi`
observed approximately 15.7/32.6 GB including Newton. Collection had no contact, termination,
or abnormal speed. Its JSON report is `capacity_4096.json`. A longer 4096-environment run
focused on the 9–16 s error-heavy portion is recorded separately after completion.

The single-motion held-out model scores test unseen environment IDs and sampled state/action
windows, not unseen motions. The teacher-heavy world-model dataset is still a distribution
limit; a low open-loop error alone does not prove safe closed-loop policy control. The
full Newton policy evaluations and separate safety termination enforce that distinction.

## Reproduction and verification

Use the pinned reference and the process-local Isaac Lab/pxr Python path established for this
host. Entry points are `scripts/supermimic/preflight.py`, `train_wave.py`,
`evaluate_wave.py`, `benchmark_capacity.py`, `plot_results.py`, and `render_policy.py`.
The last script renders **recorded Newton physical states** with MuJoCo EGL for viewing;
it does not claim a MuJoCo closed-loop policy execution.

`uv lock --check` passed. Ruff passed for the new code. The combined Marvin asset, motion,
and SuperMimic test suite reported **32 passed, 3 skipped**. Tests include random-pose FK
agreement against MuJoCo, TCP gradient finite differences, SO(3) reward gradients,
eight-step regression, episode-safe Replay sampling, 32-step BPTT gradients, toy controllable
dynamics, and task registration. The actual resume path was exercised in multiple Newton runs.
The task remains single-motion research and has no unseen-motion generalization claim.
