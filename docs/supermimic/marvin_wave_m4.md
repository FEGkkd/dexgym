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
| E4, 4096 env | 2,048,000 | 4,000 / 500 / 500 | 0.95 / 1.59 mm | full 1356-step clip, no contact or hard-limit failure |
| E5 mixed, 4096 env | 2,048,000 | 4,000 / 500 / 500 | 0.88 / 1.27 mm | safety termination at step 548 |
| E6 corrective, 4096 env | 2,048,000 | 4,000 / 200 / 200, plus 1,500 auxiliary Actor updates | 0.66 / 1.26 mm | full clip; independent right TCP position P95 50.97 mm |
| E7 prolonged, 4096 env | 2,048,000 | 4,000 / 1,000 / 1,000 | 0.60 / 0.77 mm | full clip; independent right TCP position P95 50.65 mm |
| E8 cumulative 2000, 4096 env | 2,048,000 | 4,000 / 800 / 800 | 0.60 / 1.00 mm | in-process full clip; two independent replays terminate at steps 1329/1328 |

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

The subsequent 4096-environment run used 50% of imagination starts from source frames
450–849 (the 9–16 s interval with high physical tracking error), 10% from the first 50
frames, and 40% uniformly. Its Newton evaluation completed the clip with zero contact,
zero illegal joint states, 2.08 rad/s maximum actual arm speed, and 0.205% torque
saturation. It achieved left/right mean position 7.18/18.86 mm and mean orientation
0.91°/4.83°. Left position P95 was 24.66 mm; right position P95 was **52.98 mm**.
Right orientation P95 improved to 14.00°. This is another **candidate**, since right
position P95 still exceeds 50 mm. Its training wall time was 484 s, collection throughput
50,659 physical transitions/s, and recorded PyTorch peak allocation 6.45 GB. During
the formal run `nvidia-smi` showed approximately 20.0/32.6 GB in use including Newton.
The actor and model checkpoints, learning curves, and every 50-update Actor snapshot
are under `e4_4096env_hard500/`.

Three additional isolated-process Newton replays of the same E4 Actor all completed
the full motion without contact or illegal joints, but yielded right TCP position P95
of 59.21, 60.79, and 57.28 mm. The two replays with 500 Hz contact audit recorded zero
contact physics steps. This spread is material: the in-process 52.98 mm number is not
presented as a reproducible best score. All three independent runs failed the strict
right TCP tail criterion. The 250-update intermediate Actor was also replayed and
terminated at control step 23 due to left TCP deviation; its partial-clip errors are
not compared with complete-clip scores.

The last independent replay records every Actor action, executed command, and 54DoF
Newton physical state. `validate_policy_model.py` evaluated the learned model on this
actual policy trajectory. Its right TCP 8-step error was mean/P95 5.01/13.14 mm;
the 32-step error was 12.06/35.03 mm. These values are substantially worse than the
0.95/1.59 mm eight/thirty-two-step means on teacher-heavy held-out environments.
The world model therefore has a measurable on-policy distribution gap. E5 tested a
bounded mixed teacher/Actor collection profile and active-phase resets to address it.

E5 used a 0.3 mixture of Actor and verified reference commands on about half of its
2,048,000 Newton transitions, with 50% of resets directed to the active 9–16 s interval
and 10% to the opening phase. Collection had no collision or termination and maximum
actual arm speed 3.08 rad/s. The teacher-heavy held-out 32-step TCP mean error fell to
1.27 mm, but the Actor terminated at step 548; its executed right-arm command differed
from the time-matched reference by about 0.10 rad on average during 10–11 s. Its
on-policy world model still had right TCP 32-step mean/P95 errors of 12.80/31.96 mm.
E5 demonstrates that lower teacher-heavy model loss alone did not improve the actual
policy. Its `candidate.pt`, partial Newton state trace, and failure metrics are retained.

E6 used a wider safe command mixture, then trained a corrective Actor initializer on
**real off-reference Newton states** before a separate, small-learning-rate 32-step
BPTT stage. Its 4096-environment collection yielded 2,048,000 transitions in 41.18 s
(49,732/s), with no contact or termination. The model had 0.60/0.66/1.26 mm mean
held-out TCP position error at one/eight/thirty-two steps. The 1,500-update auxiliary
initializer was evaluated separately: its independent full Newton replay had right
TCP position mean/P95 12.85/47.00 mm and rotation mean/P95 2.88/11.13°. This is a
supervised baseline, not a SuperMimic-trained result. After 200 genuine 32-step BPTT
updates, the in-process full replay met the research thresholds: left/right position
mean 7.80/12.99 mm, P95 35.34/47.57 mm; orientation mean 0.98/2.71°,
P95 4.37/9.31°. The independent zero-perturbation replay completed 1356 control steps,
with zero 500 Hz contact steps and zero illegal joint states, but right position P95
was **50.97 mm** (mean 13.99 mm), slightly above the 50 mm target. Left position
mean/P95 was 8.64/38.26 mm; left/right orientation mean/P95 was 1.08/4.78° and
2.92/8.84°. Maximum actual arm speed was 2.00 rad/s, joint position RMSE 0.0295 rad,
and torque saturation rate 0.011%. The BPTT Actor improved the right orientation tail
over its initializer but worsened the right position tail in this independent replay.
Thus the E6 `best.pt` is an in-process best **candidate**, not independently validated.
Its full traces, plots, auxiliary baseline, and independent safety report are under
`e6_4096env_corrective200/`.
On its independent full Newton policy trajectory, the E6 world model had right TCP
position prediction mean/P95 0.81/2.15 mm at one step, 2.28/7.50 mm at eight steps,
and 5.70/18.41 mm at thirty-two steps. The on-policy 32-step mean is lower than E4's
12.06 mm, but still much larger than E6's 1.26 mm teacher-heavy held-out mean.
The E6 comparison MP4 under `previews/` is 27.2 s and shows recorded Newton physical
states against the pinned FK reference; its frame around 9 s was visually checked.

E7 extended the same 4096-environment method by 1000 further 32-step Actor/Critic
updates after recollecting 2,048,000 real Newton transitions. Collection throughput
was 50,503/s; the run took 897 s including full physical evaluation, with a 6.38 GB
PyTorch peak allocation. The 500-update snapshot completed two independent full
Newton replays with zero physics-step contacts and no illegal joint state. Its right
TCP position mean/P95 was 13.79/49.22 mm on the first replay and 14.07/50.31 mm on
the repeat. The 1000-update terminal Actor's in-process right position P95 was
49.76 mm; its independent replay was **50.65 mm**, with mean 14.71 mm. Independent
left position mean/P95 was 9.42/39.91 mm, and left/right rotation mean/P95 was
1.18/4.88° and 3.14/10.50°. The terminal Actor completed all 1356 control steps,
with zero contacts and illegal joint states, 2.01 rad/s maximum actual arm speed,
and 0.0105% torque saturation. Both E7 snapshots remain candidates because the
strict right position P95 threshold is not reproducibly met. Their complete traces
and plots are under `e7_4096env_bptt1000/`. An on-policy model audit using the
independent 1000-update Newton trace found right TCP 8/32-step position prediction
mean/P95 of 2.31/7.04 mm and 5.34/17.61 mm, respectively. The model improved
slightly over E6, while physical position P95 did not improve consistently.

E8 added 800 BPTT updates and another 2,048,000 real Newton transitions, bringing
cumulative E6+E7+E8 Actor and Critic updates to **2000**. It used the same collection,
reward, and action configuration. Collection throughput was 51,063 transitions/s;
the run took 722 s including evaluation. The cumulative 1500-update snapshot
(`actor_bptt_0300.pt`) completed two independent zero-perturbation full Newton
replays, with right TCP position mean/P95 13.48/49.93 and 11.68/46.22 mm. Both
recorded zero physics-step contacts and no illegal joint states. It nevertheless
terminated at control step 16 under ±0.03 rad initial joint perturbation.

The cumulative 2000-update terminal Actor improved the active 8–12 s wave segment:
independent right TCP position mean/P95 changed from the 1500-update snapshot's
44.77/70.77 mm to 33.79/65.89 mm; right orientation mean/P95 changed from
8.95/17.31° to 6.88/15.88°. Its in-process full replay recorded right position
mean/P95 11.74/45.83 mm. However, two **independent** zero-perturbation replays
terminated at control steps 1329 and 1328 during the rapid return-to-Home phase.
In the first, left/right position max reached 159.91/93.02 mm; the second reached
122.83/70.33 mm. Contact and joint-limit counts remained zero. The command lag
against the time-matched reference grew rapidly across frames 1320–1328; this is
a genuine closed-loop tracking failure, not a collision or a partial-video artifact.
The 2000-update Actor also terminated at step 19 under ±0.03 rad initial perturbation.
Its lower partial-clip mean error must **not** be compared as a complete-motion win.

The current best **complete zero-perturbation candidate** is the cumulative 1500-update
snapshot. No Actor has passed the small-initial-perturbation robustness test. The
next bounded experiment therefore adds actual ±0.03 rad reset states and focuses
imagination sampling on the opening and terminal phases while retaining mixed
teacher/Actor Newton collection. Its result is reported separately from the clean
2000-update training-length comparison above.

The full E4 comparison video is `e4_4096env_hard500/previews/newton_policy_vs_reference.mp4`.
It shows reference FK beside the **recorded Newton physical policy state** with both
TCP orientation axes. The full-motion reference and direct PD videos remain in
`/home/ubuntu/dexgym_data/marvin_motion_library_v1/bimanual/previews/`.

The earlier 4096-environment complete-loop capacity test collected 524,288 real transitions at
48,457 transitions/s, then ran 100 eight-step world-model updates and five 32-step Actor/Critic
updates. Replay used 1.74 GB; PyTorch's peak allocation was 2.43 GB, while `nvidia-smi`
observed approximately 15.7/32.6 GB including Newton. Collection had no contact, termination,
or abnormal speed. Its JSON report is `capacity_4096.json`.

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
