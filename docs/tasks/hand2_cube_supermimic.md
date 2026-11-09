# Wuji Hand 2 cube reorientation: native SuperMimic reproduction

Status: native task port and from-scratch training are verified. A 4096-environment,
8000-iteration run completed 2387 goals across 512 evaluation episodes. Mean goals
per episode remains below the public reference model. See the
[final evaluation report](../experiments/cube_reorient_final_evaluation.md).

The initial port omitted the upstream ground plane. The first training run was
stopped after preserving its 1000-iteration checkpoint. The corrected scene adds
the ground and matches the full upstream scene options; a clean 4096-environment
run uses `train_seed42_scene_v2`. Historical results below are explicitly qualified.

The reference is CumulusAlpha/Dex-Supermimic at commit
`2becb7380fed86171056f9bf4338b2a6a0c14b01`. The task uses the reference
Hand 2.2 asset, despite the upstream display name "Wuji Hand 2". Existing
Hand 2.1 PPO and no-object SuperMimic tasks are separate and unchanged.

## Code ownership

- Registered task: `DexGym-Wuji-Hand2-Cube-Reorientation-SuperMimic-Direct-v0`.
- Task: `source/dexgym/tasks/direct/hand2_cube_supermimic/`.
  `env_cfg.py` owns physics/scene configuration; `env.py` owns real task timing,
  reset and observation history. `mdp.py` owns reward and command transitions.
  `native.py`, `randomization.py`, and `disturbance.py` implement the simulator boundary.
- `adapter.py` provides task-specific state, differentiable geometry, reward,
  command transitions and disturbance semantics to the optimizer.
- Reusable optimizer: `source/dexgym/algorithms/supermimic/dex_reference/`.
  `engine.py` owns collection, replay, WM fitting, actor BPTT, critic updates
  and checkpoints. `config.py` and `networks.py` own the reference recipe.
- Bundled assets: `source/dexgym/assets/wuji_hand2_cube/`.

## Reference recipe and task contract

Default training uses 4096 environments (requested for this host), WM batch
2048, actor batch 1024, replay capacity 1000 control steps, 64 initial collection
steps, 500 WM-only iterations, 10 WM updates per iteration, WM horizon 8 and actor
horizon 32. UTD 0.4 alternates 2 and 3 control steps per iteration. Increasing
environments increases real transitions per iteration; it does not increase
optimizer batch sizes. The LR schedule is independently fixed at 250000 iterations.
The default run is 8000 iterations; completion must be decided from evaluation,
not from reaching this iteration number.

WM widths are 2048/2048/2048, actor 512/256/128, critic 512/512/256/128.
The independent failure head is part of the pinned reference. No actor trust gate
is added. Actor updates freeze WM, failure-head and critic parameters while
preserving derivatives through their computations.

The task runs at 50 Hz (physics 100 Hz). Actor observations contain 3 histories
of 69 features, concatenated by observation term (207 total). Physical state is
58D, state delta 52D, critic input 211D and WM input 133D. Actions command absolute
joint targets through the reference scale, soft limits, EMA and encoder bias.
Contact labels are privileged simulation supervision, not physical tactile input.

Transition capture is before external pushes and before resets. Replay stores
both the pre-push next state and the observed push delta. A valid terminal
transition participates in fitting, but sequence masks prevent learning across
an episode boundary. The real collector explicitly resets terminated lanes.

## Native backend adaptations

The reference uses MJLab/MuJoCo Warp 3.11. This port uses Isaac Lab 3 with Newton
and MuJoCo Warp 3.12. It is not a claim of bitwise physical-trajectory identity.
The effective reference integrator is implicitfast, even though the underlying
asset's initial MJCF specifies Euler. The native configuration follows the
resolved task configuration.

Native pose quaternions are XYZW; task math uses WXYZ/rotation matrices.
The USD importer splits PD actuators; gains, joint effort limits and total torque
are checked at the simulator boundary. Convex collision meshes explicitly retain
source hull detail instead of Newton's 64-vertex default. The cube is a standalone
rigid body. Randomization writes the persistent Newton model and propagates changes
to MuJoCo Warp, including broadphase bounds after geometry scaling.

Fixed-base root resets write Newton's authoritative root-joint transform in
addition to Isaac Lab's pose proxy. Otherwise the proxy changes while the physical
hand remains untilted. The observation reader uses actual palm body transforms.

The pinned reference's replay disturbance reconstruction and imagined observation
encoder-bias convention are retained. These are upstream behavioral choices, not
new mechanisms. Future changes must be evaluated separately from this reproduction.

## Running on the authorized Ubuntu host

The existing host wrapper selects the installed working runtime without changing
the shared virtual environment. From the dexgym checkout:

```bash
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.validate_native --output /tmp/cube_native.json
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.validate_algorithm --reference /home/ubuntu/dexgym/artifacts/dex_supermimic_reference --output /tmp/cube_algorithm.json
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.smoke --full-config --num-envs 4096 --output /tmp/cube_capacity.json
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.train --num-envs 4096 --output /path/to/new_cube_training_run
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.evaluate --checkpoint /path/to/model.pt --output /tmp/cube_evaluation.json
```

Omit `--checkpoint` for an untrained baseline. Evaluation uses deterministic actor
actions, observation noise, startup/reset randomization and the same fresh reference
push curriculum. It records four complete episodes per lane by default; every lane
has equal weight. `--no-push` is a separately labeled evaluation condition.
Compare identical seeds and conditions using success counts, fraction completing
at least 1/3 goals, drops, episode duration and orientation error. Loss alone is
insufficient evidence of successful reproduction.

To regenerate bundled MJCF/USD (optional):

```bash
bash scripts/supermimic/hand_s0_host.sh -m dexgym.assets.wuji_hand2_cube.export_assets --importer-root /home/ubuntu/dexgym/.venv/lib/python3.12/site-packages/isaacsim
```

## Validation and historical evidence

Authoritative scene-v2 reports: `original_scene_probe.json` (full upstream scene),
`native_scene_parity.json` (live Warp solver options and actual floor contact),
`native_with_ground.json` (FK/DR/reset), and `full_config_4096_scene_v2.json`
(production-size learning chain). All pass. Live solver option differences are zero;
terrain material differences are float32 rounding. Four dropped cubes contact the
floor and settle at center height 0.0268714 m.

The earlier reports below predate the ground fix. They remain historical evidence
for the optimizer and individual interfaces, not a complete scene-fidelity claim.

Machine-readable validation snapshots are committed in
`docs/experiments/cube_reorient_validation/`. Registration and original hand-task
regression checks: 7 passed.

Host evidence directory: `/home/ubuntu/dexgym/artifacts/cube_reorient/`.

- `native_contract_rebuilt.json`: 16 environments, 240 steps with reset and pushes;
  maximum FK discrepancy 6.59e-7 m; Kp/Kd/cube mass/cube size errors zero before
  and after simulation/resets; partial-reset histories and captured transitions preserved.
- `algorithm_parity_failure_active.json`: identical source weights, minibatches
  and RNG; terminal sample included and failure head active. WM/failure-head and
  actor/critic update metrics and resulting parameters match exactly.
- `physics_comparison.json`: collision eligibility matches; convex support geometry
  matches to numerical precision. Native/backend rounding differences remain.
- `full_config_smoke.json`: 1024 environments, production network/batch sizes,
  AMP and compilation; real collection, WM8 fitting and Actor32/Critic updates pass.
  This report explicitly says `trained_policy: false`.

- `full_config_4096_smoke.json`: requested 4096 environments pass the same production
  chain; peak PyTorch allocated memory 12063504896 bytes. Compilation may fall back
  to eager reward evaluation after view-layout specialization; gradients still pass.

- `eval_untrained_seed123.json`: 256 complete episodes, 0 successful goals,
  drop fraction 0.03515625, mean duration 48.5354 s and mean orientation error
  2.14859 rad. Holding the cube without rotating it is therefore not sufficient.

The first production run uses code `70e4a6a`, 4096 environments, seed 42,
8000 configured iterations and the reference 500-iteration WM warmup. Host output:
`/home/ubuntu/dexgym/artifacts/cube_reorient/train_seed42/`. That initial run is superseded because its scene lacked ground collision. Its
1000-iteration checkpoint completed only one goal across 256 evaluation episodes.

Training and policy evaluations must be recorded before changing the status above
or claiming the task has been trained successfully.

## Published-actor diagnostic (historical scene revision 1)

`evaluate --reference-checkpoint PATH` loads the upstream public Hand 2 actor
without changing native training. Its released 8000-iteration weights predate the
current reference code's EmpiricalNorm; the diagnostic retains the saved actor's
RunningNorm computation with epsilon 0.01. It validates joint ordering and 50 Hz
configuration and records checkpoint kind, normalization and SHA-256. Checkpoints
are loaded with a restricted weights-only unpickler, including known NumPy RNG types.

The public release at `pretrained-wuji-v2/model_hand2_reorient_8000.pt` has SHA-256
`269144cc63858d04730f89a856ccadbc9a8a9a7a7961a21f993ff5aee03f1497`.
Its metadata says weights are unchanged from the earlier `pretrained-wuji-v1` release.
This diagnostic is not an evaluation of an actor trained by dexgym, nor an assertion
that the older checkpoint was trained by the current pinned source revision.

Under the same seed-123, 64-lane, four-episode protocol, the published actor achieves
7.1211 goals per episode, 90.625% of episodes with at least one success, 69.53125%
with at least three, mean angle error 0.8334 rad, and drop fraction 63.671875%.
The untrained baseline achieves zero goals. This supports functional observation,
action and physical-environment compatibility; differences from the upstream backend
and the high drop rate still warrant evaluation. Native training remains separate.
The full result is `docs/experiments/cube_reorient_validation/eval_reference8000_seed123.json`.

## Full-scene audit and correction

The upstream MJLab scene includes a plane at z=0 with friction [1, 0.005, 0.0001],
condim 3, solref [0.02, 1] and solimp [0.9, 0.95, 0.001, 0.5, 2]. The native
`ground.usda` now preserves these values and is bundled without external asset URLs.
The resolved upstream parent scene also keeps disableflags=0; the native task no
longer applies the child asset's EULERDAMP-disable flag. Both backends use implicitfast.

Compare live Warp options, not only CPU-model metadata: Newton supplies timestep
when stepping, and MuJoCo Warp clamps the requested 1e-8 solver tolerance to 1e-6.
The upstream MJLab runtime does the same. The verifier checks relative precision
for small option values rather than allowing a loose global absolute tolerance.

In a runtime with the pinned reference's MJLab dependencies:

```bash
python -m dexgym.tasks.direct.hand2_cube_supermimic.probe_reference_scene --reference /path/to/Dex-Supermimic --output /tmp/reference_scene.json
```

Then in the native runtime:

```bash
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.validate_scene --reference-scene /tmp/reference_scene.json --output /tmp/scene_parity.json
```

The stored probe used MuJoCo Warp 3.12, as did the native runtime. It resolves the
pinned source's task configuration but does not establish 3.11/3.12 trajectory identity.

## Scene-v2 baseline and published actor

Under identical seed-123, 64-environment, four-episode evaluation settings:

| Policy | Goals/episode | Episodes with >=1 goal | Episodes with >=3 goals | Drop fraction | Mean angle (rad) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untrained | 0 | 0 | 0 | 0.0234375 | 2.22934 |
| Published upstream actor | 6.50391 | 0.863281 | 0.644531 | 0.660156 | 0.834614 |

These scene-v2 results supersede the earlier diagnostic numbers. They demonstrate
that the native environment supports the reference actor's continuous reorientation;
they are not results of the native training run. Its completed results are in the
[final evaluation report](../experiments/cube_reorient_final_evaluation.md).

## Record and render native policy behavior

Evaluation can optionally record the first four environment lanes, retaining their
joint positions, actual palm/cube poses, targets, episode indices and goal counts.
Recording does not alter actions or task rules. The renderer replays those poses
without running another physics rollout. Overlay text identifies checkpoint type,
scene revision, time, environment, episode, goals and target error.

```bash
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.evaluate --checkpoint /path/to/model.pt --output /tmp/eval.json --trace-output /tmp/trace.npz
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.render_trace --trace /tmp/trace.npz --output /tmp/rollout.mp4 --env-index 0 --seconds 30
```

Rendering requires EGL, Pillow and an `ffmpeg` executable. It produces an MP4,
first-frame PNG, and JSON provenance/pose-reconstruction report. The reference
actor's 30-second example was checked with zero palm/cube root-position error.
Playback uses the bundled nominal visual geometry; it is a visualization of
recorded simulation, not real camera footage or a new dynamics simulation.

## Live TensorBoard

The running training process writes `metrics.jsonl`. The optional TensorBoard
mirror imports its existing history and follows new records every five seconds;
it also imports native checkpoint evaluation results. This does not change the
optimizer or require restarting training.

```bash
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.tensorboard --run-dir /path/to/run --evaluations /path/to/evaluation-root --follow
tensorboard --logdir /path/to/run/tensorboard --port 6007 --reload_interval 5
```

Use `world_model/train_loss` and `world_model/holdout_loss` for WM fitting;
`actor/loss`, `critic/loss`, and `gradients/*` for optimizer diagnostics;
`train/rotation_error_rad` for collected behavior; and `eval/goals_per_episode`
for completed-task evaluation. The collection drop fraction is per control step,
while `eval/drop_fraction_per_episode` is per episode. Actor/Critic curves start
after the 500-iteration WM warmup. Avoid launching multiple mirror writers into
the same events directory simultaneously.

## Frozen world-model prediction diagnostic

```bash
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.evaluate_world_model --checkpoint /path/to/model.pt --output /tmp/wm_prediction.json
```

This diagnostic collects fresh seed-321 trajectories with a frozen actor and WM,
64 environments, 150 burn-in steps, and 256 recorded control steps. It evaluates
nonoverlapping windows at horizons 1/4/8/16/32, reports physical-unit mean/p95
errors, and compares with a persistence baseline on the same states. Predictions
receive recorded executed targets and the reference replay reconstruction of
exogenous pushes. Future physical states are used only as error targets, never
as WM inputs. Terminal transitions are included; subsequent transitions across
reset boundaries are masked. Model parameters require no gradients and no
optimizer is constructed; normalization buffers are checked unchanged afterward.

These are correlated trajectory samples under the current deterministic policy,
not a complete contact-mode benchmark. The diagnostic measures open-loop state
prediction conditioned on recorded controls and pushes; it does not validate
imagined action gradients or demonstrate successful task learning.
