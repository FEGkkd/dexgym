# Wuji Hand 2 cube reorientation: native SuperMimic reproduction

Status: native task and optimizer parity validated; learned-policy reproduction is not yet established.

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
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.train --num-envs 4096 --output /home/ubuntu/dexgym/artifacts/cube_reorient/train_seed42
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

## Evidence so far

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

Training and policy evaluations must be recorded before changing the status above
or claiming the task has been trained successfully.
