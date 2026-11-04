# Marvin Bimanual S0 v0: measured research baseline

**Status: experimental, not ready for Arm S0 deployment or real hardware.** The
task runs on the complete 54-joint Marvin/Wuji articulation and accepts 14 arm
actions, but the measured policy does not yet solve full-workspace bimanual
6D reaching. No real robot, ROS motor service or online Marvin IK was used.
The [machine-readable experiment summary](experiment_summary.json) records the
final gate status and selected checkpoint.

## Exact model and runtime

The S0 branch starts from the committed asset worktree at
`43cd8c818426b25c2012afdc0143bde16fd56be1`. The external authority is
`dual_arm_marvin.xml` from whole-body-teleop commit
`5d4c345842f197bffeae7d7e08c4c3718230f9e0`, SHA256
`6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce`.
The read-only source checkout was at
`b295caff16e690915c2ba5bd6be58fc5895ed7ea` during this S0 run; the
actual XML bytes still matched the pinned SHA256.
The packaged Newton USD with disabled hand contacts is SHA256
`8ed2cb9a14a813022149aaa8a101159af86dab676aec7dfdde4a00a8d9698fa6`.
The 40 finger joints, their Home position targets, mass, inertia and fixed
mounting remain. Arm collision geometry is active; hand geometry is retained
but hand contacts follow the prior no-hand-contact asset setting. This is not
a grasping or contact-safe policy.

The final asset recheck found 54 MuJoCo controls and 54 USD revolute joints,
with no invalid mass, inertia, drive or limit entries. A fresh 1000-pose,
16-environment MuJoCo/Newton FK parity test passed with 0 failed TCP
comparisons: maximum TCP position difference 0.000950 mm and maximum
orientation difference 0.0000367°. The 32-environment S0 smoke ran 350 steps
with all three modes, 32 timeouts, 0 abnormal terminations and finite data.

Measured host: Python 3.12.3, Isaac Lab 3.0.0, Newton 1.6.1, Warp 1.17.0,
PyTorch 2.11.0+cu130, RSL-RL 5.5.1, RTX 5090 (32607 MiB), driver 595.84.
MJWarp uses CUDA with `use_cuda_graph=False`. Physics is 500 Hz and the policy
acts at 100 Hz. The imported Newton arm tracking profile uses source damping
scale 0.5; it is not a calibrated motor model.

## Contracts and code

- Gym ID: `DexGym-Marvin-Bimanual-SE3-Direct-v0`.
- Environment: `source/dexgym/tasks/marvin_s0/env.py` and `env_cfg.py`.
- Observation: [73D JSON contract](../../source/dexgym/tasks/marvin_s0/observation_schema.json).
  Actor inputs include named arm joints, velocities, left/right TCP position
  and SO(3) log errors, prior action and command, active mask and mode. The
  sampled `q_goal` is not an Actor input. MuJoCo WXYZ is explicitly converted
  to the Isaac Lab XYZW convention in the FK path.
- Action: [14D JSON contract](../../source/dexgym/tasks/marvin_s0/action_schema.json).
  The Actor requests signed joint velocity; a separate limiter integrates
  guarded position targets using 0.8 rad/s command speed, 4 rad/s² command
  acceleration, 0.05 rad command/actual lag and 0.03 rad limit margin.
- Policy: shared [256, 256, 128] ELU MLP for left, right and both modes.
  RSL-RL PPO uses the official Isaac Lab runner. The offline FK teacher uses
  generated joint goals only as labels; the deployed Actor needs no IK.
- Reward terms: coarse/fine position, orientation and coupled SE(3) scores,
  progress, 0.2 s success hold, command smoothness, joint speed and limit
  proximity. Each term is logged independently. No verified per-environment
  collision reward or hard path guard exists yet.

## Fixed evaluation and measured outcomes

`scripts/s0/generate_eval_cases.py` generated an ignored, immutable 96-case
evaluation JSON with SHA256
`62048155f4335dd3fb4cea128d98a2072a339f455114f0eaea5b681ef137d04b`.
It has 16 cases each for local, global, orientation-only, wrist, joint-boundary
and bimanual categories. The source MuJoCo model checked each endpoint and 16
straight joint-path samples for contacts; ten candidates were rejected.
Starts and target TCP poses are stored, and joint goals are retained only for
feasibility audit. The 96 cases use a separate seed from training. No IK
failure cases entered this set because available diagnostic summaries did not
provide a complete independently feasible start/goal pair. Neither model
success nor a fair IK comparison can be asserted for that category.

The success test is position ≤5 mm and orientation ≤2° for at least 20
consecutive 100-Hz steps before each case's command-limit-based time budget.
Inference is deterministic. These figures describe TCP success only; sampled
contacts do not prove continuous path safety.

| Run | Physical PPO iterations | Local 32-case success | Fixed 96-case success | Bimanual 16-case success |
| --- | ---: | ---: | ---: | ---: |
| Zero actions | 0 | 0% | 0% | 0% |
| PPO v1, 1024 env | 250 | 3.1% | not evaluated | not evaluated |
| PPO v2, 1024 env | 250 | 0% | not evaluated | not evaluated |
| PPO v3, 1024 env | 250 | 0% | 0% | 0% |
| FK teacher, 1500 updates, single arm | 0 | 59.4% | 46.9% | 0% |
| FK teacher, 6000 updates, single arm | 0 | 65.6% | 46.9% | 0% |
| FK teacher, 6000 updates, 33% bimanual | 0 | not measured on the local 32-case seed | 49–51% across two repeats | 31.3% |
| PPO from teacher, 2048 env, standard exploration | 200 | 0% | not evaluated | not evaluated |
| PPO from teacher, 2048 env, reduced exploration/learning rate | 50 | not evaluated | 45.8% | 18.8% |
| PPO from teacher, 2048 env, 33% bimanual | 100 | not evaluated | 53.1% | 31.3% |
| Same PPO at 190 cumulative iterations | 190 | not evaluated | 50.0% | 43.8% |
| Same PPO resumed for 200 more iterations | 300 cumulative | not evaluated | 46.9% | 25.0% |

The longer single-arm teacher improved local success but not broad coverage.
Adding bimanual teacher pairs raised bimanual success while boundary success
remained 0%. Mixed-mode PPO briefly raised overall success at 100 iterations,
but left-arm position P95 worsened from 62.3 to 82.4 mm. After 300 cumulative
iterations, overall success fell and left-arm position P95 reached 178.4 mm.
The 190-iteration checkpoint improved bimanual success to 43.8% but also had
179.5 mm left-arm position P95, so it was rejected by the same rule.
The automated comparison rejects both checkpoints against the teacher due to
tail-error regression. Training therefore stopped before a multi-thousand-
iteration run. The 6000-update bimanual teacher is retained as the best
balanced fixed-set checkpoint; the PPO checkpoints remain available for audit.

The latest repeated fixed evaluation of that teacher yielded local 68.8%,
global 18.8%, orientation-only 93.8%, wrist 81.3%, joint-boundary 0%, and
bimanual 31.3%. Across all active arms, mean/P95 position error was
left 19.97/62.26 mm and right 17.63/68.11 mm; orientation mean/P95 was
left 6.43/37.53° and right 4.10/17.08°. Zero cases terminated before their
budget, but one case had sampled arm/base contact. The maximum actual arm
speed reached 3.74 rad/s despite a 0.8 rad/s **command** speed limit. Physical
joint speed is therefore not hard-bounded to the command value.

## GPU capacity and training volume

The [machine-readable capacity run](../../artifacts/s0/capacity/summary.json)
used the same 54-joint asset, 24 rollout steps and 2–3 PPO iterations per size.
Measured last-iteration throughput increased from 3,127 samples/s at 128 envs
to 38,087 at 2048, 81,277 at 4096 and 156,619 at 8192. Peak GPU memory was
3,093, 8,891, 15,029 and 27,371 MiB respectively. The 8192 figure is only a
two-iteration capacity result (84% VRAM), not a proven long-run stable choice.
Learning experiments used 1024 or 2048 envs to preserve a margin and allow
fixed evaluation. GPU peak temperature in the sweep was 41°C. A separate
single sample during 2048-env PPO measured 8,978 MiB GPU memory, 28% GPU
utilization, 44°C, and 63,162 MiB available host RAM out of 128,518 MiB.
The sweep did not separately measure PPO update time or continuous CPU/RAM
peaks; those fields remain unmeasured.

Actual physical PPO volume: Round 0 7,680 transitions, three 1024×250 runs
18,432,000 transitions, 2048×200 standard-exploration run 9,830,400,
2048×50 reduced-exploration run 2,457,600, and 2048×300 mixed-mode run
14,745,600 transitions: **45,473,280 total environment transitions**.
Offline FK teacher
training generated 3,072,000 pairs for the 1500-update run and 12,288,000 per
6000-update run; these are supervised samples, **not** physical environment
transitions.

## Reproduction

Install with `uv sync --locked --group dev`, set `DEXGYM_MARVIN_MJCF` to an
authorized local `dual_arm_marvin.xml`, and use the pinned packaged USD.
Source XML, meshes, generated cases and checkpoints are not published here.
On the verified host, prepend
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` if its Conda C++ runtime
interferes. The exact evaluated data and logs are in ignored `artifacts/s0/`
and `logs/rsl_rl/dexgym_marvin_bimanual_s0_v0/` on this host.

```bash
uv run --no-sync python scripts/s0/generate_eval_cases.py --output artifacts/s0/fixed_eval_cases.json
uv run --no-sync python scripts/s0/pretrain_actor.py \
  --template /path/to/compatible/model_0.pt \
  --output artifacts/s0/pretrain/bootstrap_6000_bimanual.pt \
  --updates 6000 --batch-size 2048 --seed 42001 --bimanual-fraction 0.33
uv run --no-sync python scripts/s0/pretrain_actor.py \
  --template /path/to/compatible/model_0.pt \
  --output artifacts/s0/pretrain/bootstrap_6000_bimanual_std003.pt \
  --updates 6000 --batch-size 2048 --seed 42001 --bimanual-fraction 0.33 \
  --policy-std 0.03
uv run --no-sync python scripts/s0/evaluate_local.py \
  --checkpoint artifacts/s0/pretrain/bootstrap_6000_bimanual.pt \
  --dataset artifacts/s0/fixed_eval_cases.json --steps 250 \
  --output artifacts/s0/pretrain/eval_fixed.json
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task DexGym-Marvin-Bimanual-SE3-Direct-v0 --num_envs 2048 \
  --max_iterations 100 --seed 42 --run_name s0_mixed_bounded \
  --checkpoint artifacts/s0/pretrain/bootstrap_6000_bimanual_std003.pt \
  --no-resume_optimizer_state agent.algorithm.learning_rate=1e-5 \
  agent.algorithm.entropy_coef=0.0 agent.algorithm.schedule=fixed \
  env.bimanual_fraction=0.33 physics=newton_mjwarp
uv run --no-sync python scripts/s0/smoke.py --num-envs 32 --steps 350 \
  --output artifacts/s0/smoke.json
uv run --no-sync python scripts/s0/compare_eval.py \
  --baseline artifacts/s0/pretrain/eval_fixed.json \
  --candidate artifacts/s0/ppo/eval_fixed.json \
  --output artifacts/s0/ppo/comparison.json
```

`evaluate_local.py` loads the official RSL-RL checkpoint, runs finite Newton
steps with deterministic Actor inference and writes per-side/per-category
metrics. `compare_eval.py` refuses changed case hashes or success definitions
and rejects a candidate with measured safety or major category regressions.
The comparison is a checkpoint selection aid, not a safety certification.

For finite replay of the retained best Actor, point `S0_CHECKPOINT_DIR` at the
local ignored directory containing `bootstrap_6000_bimanual.pt` and
`fixed_eval_cases.json`:

```bash
uv run --no-sync python scripts/s0/evaluate_local.py \
  --checkpoint "$S0_CHECKPOINT_DIR/bootstrap_6000_bimanual.pt" \
  --dataset "$S0_CHECKPOINT_DIR/fixed_eval_cases.json" --steps 250 \
  --output artifacts/s0/replay_best.json
```

The last PPO model is retained for investigation, **not** selected as the
best. A bounded resume from it uses its optimizer state and the exact mixed
mode configuration; this is a recovery command, not a recommendation for an
unreviewed long run:

```bash
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task DexGym-Marvin-Bimanual-SE3-Direct-v0 --num_envs 2048 \
  --max_iterations 50 --seed 42 --run_name s0_resume_diagnostic \
  --checkpoint "$S0_CHECKPOINT_DIR/model_298.pt" --resume_optimizer_state \
  agent.algorithm.learning_rate=1e-5 agent.algorithm.entropy_coef=0.0 \
  agent.algorithm.schedule=fixed env.bimanual_fraction=0.33 \
  physics=newton_mjwarp
```

## Open research and safety gates

1. Train/evaluate with collision-checked training starts and targets, and add
   a performant per-environment contact/path guard. The current local GPU
   sampler enforces joint bounds but does not run MuJoCo/Newton collision
   checks on every generated training goal.
2. Calibrate actual velocity and acceleration behavior. The command limiter
   cannot guarantee physical joint speed under this imported position drive.
3. Build a successful PPO fine-tune without destroying the FK-teacher
   baseline; only then advance to wrist, global/boundary and full bimanual
   curricula based on held-out success rather than iteration count.
4. Obtain complete, independently feasible IK-failure start/target cases and
   compare both controllers under the same collision and limit checks.
5. Validate policy behavior in MuJoCo separately. Existing asset FK parity is
   not a policy sim-to-sim result.

The current answer to “has S0 learned robust 6D bimanual control from varied
starts?” is **no**. The FK-supervised Actor controls many local and wrist cases
in Newton, but joint-boundary success is 0%, bimanual success is 31.3% and
PPO has not improved the independent fixed set. This is a research baseline,
not a deployable S0.
