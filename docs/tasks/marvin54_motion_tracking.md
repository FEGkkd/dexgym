# Marvin M6 and dual Wuji Hand2.2 motion tracking

Base commit `b7749f3eae84ddf820d9fde913eb6ab411a8decc`. Task-specific choices below are
local implementation defaults unless specified by the user.

- Input is the verified 50 Hz reference NPZ. Accept multiple validated clips with
  identical model and joint schemas. Advance each clip by one frame per control
  step, sample reset frames, and truncate at its end. The first pilot uses the
  single Adjust Shelf Labels clip, so full-clip replay measures in-clip tracking.
- Real transitions use full-contact Newton MJWarp at 500 Hz. One 50 Hz action
  holds 54 position targets over ten substeps. The default `home` action mapping
  is `clip(q_home + scale * actor_output, lower, upper)`. The fixed Home posture
  comes from the MJCF home keyframe. Scale is the greater of the two distances
  from Home to the combined joint and actuator limit. Evaluation compares the
  learned policy with direct reference-position PD replay.
- The previous `reference` mode centers a bounded residual on the next
  reference pose, scaled to 0.18 rad for arms and 0.25 rad for hands.
  The previous `feedforward` mode derives position offsets at
  each reference state from MuJoCo bias and passive forces, PD velocity
  compensation, and the original actuator gains. It uses residual bounds of
  0.008 rad per arm joint and 0.025 rad per hand joint. The same hard physical
  termination thresholds apply. The zero-residual baseline uses the same
  feedforward. Keep the checkpoint together with the run's provenance.json,
  which records the nominal controller and Motion Library hashes. The nominal
  target holds for 20 ms and excludes acceleration compensation. The earlier
  offline replay used interpolated compensation at 2 ms.
- The actor observes joint position and velocity, the next reference joint pose
  and velocity, a further preview pose, and its previous action. The World Model
  observes joint position, joint velocity, and the executed 54 servo targets.
- A differentiable FK reward uses both wrists and ten tip poses in the robot base
  frame, both elbow positions, joint and velocity tracking, relative wrist
  separation, action smoothness and deep contact. Physical contact is measured
  over all ten substeps. More than 3 mm penetration, a joint violation over
  0.05 rad, or a nonfinite state terminates a physical rollout. Truncation and
  termination remain separate.
- Reuse the DexGym `dex_reference` World Model, replay, failure head, analytic
  Actor BPTT and TD(lambda) Critic. Use 8 and 32 step horizons with bounded
  warmup and small networks for the host test. The pilot starts from random
  weights and uses real Newton transitions throughout training.
- The clip starts at time zero, has at least 33 frames and must match the
  model hash, joint order and unit WXYZ quaternion contract. Reset samples a
  reference frame weighted by clip length and resets to the reference q and
  qvel with zero previous action. The latest allowable reset is the penultimate
  frame for clips longer than 64 frames. Shorter clips reserve all 32 imagination
  steps, including the case where frame zero is the only valid reset. The
  five-frame preview clamps to the last clip frame.
- Physical contact labels use any penetration and depth above 0.5 mm. Joint
  and nonfinite checks use the end-of-control-step state. Imagined continuation
  uses a smooth joint-limit threshold, exact clip-end termination and a
  learned, capped failure probability after warmup. Imagination starts leave
  a full 32-step window. World model holdout is by environment, on the same
  motion library. The reward uses position and matrix errors and penalizes
  changes in action. The previous residual modes also penalize residual magnitude.
- Replay features used by the failure head advance to the post-transition
  reference phase and include the executed action. Actor imagination starts
  and observation normalization use the training environments. Evaluation
  reports both complete episode statistics and means over the shared prefix,
  and identifies the checkpoint, evaluated clip and nominal mode.

## Run on the Ubuntu training host

```bash
cd ~/dexgym/worktrees/marvin54-motion-tracking
reference="$HOME/dexgym_data/egopro_adjust_shelf_54dof/clearance_7mm/reference_54dof.npz"
task=dexgym.tasks.direct.marvin54_tracking.train
run="$HOME/dexgym_data/marvin54_home4096_new"
bash scripts/supermimic/host.sh -m "$task" --reference "$reference" --check-only \
  --nominal home --num-envs 4096 --output "$run/contract"
bash scripts/supermimic/host.sh -m "$task" --reference "$reference" \
  --nominal home --num-envs 4096 --iterations 1000 --wm-only-iters 40 \
  --warmup-steps 32 --buffer-steps 128 --wm-batches 4 --wm-batch 1024 \
  --pi-batch 192 --save-interval 50 --log-interval 10 --output "$run/train"
bash scripts/supermimic/host.sh -m "$task" --reference "$reference" \
  --nominal home --evaluate "$run/train/model_final.pt" --output "$run/eval"
```

The entry point uses DexGym's existing SuperMimic engine and the host runtime
wrapper. Add additional `--reference` files to the same Motion Library.
Use `--clip-index` for matched reference PD and policy evaluation in Home mode.

## Adjust Shelf Labels pilot

Run root `~/dexgym_data/marvin54_tracking_pilot/`. The 80-iteration
`feedforward_seed42` run collected 8,448 physical transitions and applied 160
World Model, 160 failure-head, 50 Actor and 1,000 Critic optimizer steps.
Newton reset position matched the reference exactly. The maximum sampled
Newton wrist position difference from differentiable FK was 0.000813 mm.

| Full Newton replay | Zero residual, feedforward | Learned residual, feedforward |
| --- | ---: | ---: |
| Control steps completed | 1,691 / 1,691 | 1,691 / 1,691 |
| Mean wrist position error (mm) | 4.88 | 9.20 |
| Mean elbow position error (mm) | 0.73 | 2.54 |
| Mean fingertip position error (mm) | 8.74 | 14.05 |
| Mean wrist orientation error (deg) | 2.16 | 2.85 |
| Mean fingertip orientation error (deg) | 2.48 | 3.80 |
| Mean relative wrist vector error (mm) | 7.95 | 13.22 |
| Steps with penetration over 0.5 mm | 4 | 7 |
| Maximum penetration (mm) | 1.17 | 0.74 |

Both completed the clip without triggering safety termination. The learned
residual did not improve mean tracking over its matched nominal controller.
The 120-iteration raw-reference run applied 80 Actor updates and its final
policy terminated on step 13. Its own zero-residual baseline ran the entire
clip. The training and evaluation logs preserve Newton line-search warnings.
Only one clip is available, so these runs measure in-clip performance.

Keep `feedforward_seed42/model_final.pt` and
`feedforward_seed42/provenance.json` together. The physical replay statistics
are stored under `eval_feedforward_seed42_release/evaluation.json`.

[SuperMimic paper](https://arxiv.org/abs/2610.28480) · [Official SuperMimic](https://github.com/HybridRobotics/supermimic) · [Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic) · [TrackerLab SuperMimic](https://github.com/Renforce-Dynamics/trackerlab-supermimic)
