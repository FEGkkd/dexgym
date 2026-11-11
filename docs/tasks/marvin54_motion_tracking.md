# Marvin M6 and dual Wuji Hand2.2 motion tracking

## First-event task contract

Base commit `fd2cfb3b94876ba4f38d51a68a63c1423254998d`. The training
entry point defaults to the first-event contract. Older checkpoints record
the control-boundary contract through their original provenance.

| Decision | Source |
| --- | --- |
| Hold each command for a 20 ms control period. Check safety every 2 ms and latch the first failing state, actuator effort, depth and cause for each environment. Newton keeps its synchronized batch clock. The failed episode ends at its captured state and that environment resets at the control boundary. | User direction. Logical per-environment stopping is the local choice for the batch solver. |
| Apply the existing 3 mm limit to all active Newton contact pairs. Check actual joint overflow above 0.05 rad and nonfinite state at the same rate. Record simultaneous causes at the first event. | Existing model contract and user direction. |
| Check maximum left/right wrist position error at the control boundary. Use 0.25 m as a configurable first candidate and 0.02 m for the analytic sigmoid width in imagination. Report elbow and fingertip tracking in physical evaluation. | OpenArm §4.6 and local choice. The previous PD wrist maximum was 97.7 mm on Adjust Shelf Labels. |
| Treat clip completion as a normal task terminal state and an optional training step cap as external truncation. Report failure, completion and truncation independently. A safety event takes precedence when several endings coincide. | User direction and local API choice. |
| Fit fixed-duration dynamics and contact targets only on complete control intervals. A finite partial interval can supervise the one physical Termination Head. Combine its continuation with analytic tracking and clip-end continuation. | User direction. |
| Scale each of the 54 normalized actions by the available travel on its side of fixed Home. Record both travel vectors in provenance. Use the same servo targets for physics and learned rollouts. | User direction. |
| Allocate one eighth of each 4,096-environment batch to reference PD and one eighth to reference PD with Gaussian exploration of 0.12 in normalized action units. Use policy exploration in the rest. Keep those assignments across resets. | Local choice to maintain coverage of safe and near-contact transitions in the rolling replay. |
| Use the original single-clip PD and checkpoint evaluations as comparators. Check the first-event contract, paired 32-step continuation-weighted returns and complete Newton rollouts before a longer run. | User direction. |

## Motion Encoder upgrade

Upgrade base `74dbe367956e068ceed45548011e3ee0172e2ce3`. Keep the fixed-Home
54-joint target mapping and the Newton MJWarp task. Reuse the real-physics
collector and 8-step World Model / 32-step Actor training loop.

| Choice | Source |
| --- | --- |
| Encode ten consecutive future reference frames as a 32-dimensional latent. Each frame contributes its 54 joint angles and 54 joint velocities. Repeat the last reference frame at a clip boundary. | User direction and SuperMimic Appendix C. Per-frame feature choice is local. |
| The decoder uses ten chronological samples of joint angles relative to Home, joint velocities and previous bounded policy-space actions before servo mapping. Repeat the initial observation at reset. Rebuild deterministic future goals from the replay phase and preserve history through imagination. | User direction. History fields and reset padding are local choices. |
| Use the paper widths 2048, 1024, 512, 256 for encoder and decoder, with tanh on the latent and 54 action outputs. Preserve the existing Critic inputs. | Paper and local choice. |
| Compare World Model widths 384 and 2048 at 4,096 physical environments, using the same encoder, action mapping, multiscale reward, Termination Head, seed, replay and optimization budget. Isolate World Model initialization RNG to keep the other initial networks identical across widths. Run the 384-wide precision reward as a separate control. These tests use the one available clip. | User direction. Identical budget and the single-clip scope are local. |
| Compare Newton and learned one-step predictions from identical reference states and executed commands. Sample the first frame and up to 16 other frames ranked by the minimum inter-hand tip distance, with at least five frames between selected samples. Each sampled phase starts with reset-padded history. Use a probability threshold of 0.5 to count missed terminations, missed hard contacts and false alarms within this selected sample. | User direction. The diagnostic design is local. |
| Train one Termination Head on Newton's physical `terminated` signal using finite replay transitions and the existing importance-weighted binary loss. It takes current joint angles and velocities and the 54 executed servo targets. Use the full sigmoid probability from the first Actor update after World Model warmup. | User direction to include collision termination in the same head. State and executed target are local inputs chosen to predict failure over the whole control interval. |
| The multiscale reward retains the peak weight of each tracking term. Blend 75% of the existing exponential term and 25% of a wider exponential term for pose, joint angle and relative wrist error. Use the original reward as a named control. | Local choice based on measured Home-to-reference errors. |

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
- The legacy flat actor observes joint position and velocity, the next
  reference joint pose and velocity, a further preview pose, and its previous
  action. The World Model observes joint position, joint velocity, and the
  executed 54 servo targets.
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
  phase sampler clamps draws beyond the latest allowable reset to that phase.
  The flat-policy five-frame preview clamps to the last clip frame.
- The World Model's two contact labels report any penetration and depth above
  0.5 mm for reward shaping. Newton assigns the physical termination label from
  peak penetration across ten substeps and joint-limit or nonfinite checks at
  the control boundary. Nonfinite transitions are masked out of the training
  losses. Imagined continuation multiplies the probability of surviving
  physical termination by the deterministic clip-end mask. Imagination starts
  leave a full 32-step window. World Model holdout is by environment, on the same
  Motion Library. The reward uses position and matrix errors and penalizes
  changes in action. The previous residual modes also penalize residual magnitude.
- Actor imagination starts and observation normalization use the training
  environments. Evaluation reports both complete episode statistics and means
  over the shared prefix, and identifies the checkpoint, evaluated clip and
  nominal mode. Earlier flat checkpoints load the legacy state-based failure
  head from their provenance. The experimental three-contact checkpoints use
  their original commit `3147228` for evaluation.

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
