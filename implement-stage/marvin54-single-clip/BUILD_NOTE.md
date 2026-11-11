# Build record

| Rung | Acceptance |
| --- | --- |
| F0 | Compose updated arm and existing Hand2.2 with 54 valid position servos and run a full native MuJoCo baseline. |
| F1 | Run Actor initialization and evaluate the closed-loop checkpoint with no online teacher in both engines. |
| F2 | Train with Newton, World Model, Termination Head, Actor and Critic, then compare full-clip checkpoints. |
| F3 | Deliver a selected RL checkpoint, sim-to-sim replay, measured trace and reproducible invocation. |

## Run record

- F0 passed. The generated paired asset compiled with 54 joints and 54 drives.
  Native MuJoCo reference PD completed 1,691 intervals.
- F1 passed. The behavior-cloned Actor completed 1,691 intervals in both
  Newton and native MuJoCo. The reference teacher is absent from both policy
  inference calls.
- F2 short-run checks passed. The full Actor recipe reached physical model-based
  checkpoint evaluation. The local-dynamics variant completed an 8-iteration
  integration smoke with all four optimizers updated. The 4,096-environment
  aligned-asset check and 42 tests passed.
- F3 comparison, reset-perturbation runs and deployment export use the full
  physical runs tracked in the experiment output directory.

## Final physical checks

- The 2,000-iteration full-Actor run and the 1,600-iteration local-dynamics
  run completed at 4,096 Newton environments. The selected local checkpoint
  is iteration 1,000 with 680 model-based Actor optimizer updates.
- F3 passed. The selected checkpoint and its compact exported Actor completed
  all 1,691 steps in both engines. Each engine also completed the three paired
  perturbed initial-state trials. MuJoCo full-clip policy mean wrist and tip
  errors were 2.70 and 3.63 mm. Newton errors were 5.15 and 8.72 mm.
- Full-clip scoring and ranking used the supplied clip. The MuJoCo teacher
  contributes to initialization, so native deployment assesses the paired
  simulation stack on the trained motion.
- The safety head missed 12 of 12 physically terminated, high-action-noise
  probes at a diagnostic probability cutoff of 0.5. The physics tests and the
  native replay use their measured first-event termination conditions.
- Artifacts are in `/home/ubuntu/dexgym_data/marvin54_aris_single_clip_20261112`.
  Run `bash deploy/replay.sh` there for a new native MuJoCo evaluation.
