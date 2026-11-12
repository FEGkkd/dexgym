# Hand-centric Bimanual S0 build

| Rung | Acceptance |
| --- | --- |
| F0 | Verify the manifest and decode a genuine MCAP into aligned 50 Hz hand/body segments. |
| F1 | Generate hand-centric 54-joint segments and independently test their native MuJoCo geometry and physical replay. |
| F2 | Load the validated Motion Library, perform real batched MuJoCo collection and complete model-based Actor and World Model updates from random weights. |
| F3 | Train on multiple source episodes, evaluate on held-out episodes from perturbed starts, and export one native MuJoCo policy and measured videos. |

## Run record

- F0 passed. The 278 verified episodes decode to 854,348 aligned frames
  across three episode-disjoint splits. The witness decodes to 1,692 frames.
- F1 passed. Baseline hand-centric fitting produced 1,832 native-replayable
  windows. On the matched witness, wrist-first and hand-centric fingertip
  errors were 10.84 and 5.95 mm. Local refinement and physical contact
  repair reached 4.67 mm tip and 6.56 mm cross-tip error over the same frames.
  The repaired witness completed all 1,691 control steps with peak penetration
  0.42 mm and zero 2 ms substeps above 0.5 mm.
- F2 passed. A 64-environment MuJoCo Warp run completed 6,016 physical
  transitions and updated all four networks. The Actor started at random
  weights and learned from differentiable model rollout rewards.
