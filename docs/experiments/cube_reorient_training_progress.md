# Native cube reorientation training: scene revision 2

The full task-reproduction goal remains incomplete. Published upstream weights are
used only as a diagnostic; native training starts from random initialization.

Run: `/home/ubuntu/dexgym/artifacts/cube_reorient/train_seed42_scene_v2`.
Training code: `2e238d5`, seed 42, 4096 environments, 8000 configured iterations,
500 WM-only iterations. Optimizer batches and schedules follow the pinned reference.
The earlier `train_seed42` run is superseded because its scene lacked the ground.

Evaluation: seed 123, 64 environments, four complete episodes per environment,
deterministic actions, observation noise, startup/reset randomization, interval pushes
with the same fresh reference curriculum. Every environment contributes four episodes.

| Policy | Goals/episode | >=1 goal fraction | >=3 goals fraction | Drop fraction | Mean angle (rad) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untrained | 0 | 0 | 0 | 0.0234375 | 2.22934 |
| Native iteration 1000 | 0.00390625 | 0.00390625 | 0 | 0.03515625 | 2.04405 |
| Native iteration 2000 | 0.00390625 | 0.00390625 | 0 | 0.09375 | 1.88495 |
| Published upstream actor, iteration 8000 | 6.50391 | 0.863281 | 0.644531 | 0.660156 | 0.834614 |

At iteration 1000 the native actor has had 500 policy updates after WM warmup.
One completed goal across 256 episodes is not evidence that it has learned the task.
The run continues unchanged. Independent evaluations at iterations 2000, 4000 and
8000 are queued against this specific running process. The final checkpoint evaluation
also records native poses for behavior playback.

Machine-readable per-episode results are in `cube_reorient_validation/`.

At iteration 2000 the native actor still completed only one goal across 256 episodes.
Mean orientation error improved from 2.04405 to 1.88495 rad, while drop fraction
increased from 0.03515625 to 0.09375. This remains insufficient learned behavior.
Training continues with the same reference recipe; the next scheduled evaluation
is iteration 4000. No mechanism or task simplification was added in response to this result.

A separate behavior recording at iteration 2000 uses seed 123, four environments
and one complete episode per environment. All four lasted 50 seconds without a
drop and completed zero goals (mean angle 1.78154 rad). This small recording is
not a replacement for the 256-episode evaluation above. The rendered video uses
environment 0 selected before inspection, the first 30 seconds at 25 fps, and
recorded native poses with no physics resimulation. Root-position reconstruction
error is zero. Raw trace and video are under the host artifact directory as
`native_iter2000_scene_v2_trace.npz` and `native_iter2000_scene_v2.mp4`.
Reports are `native_iter2000_trace_eval.json` and
`native_iter2000_trace_render.json` in the validation directory.
