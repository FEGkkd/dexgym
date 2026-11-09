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
| Native iteration 4000 | 0.03125 | 0.03125 | 0 | 0.08984375 | 1.73836 |
| Native iteration 6000 | 0.71875 | 0.4296875 | 0.0703125 | 0.421875 | 1.22332 |
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

At iteration 4000, evaluation completed eight goals across 256 episodes, in eight
distinct episodes (3.125% with any goal). No episode completed more than one goal.
Mean orientation error was 1.73836 rad, and drop fraction was 0.08984375.
This is a small improvement over iteration 2000, but it does not demonstrate
continuous reorientation. The 4096-environment training run continues unchanged
to iteration 8000; the checkpoint watcher is now waiting for that checkpoint.

At iteration 6000, the same 256-episode protocol completed 184 goals: 110
nonzero-success episodes (42.96875%), 18 episodes with at least three goals,
and a maximum of five goals in one episode. Mean goals/episode rose from
0.03125 at iteration 4000 to 0.71875. Mean orientation error fell to 1.22332 rad,
but 108 episodes ended in a drop (42.1875%). This is evidence of emerging
multi-goal reorientation, with substantial instability and a remaining gap to
the published actor diagnostic (6.50391 goals/episode, 86.3281% with any goal).
The native run remains active and unchanged; the 8000-iteration evaluation and
pose recording are still queued. This intermediate result does not complete the
full task-reproduction goal.

A separate four-environment, one-episode recording at iteration 6000 completed
one goal in environment 2; environments 0 and 1 dropped after 6.98 and 8.78 s.
Environment 3 survived 50 s without a goal. Two videos accompany the raw trace:
`native_iter6000_env0_drop.mp4` shows the predetermined environment 0 up to
6.96 s (before its episode ends), and
`native_iter6000_env2_selected_success.mp4` shows environment 2 for 50 s.
The latter is explicitly selected after observing success and is not a
representative success-rate estimate. Both reconstruct recorded native poses
with zero root-position error; they do not re-simulate physics. The 256-episode
result above remains the performance estimate, including failures.
