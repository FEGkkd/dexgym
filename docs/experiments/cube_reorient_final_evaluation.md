# Native Wuji Hand 2 cube reorientation: trained-policy result

The task is implemented inside dexgym and a native policy has been trained from
random initialization with SuperMimic. Both evaluation seeds demonstrate
continuous multi-goal reorientation in real native physics. This establishes the
requested task port and working training pipeline; it does not establish equality
to the published checkpoint's performance or that every future task will learn.

Training used seed 42, 4096 environments and 8000 iterations, including 500
WM-only iterations, on Isaac Lab 3 / Newton MuJoCo Warp. No pretrained actor or
world model initialized this run. The fixed reference is
`2becb7380fed86171056f9bf4338b2a6a0c14b01`; training code was `2e238d5`.
The task and optimizer core have not changed since that training revision.

Each evaluation uses 64 environments and four complete episodes per environment,
deterministic actions, observation noise, startup/reset randomization, and
interval pushes with a fresh reference curriculum. These are evaluation seeds,
not independent training seeds. All 512 native episodes, including failures,
contribute to the following results.

| Policy | Evaluation seed | Goals/episode | >=1 goal fraction | >=3 goals fraction | Drop fraction | Mean angle (rad) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Native | 123 | 4.97656 | 0.79688 | 0.58203 | 0.63281 | 0.97476 |
| Native | 456 | 4.34766 | 0.76172 | 0.55859 | 0.62891 | 0.97961 |
| Published reference | 123 | 6.50391 | 0.86328 | 0.64453 | 0.66016 | 0.83461 |
| Published reference | 456 | 6.44141 | 0.81641 | 0.64453 | 0.67969 | 0.84803 |

Across both native evaluations: **2387 goals in 512 episodes**, 399 episodes
with at least one goal, 292 with at least three, and 323 drops. The maximum
was 16 goals in one episode. Mean goals/episode is **4.66211**, versus
**6.47266** for the published reference on the same two evaluation seeds.
The native policy has a lower drop fraction (63.09% versus 66.99%), but the
roughly 28% goal-count gap remains a material performance limitation.

The public checkpoint predates the pinned main source. Actor normalization,
critic depth and physical-context representation differ between that release
and current main; the native run follows the pinned main implementation. Also,
4096 environments with unchanged update batches reduce per-transition update
density relative to the public 1024-environment run. These are documented
configuration differences, not established explanations for the performance gap.
See [reference version audit](cube_reorient_reference_versions.md).

## Training and model evidence

[Final training audit](cube_reorient_validation/final_training_audit.json)
verifies 82,182,144 collected transitions, 80,000 WM updates, 80,000 failure-head
updates, 7500 actor updates and 150,000 critic minibatch updates. Recorded wall
time was 9426 seconds. All 800 logged scalar rows are finite; logs sample every
10 iterations. `model_final.pt` and `model_8000.pt` have identical module states.

Host checkpoint:
`/home/ubuntu/dexgym/artifacts/cube_reorient/train_seed42_scene_v2/model_final.pt`

SHA256: `7aceba2b59208393d0568bfc9f3aed86854142df4a5edbd89d303eff63642f97`

From the task worktree, reproduce evaluation with:

```bash
bash scripts/supermimic/hand_s0_host.sh -m dexgym.tasks.direct.hand2_cube_supermimic.evaluate --checkpoint /home/ubuntu/dexgym/artifacts/cube_reorient/train_seed42_scene_v2/model_final.pt --num-envs 64 --episodes-per-env 4 --seed 123 --output /tmp/cube_final_evaluation.json
```

Training, task ownership, native adaptations and a fresh-run command are in the
[task guide](../tasks/hand2_cube_supermimic.md). Per-episode evaluation JSONs and
checkpoint hashes are in `cube_reorient_validation/`.

## Recorded behavior

`native_iter8000_scene_v2.mp4` replays the first 60 seconds of environment 0 from
the primary final evaluation. Environment 0 is the fixed default selection,
not the best lane. All displayed frames remain inside its four scored episodes.
The trace includes drops/resets as they occur. It records up to nine completed
goals in a displayed episode; the 30-second preview shows six completed goals.
Root-position reconstruction error is zero. The video uses recorded native
poses and nominal visualization geometry, without physics resimulation.

The host video and raw trace are in `/home/ubuntu/dexgym/artifacts/cube_reorient/`.
The MP4 and preview were copied to the shared Windows workspace for direct viewing.
The [completion audit](cube_reorient_completion_audit.md) maps the requested
implementation and training requirements to their evidence and limitations.
