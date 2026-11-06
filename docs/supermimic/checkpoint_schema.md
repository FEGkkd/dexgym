# M4.2 checkpoint schema

`CheckpointManager` writes `m42_v1` records atomically. Each checkpoint contains:

- Algorithm, robot adapter and task adapter versions; state, action, observation and goal schemas.
- Reference and MJCF hashes, termination mode, full CLI/trainer configuration and recorded Git SHA.
- Actor, world-model and Critic parameters, including every registered RunningNorm mean, variance and count.
- All three optimizer states, optional learning-rate scheduler states, iteration and cumulative real/gradient update counts, model-trust report, seed and CPU/CUDA RNG state.
- Optional exact Ring Replay tensors and provenance. Learned termination defaults to `replay_saved=true`; analytic mode defaults to `false` to control checkpoint size.

Resume validates contracts and strictly loads network shapes. A saved Replay restores absolute timestamps, source, failure reason, and episode counters; the new Newton environments begin distinct episode IDs. If Replay is absent, the trainer recollects 32 real Newton steps and reestablishes model trust before Actor updates. Learned termination additionally collects 32 boundary-probe steps to avoid erasing its positive class. RNG and optimizer states are restored; the iteration and cumulative counters continue. A resumed run's `--iterations` means **additional** iterations. The new physics environment is reset, so resumed runs are reproducible in state but not a bitwise continuation of the previous physical episode.

`--lr-schedule constant|cosine` selects the schedule. `--schedule-total-iterations` is the maximum number of successful update-bearing iterations for each optimizer; individual schedules advance only after their own optimizer updates. `--actor-lr`, `--world-lr`, `--critic-lr`, and `--min-lr-fraction` are part of the resume contract. New runs default to constant LR to preserve prior M4.2 experiments.

`load_legacy_m41_actor` is explicit. It requires the pinned Wave reference SHA and constructs the historical Actor with legacy identity normalization and no goal-latent `tanh`. The old selected actor's action tensors were compared numerically at multiple Wave frames with exact equality before the full Newton regression. The Legacy action map remains `legacy_wave_v1`; v2 and v3 checkpoints cannot be silently loaded under it.

The complete run log is `training.jsonl` and `summary.json` beside the checkpoint. Store physical evaluation traces and independent validation reports under a separate experiment directory. The source/reference/asset hashes are intended to make those results reviewable; an untracked external NPZ should be copied with its hash when archiving the experiment.
