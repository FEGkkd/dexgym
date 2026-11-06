# Training resume and independent evaluation

`CoreCheckpoint.load()` remains a strict training resume: Robot/Task/TrainerConfig, full `physics_config`, reference hashes, network and normalization semantics, optimizer/scheduler schemas and resume mode must match. It restores optimizer and optional Replay state, and invalidates stale Model Trust. Newton uses `fresh_physics` unless exact collector restoration is available.

`CoreCheckpoint.load_actor_for_evaluation()` is inference-only. It verifies the stored v2 manifest SHA and rejects a changed robot asset, action/controller contract, state/goal/Actor schema, normalizer semantics, reward/termination task contract, backend, physical/policy dt, hand contact configuration or episode semantics. It loads the Actor and its saved normalizer buffers with `strict=True` and sets evaluation mode; it does not load optimizers or Replay.

Evaluation may explicitly change `num_envs`, `seed`, reset perturbation, random reference start, reference start frame, observation noise mode, episode count and trace path. The loader returns the effective protocol and every override; the independent evaluator writes the protocol into `report.json`. A changed `physics_config` remains illegal for training resume even if the field is permitted for inference.

The v2 checkpoint file is read as-is. There is no silent rewrite or shape-only migration. M4.1 still uses the explicit historical legacy evaluator; M4.3.1 Core v2 uses this inference path.

Executed example: one-env evaluation of a 16-env Reach checkpoint with a new seed:

```bash
PYTHONPATH=/home/ubuntu/dexgym_data/marvin_supermimic_m42/runtime_overrides:source /home/ubuntu/isaacsim-6.0.1/bin/python scripts/supermimic/isaac_runtime.py evaluate_core_m431 --checkpoint /home/ubuntu/dexgym_data/marvin_supermimic_m431/reach_training/reach_m43_smoke.pt --reference /home/ubuntu/dexgym_data/marvin_supermimic_m42/second_task/procedural_bimanual_reach_v1.npz --num-envs 1 --seed 987 --output /tmp/m432_reach_independent
```

For Tracking, `--reference-start-frame 0 --reset-perturb-rad 0.0` gives a full Wave start. The evaluator follows the real hard termination, so a weak policy may end before the clip. `--episode-count 2` runs and records two independent Episodes per environment; this was verified with a 200-step one-env Reach run. `--noisy-observation` uses the checkpoint's saved Actor noise distribution and a new evaluation seed; a checkpoint without a noise contract rejects that flag. `--no-trace` omits the physical trace. Evaluation never adds exploration action noise. Episode records are isolated after every physical reset.
