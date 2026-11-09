# Standalone Hand21 PPO task

Gym ID: `DexGym-Wuji-Hand21-TeleDexter-Direct-v0`.

The registered environment is `Hand21PPOEnv` in `env.py`. All task rules belong
to this directory. Its environment and configuration do not inherit from another
DexGym task. The official 20-DoF standalone hand runs at 480 Hz physics and
30 Hz control, with 15 mm fingertip subgoals and the existing GRAB library.

| Change | File |
| --- | --- |
| Threshold, dwell, curriculum, failure budgets, physics | `env_cfg.py` / `WujiHand21TeleDexterCfg` |
| Physical step, reward, termination, reset | `env.py` / `Hand21PPOEnv` |
| Goal advancement, cross-trajectory switching, counters | `subgoals.py` |
| Action mapping, observations, reward and timing helpers | `logic.py` |
| PPO network, optimizer, training hyperparameters | `ppo_cfg.py` / `WujiHand21TeleDexterPPOCfg` |

Use the existing Isaac Lab RSL-RL training entry with this Gym ID.
Changing this task's files does not change the separate
[SuperMimic task](../hand21_supermimic/README.md).
Robot asset files, motion-data loaders and generic algorithms remain shared infrastructure.
