# Standalone Hand21 PPO task

Gym ID: `DexGym-Wuji-Hand21-TeleDexter-Direct-v0`.

The registered environment is `Hand21PPOEnv` in `env.py`. All task rules belong
to this directory. Its environment and configuration do not inherit from another
DexGym task. The official 20-DoF standalone hand runs at 480 Hz physics and
30 Hz control, with one 15 mm fingertip target for every consecutive GRAB frame.
After each physics step the current reference frame is scored and the next
frame becomes the observation target, whether or not the current frame was
reached. There is no random span, multi-step dwell, or mid-segment trajectory
switch. A segment ends after its last frame is scored; 46 consecutive missed
frames also terminate because the failure budget is 45.

| Change | File |
| --- | --- |
| Threshold, fixed one-frame stride, failure budget, physics | `env_cfg.py` / `WujiHand21TeleDexterCfg` |
| Physical step, reward, termination, reset | `env.py` / `Hand21PPOEnv` |
| Per-frame advancement, completion, counters | `subgoals.py` |
| Action mapping, observations, reward and timing helpers | `logic.py` |
| PPO network, optimizer, training hyperparameters | `ppo_cfg.py` / `WujiHand21TeleDexterPPOCfg` |

Use the existing Isaac Lab RSL-RL training entry with this Gym ID.
Changing this task's files does not change the separate
[SuperMimic task](../hand21_supermimic/README.md).
Robot asset files, motion-data loaders and generic algorithms remain shared infrastructure.
