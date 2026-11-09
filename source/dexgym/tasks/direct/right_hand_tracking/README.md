# Historical right-hand tracking tasks and import compatibility

The two current Hand21 tasks have separate implementations:

- [PPO](../hand21_ppo/README.md): `hand21_ppo/env.py` and `hand21_ppo/env_cfg.py`.
- [SuperMimic](../hand21_supermimic/README.md): `hand21_supermimic/env.py` and `hand21_supermimic/env_cfg.py`.

This directory retains historical tracking/ablation environments. Its old
`teledexter_env_cfg.py` and `supermimic_*.py` import paths forward to the new
packages for existing callers. Edit the new task directories to change either
current registered task.
