"""Marvin bimanual SuperMimic task registration."""

import gymnasium as gym

TASK_ID = "DexGym-Marvin-SuperMimic-Wave-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point="dexgym.tasks.marvin_supermimic.env:MarvinWaveDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "dexgym.tasks.marvin_supermimic.env_cfg:MarvinWaveDirectCfg"},
)
