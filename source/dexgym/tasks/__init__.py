"""Gymnasium registration consumed by Isaac Lab's isaaclab.tasks entry point."""

import gymnasium as gym

HAND2_CUBE_SUPERMIMIC_ID = "DexGym-Wuji-Hand2-Cube-Reorientation-SuperMimic-Direct-v0"

gym.register(
    id=HAND2_CUBE_SUPERMIMIC_ID,
    entry_point="dexgym.tasks.direct.hand2_cube_supermimic.env:Hand2CubeSuperMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.hand2_cube_supermimic.env_cfg:Hand2CubeSuperMimicCfg",
        "supermimic_cfg_entry_point": "dexgym.tasks.direct.hand2_cube_supermimic.supermimic_cfg:default_experiment",
    },
)

TASK_IDS = (HAND2_CUBE_SUPERMIMIC_ID,)
