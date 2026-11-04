"""Gymnasium registration consumed by Isaac Lab's isaaclab.tasks entry point."""

import gymnasium as gym

DIRECT_ID = "DexGym-Joint-Tracking-Direct-v0"
MANAGER_ID = "DexGym-Joint-Tracking-Manager-v0"
TASK_IDS = (DIRECT_ID, MANAGER_ID)


gym.register(
    id=DIRECT_ID,
    entry_point="dexgym.tasks.direct.env:JointTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.env_cfg:JointTrackingDirectCfg",
        "rsl_rl_cfg_entry_point": "dexgym.tasks.ppo_cfg:JointTrackingPPOCfg",
        "default_agent": "rsl_rl",
    },
)
gym.register(
    id=MANAGER_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.manager_based.env_cfg:JointTrackingManagerCfg",
        "rsl_rl_cfg_entry_point": "dexgym.tasks.ppo_cfg:JointTrackingPPOCfg",
        "default_agent": "rsl_rl",
    },
)
