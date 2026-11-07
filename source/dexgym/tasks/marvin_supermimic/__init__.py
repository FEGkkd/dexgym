"""Marvin bimanual SuperMimic task registration."""

import gymnasium as gym

TASK_ID = "DexGym-Marvin-SuperMimic-Wave-Direct-v0"
REACH_TASK_ID = "DexGym-Marvin-SuperMimic-Reach-Direct-v0"
SE3_REACH_HOLD_TASK_ID = "DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0"
CONTINUOUS_SE3_FOLLOW_TASK_ID = "DexGym-Marvin-SuperMimic-Continuous-SE3-Follow-Direct-v0"
TELEOP_WORKSPACE_TASK_ID = "DexGym-Marvin-SuperMimic-Teleop-Workspace-SE3-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point="dexgym.tasks.marvin_supermimic.env:MarvinWaveDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "dexgym.tasks.marvin_supermimic.env_cfg:MarvinWaveDirectCfg"},
)

gym.register(
    id=REACH_TASK_ID,
    entry_point="dexgym.tasks.marvin_supermimic.reach_env:MarvinReachDirectEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "dexgym.tasks.marvin_supermimic.reach_env:MarvinReachDirectCfg"},
)

gym.register(
    id=SE3_REACH_HOLD_TASK_ID,
    entry_point="dexgym.tasks.marvin_supermimic.se3_env:MarvinSE3ReachHoldEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "dexgym.tasks.marvin_supermimic.se3_env:MarvinSE3ReachHoldCfg"},
)

gym.register(
    id=CONTINUOUS_SE3_FOLLOW_TASK_ID,
    entry_point="dexgym.tasks.marvin_supermimic.se3_trajectory_env:MarvinContinuousSE3FollowEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point":
            "dexgym.tasks.marvin_supermimic.se3_trajectory_env:MarvinContinuousSE3FollowCfg"},
)

gym.register(
    id=TELEOP_WORKSPACE_TASK_ID,
    entry_point="dexgym.tasks.marvin_supermimic.teleop_trajectory_env:MarvinTeleopWorkspaceEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point":
            "dexgym.tasks.marvin_supermimic.teleop_trajectory_env:MarvinTeleopWorkspaceCfg"},
)
