"""Gymnasium registration consumed by Isaac Lab's isaaclab.tasks entry point."""

import gymnasium as gym

DIRECT_ID = "DexGym-Joint-Tracking-Direct-v0"
MANAGER_ID = "DexGym-Joint-Tracking-Manager-v0"
MARVIN_SUPERMIMIC_WAVE_ID = "DexGym-Marvin-SuperMimic-Wave-Direct-v0"
RIGHT_HAND_ID = "DexGym-Wuji-Right-Hand-Tracking-Direct-v0"
RIGHT_HAND_CLEAN_ID = "DexGym-Wuji-Right-Hand-Clean-Tracking-Direct-v0"
RIGHT_HAND_OBS125_ID = "DexGym-Wuji-Right-Hand-Clean-Obs125-Direct-v0"
RIGHT_HAND_HYBRID125_ID = "DexGym-Wuji-Right-Hand-Clean-Hybrid125-Direct-v0"
RIGHT_HAND_HYBRID125_ENTROPY0_ID = "DexGym-Wuji-Right-Hand-Clean-Hybrid125-Entropy0-Direct-v0"
RIGHT_HAND_HYBRID125_TANH_GAUSSIAN_ID = "DexGym-Wuji-Right-Hand-Clean-Hybrid125-TanhGaussian-Direct-v0"
RIGHT_HAND_HYBRID125_BETA_ID = "DexGym-Wuji-Right-Hand-Clean-Hybrid125-Beta-Direct-v0"
HAND21_TELEDEXTER_ID = "DexGym-Wuji-Hand21-TeleDexter-Direct-v0"
HAND21_SUPERMIMIC_ID = "DexGym-Wuji-Hand21-TeleDexter-SuperMimic-Direct-v0"
CARTPOLE_TASK_IDS = (DIRECT_ID, MANAGER_ID)


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

gym.register(
    id=RIGHT_HAND_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.env_cfg:RightHandTrackingDirectCfg",
    },
)

gym.register(
    id=RIGHT_HAND_CLEAN_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.clean_env_cfg:RightHandCleanTrackingCfg",
    },
)

gym.register(
    id=RIGHT_HAND_OBS125_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.ablation_cfg:RightHandCleanObs125Cfg",
    },
)

gym.register(
    id=RIGHT_HAND_HYBRID125_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.ablation_cfg:RightHandCleanHybrid125Cfg",
    },
)

gym.register(
    id=RIGHT_HAND_HYBRID125_ENTROPY0_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.ablation_cfg:RightHandCleanHybrid125Cfg",
    },
)

gym.register(
    id=RIGHT_HAND_HYBRID125_TANH_GAUSSIAN_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.ablation_cfg:RightHandCleanHybrid125Cfg",
    },
)

gym.register(
    id=RIGHT_HAND_HYBRID125_BETA_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.ablation_cfg:RightHandCleanHybrid125Cfg",
    },
)

gym.register(
    id=HAND21_TELEDEXTER_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg:WujiHand21TeleDexterCfg",
        "rsl_rl_cfg_entry_point": "dexgym.tasks.ppo_cfg:WujiHand21TeleDexterPPOCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id=HAND21_SUPERMIMIC_ID,
    entry_point="dexgym.tasks.direct.right_hand_tracking.env:RightHandTrackingDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg:WujiHand21SuperMimicCfg",
        "supermimic_cfg_entry_point": "dexgym.tasks.direct.right_hand_tracking.supermimic_cfg:default_experiment",
    },
)

from . import marvin_supermimic as _marvin_supermimic  # noqa: E402, F401

TASK_IDS = CARTPOLE_TASK_IDS + (
    RIGHT_HAND_ID,
    RIGHT_HAND_CLEAN_ID,
    RIGHT_HAND_OBS125_ID,
    RIGHT_HAND_HYBRID125_ID,
    RIGHT_HAND_HYBRID125_ENTROPY0_ID,
    RIGHT_HAND_HYBRID125_TANH_GAUSSIAN_ID,
    RIGHT_HAND_HYBRID125_BETA_ID,
    HAND21_TELEDEXTER_ID,
    HAND21_SUPERMIMIC_ID,
    _marvin_supermimic.TASK_ID,
    _marvin_supermimic.REACH_TASK_ID,
    _marvin_supermimic.SE3_REACH_HOLD_TASK_ID,
)
