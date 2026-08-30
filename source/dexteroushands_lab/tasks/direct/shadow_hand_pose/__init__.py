"""Direct Shadow Hand architecture baseline."""


def register_task():
    import gymnasium as gym

    from . import agents

    if "DexterousHands-Shadow-Hand-Pose-Direct-v0" not in gym.registry:
        gym.register(
            id="DexterousHands-Shadow-Hand-Pose-Direct-v0",
            entry_point=f"{__name__}.shadow_hand_pose_env:DirectShadowHandPoseEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.shadow_hand_pose_env_cfg:DirectShadowHandPoseEnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DirectShadowHandPosePPORunnerCfg",
            },
        )
