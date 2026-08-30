"""Native DirectTask implementation of Shadow Hand reorientation."""


def register_task():
    import gymnasium as gym

    from . import agents

    if "DexterousHands-Shadow-Hand-Reorientation-Direct-v0" not in gym.registry:
        gym.register(
            id="DexterousHands-Shadow-Hand-Reorientation-Direct-v0",
            entry_point=f"{__name__}.shadow_hand_reorientation_env:ShadowHandReorientationDirectEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.shadow_hand_reorientation_env_cfg:ShadowHandReorientationDirectEnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandReorientationPPORunnerCfg",
            },
        )
