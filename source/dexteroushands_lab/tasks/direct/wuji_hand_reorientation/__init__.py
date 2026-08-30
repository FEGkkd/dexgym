"""Native DirectTask implementation for Wuji Hand reorientation."""


def register_task():
    import gymnasium as gym

    from . import agents

    task_id = "DexterousHands-Wuji-Hand-Reorientation-Direct-v0"
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point=f"{__name__}.wuji_hand_reorientation_env:WujiHandReorientationDirectEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.wuji_hand_reorientation_env_cfg:WujiHandReorientationDirectEnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandReorientationPPORunnerCfg",
            },
        )
