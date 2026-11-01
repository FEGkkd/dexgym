"""Unified pregrasp-to-cap-turn task."""


def register_tasks():
    import gymnasium as gym

    task_id = "DexterousHands-WujiHand2-Bimanual-PregraspCapTurn30-Direct-v0"
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point=f"{__name__}.env:PregraspCapTurn30DirectEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:PregraspCapTurn30DirectEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UnifiedPPORunnerCfg",
            },
        )
