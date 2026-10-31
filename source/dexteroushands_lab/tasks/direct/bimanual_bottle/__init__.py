"""Shared bimanual bottle/cap DirectRLEnv task registration."""


def register_tasks():
    import gymnasium as gym
    from . import agents

    common = f"{__name__}.bimanual_bottle_env:BimanualBottleDirectEnv"
    ppo = f"{agents.__name__}.rsl_rl_ppo_cfg:BimanualBottlePPORunnerCfg"
    for task, config in (
        ("GraspHold", "GraspHoldDirectEnvCfg"),
        ("CapTurn30", "CapTurn30DirectEnvCfg"),
    ):
        task_id = f"DexterousHands-WujiHand2-Bimanual-{task}-Direct-v0"
        if task_id not in gym.registry:
            gym.register(
                id=task_id, entry_point=common, disable_env_checker=True,
                kwargs={"env_cfg_entry_point": f"{__name__}.bimanual_bottle_env_cfg:{config}",
                        "rsl_rl_cfg_entry_point": ppo},
            )
