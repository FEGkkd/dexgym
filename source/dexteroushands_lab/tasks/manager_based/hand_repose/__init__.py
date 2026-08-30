"""Manager-Based Allegro hand baseline."""


def register_task():
    import gymnasium as gym

    from . import agents

    if "DexterousHands-Allegro-Repose-ManagerBased-v0" not in gym.registry:
        gym.register(
            id="DexterousHands-Allegro-Repose-ManagerBased-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.hand_repose_env_cfg:ManagerBasedAllegroReposeEnvCfg",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ManagerBasedAllegroReposePPORunnerCfg",
            },
        )
