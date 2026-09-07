"""DexTrack Allegro Hand tracking DirectTask package."""


TASK_ID = "DexterousHands-Allegro-Hand-Track-Direct-v0"


def register_task() -> None:
    """Register the task and its RSL-RL config entry point once."""

    import gymnasium as gym

    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.allegro_hand_track_env:AllegroHandTrackDirectEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    f"{__name__}.allegro_hand_track_env_cfg:"
                    "AllegroHandTrackDirectEnvCfg"
                ),
                "rsl_rl_cfg_entry_point": (
                    f"{__name__}.agents.rsl_rl_ppo_cfg:"
                    "AllegroHandTrackPPORunnerCfg"
                ),
            },
        )
