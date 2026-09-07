from __future__ import annotations

import gymnasium as gym

from dexteroushands_lab.tasks.direct import allegro_hand_track


def test_allegro_task_registration_exposes_stable_entry_points():
    allegro_hand_track.register_task()
    spec = gym.spec(allegro_hand_track.TASK_ID)
    assert spec.entry_point.endswith(
        "allegro_hand_track_env:AllegroHandTrackDirectEnv"
    )
    assert spec.kwargs["env_cfg_entry_point"].endswith(
        "allegro_hand_track_env_cfg:AllegroHandTrackDirectEnvCfg"
    )
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        "agents.rsl_rl_ppo_cfg:AllegroHandTrackPPORunnerCfg"
    )
