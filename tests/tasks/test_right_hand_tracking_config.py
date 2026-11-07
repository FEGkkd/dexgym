"""Static registration and PPO task contract before GPU smoke testing."""

import gymnasium as gym
from dexgym.assets.marvin_wuji.robot_cfg import NO_HAND_CONTACT_USD
from dexgym.tasks import RIGHT_HAND_ID
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg
from dexgym.tasks.ppo_cfg import RightHandTrackingPPOCfg


def test_right_hand_task_uses_reachable_library_and_twenty_actions():
    cfg = RightHandTrackingDirectCfg()
    assert cfg.action_space == 20
    assert cfg.observation_space == 105
    assert cfg.decimation == 4
    assert cfg.sim.dt == 1 / 120
    assert cfg.scene.robot.spawn.usd_path == str(NO_HAND_CONTACT_USD.resolve())
    assert cfg.subgoal_stride >= 1
    assert cfg.goal_tolerance_m > 0


def test_right_hand_task_registers_its_own_ppo_configuration():
    spec = gym.spec(RIGHT_HAND_ID)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":RightHandTrackingDirectCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":RightHandTrackingPPOCfg")
    ppo = RightHandTrackingPPOCfg()
    assert ppo.num_steps_per_env > 0
    assert ppo.max_iterations <= 20  # bounded workflow validation only
