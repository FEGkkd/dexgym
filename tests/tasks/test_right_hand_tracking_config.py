"""Static registration and PPO task contract before GPU smoke testing."""

import gymnasium as gym
import torch
from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_USD
from dexgym.tasks import RIGHT_HAND_ID
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg
from dexgym.tasks.direct.right_hand_tracking.logic import tracking_step
from dexgym.tasks.ppo_cfg import RightHandTrackingPPOCfg


def test_right_hand_task_uses_reachable_library_and_twenty_actions():
    cfg = RightHandTrackingDirectCfg()
    assert cfg.action_space == 20
    assert cfg.observation_space == 105
    assert cfg.decimation == 4
    assert cfg.sim.dt == 1 / 120
    assert cfg.scene.robot.spawn.usd_path == str(PACKAGED_USD.resolve())
    assert cfg.subgoal_stride >= 1
    assert cfg.goal_tolerance_m > 0


def test_default_goal_requires_precise_five_finger_tracking():
    cfg = RightHandTrackingDirectCfg()
    current = torch.zeros((2, 5, 3))
    current[0, 0, 0] = 0.009
    current[1, 0, 0] = 0.011
    _, reached, _, _ = tracking_step(
        current,
        torch.zeros_like(current),
        torch.ones(2, dtype=torch.long),
        torch.zeros((2, 20)),
        tolerance_m=cfg.goal_tolerance_m,
        dwell_steps=cfg.goal_dwell_steps,
    )
    assert reached.tolist() == [True, False]


def test_right_hand_task_registers_its_own_ppo_configuration():
    spec = gym.spec(RIGHT_HAND_ID)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":RightHandTrackingDirectCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":RightHandTrackingPPOCfg")
    ppo = RightHandTrackingPPOCfg()
    assert ppo.num_steps_per_env > 0
    assert ppo.max_iterations <= 20  # bounded workflow validation only
