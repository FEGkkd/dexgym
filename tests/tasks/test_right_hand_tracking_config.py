"""Static registration and PPO task contract before GPU smoke testing."""

import gymnasium as gym
import torch
from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_USD
from dexgym.tasks import RIGHT_HAND_CLEAN_ID, RIGHT_HAND_ID
from dexgym.tasks.direct.right_hand_tracking.clean_env_cfg import RightHandCleanTrackingCfg
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg
from dexgym.tasks.direct.right_hand_tracking.logic import tracking_step
from dexgym.tasks.ppo_cfg import RightHandCleanTrackingPPOCfg, RightHandTrackingPPOCfg


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


def test_clean_task_has_independent_30hz_full_contact_configuration():
    clean = RightHandCleanTrackingCfg()
    old = RightHandTrackingDirectCfg()
    assert clean.library_path.endswith("hand_only_right_collision_clean/manifest.json")
    assert clean.scene.robot.spawn.usd_path.endswith("hand_only_drive_corrected_hand_x2.usda")
    assert clean.sim.dt == 1 / 480
    assert clean.decimation == 16
    assert clean.sim.dt * clean.decimation == 1 / 30
    assert (clean.action_space, clean.observation_space) == (20, 105)
    assert (
        clean.subgoal_stride,
        clean.goal_tolerance_m,
        clean.goal_dwell_steps,
        clean.max_goal_steps,
    ) == (4, 0.01, 2, 45)
    assert clean.episode_length_s == 30.0  # allow long clean clips to finish
    assert old.sim.dt == 1 / 120
    assert old.library_path != clean.library_path
    assert old.scene.robot.spawn.usd_path != clean.scene.robot.spawn.usd_path


def test_clean_task_has_separate_gym_id_and_ppo_log_directory():
    spec = gym.spec(RIGHT_HAND_CLEAN_ID)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":RightHandCleanTrackingCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":RightHandCleanTrackingPPOCfg")
    assert RightHandCleanTrackingPPOCfg().experiment_name != RightHandTrackingPPOCfg().experiment_name
