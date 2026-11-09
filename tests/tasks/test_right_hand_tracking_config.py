"""Static registration and PPO task contract before GPU smoke testing."""

from dataclasses import asdict

import gymnasium as gym
import torch
from dexgym.assets.marvin_wuji.robot_cfg import PACKAGED_USD
from dexgym.assets.wuji_hand21 import OFFICIAL_USD
from dexgym.tasks import (
    HAND21_TELEDEXTER_ID,
    RIGHT_HAND_CLEAN_ID,
    RIGHT_HAND_HYBRID125_BETA_ID,
    RIGHT_HAND_HYBRID125_ENTROPY0_ID,
    RIGHT_HAND_HYBRID125_ID,
    RIGHT_HAND_HYBRID125_TANH_GAUSSIAN_ID,
    RIGHT_HAND_ID,
    RIGHT_HAND_OBS125_ID,
)
from dexgym.tasks.direct.right_hand_tracking.ablation_cfg import RightHandCleanHybrid125Cfg, RightHandCleanObs125Cfg
from dexgym.tasks.direct.right_hand_tracking.clean_env_cfg import RightHandCleanTrackingCfg
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg
from dexgym.tasks.direct.right_hand_tracking.logic import tracking_step
from dexgym.tasks.direct.right_hand_tracking.teledexter_env_cfg import WujiHand21TeleDexterCfg
from dexgym.tasks.ppo_cfg import (
    RightHandCleanHybrid125BetaPPOCfg,
    RightHandCleanHybrid125Entropy0PPOCfg,
    RightHandCleanHybrid125PPOCfg,
    RightHandCleanHybrid125TanhGaussianPPOCfg,
    RightHandCleanObs125PPOCfg,
    RightHandCleanTrackingPPOCfg,
    RightHandTrackingPPOCfg,
    WujiHand21TeleDexterPPOCfg,
)


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


def test_legacy_right_hand_environment_is_not_an_active_ppo_task():
    spec = gym.spec(RIGHT_HAND_ID)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":RightHandTrackingDirectCfg")
    assert "rsl_rl_cfg_entry_point" not in spec.kwargs


def test_clean_task_has_independent_30hz_full_contact_configuration():
    clean = RightHandCleanTrackingCfg()
    old = RightHandTrackingDirectCfg()
    assert clean.library_path.endswith("hand21_right_geometry_feasible/manifest.json")
    assert clean.scene.robot.spawn.usd_path == str(OFFICIAL_USD)
    assert clean.scene.robot.spawn.variants == {"Physics": "PhysX"}
    assert clean.sim.physics.newton_mjwarp.solver_cfg.integrator == "implicitfast"
    assert (clean.action_delta_rad, clean.action_deadzone) == (0.1, 0.1)
    assert (old.action_delta_rad, old.action_deadzone) == (0.08, 0.0)
    assert clean.sim.dt == 1 / 480
    assert clean.decimation == 16
    assert clean.sim.dt * clean.decimation == 1 / 30
    assert (clean.action_space, clean.observation_space) == (20, 105)
    assert (
        clean.subgoal_stride,
        clean.goal_tolerance_m,
        clean.goal_dwell_steps,
        clean.max_goal_steps,
    ) == (4, 0.02, 2, 45)
    assert clean.episode_length_s == 30.0  # allow long clean clips to finish
    assert old.sim.dt == 1 / 120
    assert old.library_path != clean.library_path
    assert old.scene.robot.spawn.usd_path != clean.scene.robot.spawn.usd_path


def test_clean_baseline_is_diagnostic_only():
    spec = gym.spec(RIGHT_HAND_CLEAN_ID)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":RightHandCleanTrackingCfg")
    assert "rsl_rl_cfg_entry_point" not in spec.kwargs
    assert RightHandCleanTrackingPPOCfg().experiment_name != RightHandTrackingPPOCfg().experiment_name
    assert "deadzone" in RightHandCleanTrackingPPOCfg().experiment_name


def test_observation_and_reward_ablations_are_isolated_from_clean_baseline():
    baseline = RightHandCleanTrackingCfg()
    obs = RightHandCleanObs125Cfg()
    hybrid = RightHandCleanHybrid125Cfg()
    assert (baseline.observation_space, obs.observation_space, hybrid.observation_space) == (105, 125, 125)
    assert not baseline.include_command_error
    assert obs.include_command_error and hybrid.include_command_error
    assert (baseline.reward_dense_scale, baseline.reward_time_cost) == (1.0, 0.01)
    assert (obs.reward_dense_scale, obs.reward_time_cost) == (1.0, 0.01)
    assert (hybrid.reward_dense_scale, hybrid.reward_time_cost) == (0.1, 0.05)
    assert obs.library_path == hybrid.library_path == baseline.library_path
    assert obs.scene.robot.spawn.usd_path == hybrid.scene.robot.spawn.usd_path == baseline.scene.robot.spawn.usd_path
    assert obs.sim.dt == hybrid.sim.dt == baseline.sim.dt == 1 / 480
    for task_id, cfg_name in (
        (RIGHT_HAND_OBS125_ID, "RightHandCleanObs125Cfg"),
        (RIGHT_HAND_HYBRID125_ID, "RightHandCleanHybrid125Cfg"),
    ):
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"].endswith(f":{cfg_name}")
        assert "rsl_rl_cfg_entry_point" not in spec.kwargs
    names = {
        RightHandCleanTrackingPPOCfg().experiment_name,
        RightHandCleanObs125PPOCfg().experiment_name,
        RightHandCleanHybrid125PPOCfg().experiment_name,
    }
    assert len(names) == 3


def test_entropy_zero_variant_changes_only_ppo_entropy_coefficient():
    baseline = RightHandCleanHybrid125PPOCfg()
    no_entropy = RightHandCleanHybrid125Entropy0PPOCfg()
    baseline_algorithm = asdict(baseline.algorithm)
    no_entropy_algorithm = asdict(no_entropy.algorithm)
    baseline_algorithm["entropy_coef"] = 0.0
    assert no_entropy_algorithm == baseline_algorithm
    assert baseline.algorithm.entropy_coef == 0.005
    assert no_entropy.experiment_name != baseline.experiment_name
    spec = gym.spec(RIGHT_HAND_HYBRID125_ENTROPY0_ID)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":RightHandCleanHybrid125Cfg")
    assert "rsl_rl_cfg_entry_point" not in spec.kwargs


def test_bounded_distribution_pilots_keep_the_hybrid_task_and_zero_entropy():
    entropy0 = RightHandCleanHybrid125Entropy0PPOCfg()
    for task_id, ppo_class, class_name in (
        (
            RIGHT_HAND_HYBRID125_TANH_GAUSSIAN_ID,
            RightHandCleanHybrid125TanhGaussianPPOCfg,
            "dexgym.algorithms.action_distributions:TanhGaussianDistribution",
        ),
        (
            RIGHT_HAND_HYBRID125_BETA_ID,
            RightHandCleanHybrid125BetaPPOCfg,
            "dexgym.algorithms.action_distributions:MatchedBetaDistribution",
        ),
    ):
        ppo = ppo_class()
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"].endswith(":RightHandCleanHybrid125Cfg")
        assert ppo.algorithm.entropy_coef == 0.0
        assert asdict(ppo.algorithm) == asdict(entropy0.algorithm)
        assert ppo.actor.distribution_cfg.class_name == class_name
        assert ppo.experiment_name != entropy0.experiment_name
        assert "rsl_rl_cfg_entry_point" not in spec.kwargs


def test_current_hand21_teledexter_ppo_contract_uses_15mm():
    spec = gym.spec(HAND21_TELEDEXTER_ID)
    assert spec.kwargs["env_cfg_entry_point"].endswith(":WujiHand21TeleDexterCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":WujiHand21TeleDexterPPOCfg")
    cfg = WujiHand21TeleDexterCfg()
    ppo = WujiHand21TeleDexterPPOCfg()
    assert cfg.scene.robot.spawn.usd_path == str(OFFICIAL_USD)
    assert cfg.library_path.endswith("hand21_right_geometry_feasible/manifest.json")
    assert cfg.teledexter_style and cfg.goal_tolerance_m == 0.015
    assert (cfg.observation_space, cfg.action_space) == (125, 20)
    assert (cfg.dwell_min_steps, cfg.dwell_max_steps) == (1, 1)
    assert (cfg.curriculum_stride_min, cfg.curriculum_stride_max) == (1, 1)
    assert cfg.cross_trajectory_probability == 0.0
    assert "frame_tracking" in ppo.experiment_name
    assert ppo.actor.distribution_cfg.class_name == "dexgym.algorithms.action_distributions:TanhGaussianDistribution"
    assert ppo.actor.distribution_cfg.init_std == 1.0
    assert ppo.algorithm.entropy_coef == 0.005
