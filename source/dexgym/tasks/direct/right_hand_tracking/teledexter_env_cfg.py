"""Hand-only TELEDEXTER controller on official Wuji Hand 2.1 and full GRAB data."""

from isaaclab.utils import configclass

from .clean_env_cfg import RightHandCleanTrackingCfg


@configclass
class WujiHand21TeleDexterCfg(RightHandCleanTrackingCfg):
    """Use paper-style subgoal curriculum and reset logic at 30 Hz."""

    goal_tolerance_m = 0.015
    observation_space = 125
    include_command_error = True
    teledexter_style = True
    curriculum_stride_min: int = 20  # 40 frames at the paper's 60-Hz reference rate
    curriculum_stride_max: int = 40  # 80 frames at the paper's 60-Hz reference rate
    curriculum_anneal_steps: int = 25600
    curriculum_sigma_min: float = 0.7
    reference_start_fraction: float = 0.9
    dwell_min_steps: int = 5
    dwell_max_steps: int = 15
    cross_trajectory_probability: float = 0.1
    failure_tolerance_scale: float = 1.5
    cross_trajectory_failure_budget: int = 300
    reward_dense_scale = 0.1
    reward_time_cost = 0.1
    reward_action_rate_penalty = 0.0
