"""Independent clean-bank observation and reward ablation configurations."""

from isaaclab.utils import configclass

from .clean_env_cfg import RightHandCleanTrackingCfg


@configclass
class RightHandCleanObs125Cfg(RightHandCleanTrackingCfg):
    """Expose the 20D accumulated PD target error without changing reward."""

    observation_space = 125
    include_command_error = True


@configclass
class RightHandCleanHybrid125Cfg(RightHandCleanObs125Cfg):
    """Make subgoal success dominate the dense tracking and time terms."""

    reward_dense_scale = 0.1
    reward_time_cost = 0.05
