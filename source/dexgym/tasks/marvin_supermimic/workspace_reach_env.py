"""Newton far-goal reach-and-hold task on the common Marvin arm servo."""

from isaaclab.utils import configclass

from .teleop_trajectory_env import MarvinTeleopWorkspaceCfg, MarvinTeleopWorkspaceEnv
from .workspace_reach_hold import (MarvinWorkspaceSE3ReachHold,
                                   WorkspaceReachRewardConfig,
                                   WorkspaceReachRewardConfigV5)


@configclass
class MarvinWorkspaceSE3ReachHoldCfg(MarvinTeleopWorkspaceCfg):
    episode_length_s = 12.0
    reach_success_semantics = "hold_start_deadline_v2"
    reach_temporal_context = None
    reach_reward_profile = "legacy_v1"
    reach_curriculum_version = ""


class MarvinWorkspaceSE3ReachHoldEnv(MarvinTeleopWorkspaceEnv):
    """Keep reference-bank reset mechanics; train a distinct arrival MDP."""

    cfg: MarvinWorkspaceSE3ReachHoldCfg

    def _make_task(self, reference, robot):
        profile = getattr(self.cfg, "reach_reward_profile", "legacy_v1")
        if profile not in ("legacy_v1", "hold_stationarity_v2"):
            raise ValueError(f"Unknown workspace reach reward profile: {profile}")
        return MarvinWorkspaceSE3ReachHold(
            reference, robot, success_semantics=self.cfg.reach_success_semantics,
            temporal_context=getattr(self.cfg, "reach_temporal_context", None),
            reward_config=(WorkspaceReachRewardConfigV5()
                           if profile == "hold_stationarity_v2" else
                           WorkspaceReachRewardConfig()))
