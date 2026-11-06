"""Marvin plugin registration: all concrete imports remain outside Core."""

from .core_plugins import MarvinCoreRobot, MarvinDualTCPReach, MarvinMotionTracking
from .se3_reach_hold import MarvinBimanualSE3ReachHold


def register_marvin(registry):
    registry.register_robot("marvin_arm_only", MarvinCoreRobot)
    registry.register_task("marvin_motion_tracking", MarvinMotionTracking)
    registry.register_task("marvin_dual_tcp_reach", MarvinDualTCPReach)
    registry.register_task("marvin_bimanual_se3_reach_hold", MarvinBimanualSE3ReachHold)
