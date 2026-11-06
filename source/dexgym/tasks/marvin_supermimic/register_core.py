"""Marvin plugin registration: all concrete imports remain outside Core."""

from .core_plugins import MarvinCoreRobot, MarvinDualTCPReach, MarvinMotionTracking


def register_marvin(registry):
    registry.register_robot("marvin_arm_only", MarvinCoreRobot)
    registry.register_task("marvin_motion_tracking", MarvinMotionTracking)
    registry.register_task("marvin_dual_tcp_reach", MarvinDualTCPReach)
