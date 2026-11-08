"""Reusable standalone Wuji Hand 2.1 scene for Isaac Lab Newton probes."""

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from .robot_cfg import make_robot_cfg


@configclass
class WujiHand21SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = make_robot_cfg()


def make_newton_sim_cfg(dt: float = 1 / 500) -> SimulationCfg:
    if dt <= 0:
        raise ValueError("Physics time step must be positive")
    return SimulationCfg(
        dt=dt,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500, integrator="implicitfast"),
            use_cuda_graph=False,
        ),
    )
