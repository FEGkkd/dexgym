"""30 Hz right-hand tracking with the official standalone Wuji Hand 2.1."""

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg

from dexgym.assets.wuji_hand21.robot_cfg import make_robot_cfg

from .env_cfg import REPO_ROOT, RightHandTrackingDirectCfg


@configclass
class RightHandCleanSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = make_robot_cfg()


@configclass
class RightHand21PhysicsCfg(PresetCfg):
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500, integrator="implicitfast"),
        num_substeps=1,
        use_cuda_graph=False,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class RightHandCleanTrackingCfg(RightHandTrackingDirectCfg):
    """Keep the tracking contract while using the official Hand 2.1 model and GRAB library."""

    decimation = 16
    episode_length_s = 30.0
    goal_tolerance_m = 0.020
    action_delta_rad = 0.1
    action_deadzone = 0.1
    sim: SimulationCfg = SimulationCfg(dt=1 / 480, render_interval=16, physics=RightHand21PhysicsCfg())
    scene: RightHandCleanSceneCfg = RightHandCleanSceneCfg(num_envs=16, env_spacing=2.0)
    library_path: str = str(REPO_ROOT / "data/processed/hand21_right_geometry_feasible/manifest.json")
