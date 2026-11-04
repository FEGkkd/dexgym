"""Newton MJWarp direct configuration for Marvin local SE(3) reaching."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg

from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg


@configclass
class MarvinS0SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = make_robot_cfg()


@configclass
class MarvinS0PhysicsCfg(PresetCfg):
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), num_substeps=1, use_cuda_graph=False
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class MarvinBimanualS0Cfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = 3.0
    action_space = 14
    observation_space = 73
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=5, physics=MarvinS0PhysicsCfg())
    scene: MarvinS0SceneCfg = MarvinS0SceneCfg(num_envs=32, env_spacing=3.0, clone_in_fabric=True)
    goal_radius_rad = 0.12
    initial_radius_rad = 0.08
    bimanual_fraction = 0.0
    command_velocity_limit_rad_s = 0.8
    command_acceleration_limit_rad_s2 = 4.0
    command_tracking_lag_limit_rad = 0.05
    joint_limit_margin_rad = 0.03
    position_success_m = 0.005
    orientation_success_rad = 0.03490658503988659
    success_hold_s = 0.2
    controller_damping_scale = 0.5
    se3_coarse_weight = 4.0
    se3_fine_weight = 1.0
