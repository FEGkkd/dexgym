"""Isaac Lab 3.0 Newton configuration for 30-Hz right-hand tracking.

Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/concepts/task_workflows.html
"""

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg

from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg

REPO_ROOT = Path(__file__).resolve().parents[5]


@configclass
class RightHandTrackingSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = make_robot_cfg()


@configclass
class RightHandTrackingPhysicsCfg(PresetCfg):
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=200), num_substeps=1, use_cuda_graph=False
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class RightHandTrackingDirectCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 8.0
    action_space = 20
    observation_space = 105  # q, qdot, current tips, target tips, tip error, previous action
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4, physics=RightHandTrackingPhysicsCfg())
    scene: RightHandTrackingSceneCfg = RightHandTrackingSceneCfg(num_envs=16, env_spacing=2.0)
    library_path: str = str(REPO_ROOT / "data/processed/hand_only_right/manifest.json")
    split: str = "train"
    subgoal_stride: int = 4
    goal_tolerance_m: float = 0.025
    goal_dwell_steps: int = 2
    max_goal_steps: int = 45
    action_delta_rad: float = 0.08
