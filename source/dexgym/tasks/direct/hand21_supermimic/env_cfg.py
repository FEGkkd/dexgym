"""Self-contained standalone Hand21 task rules and physical configuration."""

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg

from dexgym.assets.wuji_hand21.robot_cfg import make_robot_cfg

REPO_ROOT = Path(__file__).resolve().parents[5]


@configclass
class Hand21SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = make_robot_cfg()


@configclass
class Hand21PhysicsCfg(PresetCfg):
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500, integrator="implicitfast"),
        num_substeps=1,
        use_cuda_graph=False,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class WujiHand21SuperMimicCfg(DirectRLEnvCfg):
    decimation = 16
    episode_length_s = 30.0
    action_space = 20
    observation_space = 125
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 480, render_interval=16, physics=Hand21PhysicsCfg())
    scene: Hand21SceneCfg = Hand21SceneCfg(num_envs=16, env_spacing=2.0)
    library_path: str = str(REPO_ROOT / "data/processed/hand21_right_geometry_feasible/manifest.json")
    split: str = "train"
    subgoal_stride: int = 4
    goal_tolerance_m = 0.015
    goal_dwell_steps: int = 2
    max_goal_steps: int = 45
    action_delta_rad = 0.1
    action_deadzone = 0.1
    include_command_error = True
    reward_sparse_bonus: float = 10.0
    reward_dense_scale = 0.1
    reward_time_cost = 0.1
    reward_action_rate_penalty = 0.0
    teledexter_style = True
    curriculum_stride_min: int = 20
    curriculum_stride_max: int = 40
    curriculum_anneal_steps: int = 25600
    curriculum_sigma_min: float = 0.7
    reference_start_fraction: float = 0.9
    dwell_min_steps: int = 5
    dwell_max_steps: int = 15
    cross_trajectory_probability: float = 0.1
    failure_tolerance_scale: float = 1.5
    cross_trajectory_failure_budget: int = 300
    paper_hand_dense: bool = True
    capture_model_transitions: bool = True
