"""Pinned 500 Hz Newton physics and 50 Hz arm-only policy configuration."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg


@configclass
class MarvinWaveSceneCfg(InteractiveSceneCfg):
    robot = make_robot_cfg()


@configclass
class MarvinWaveDirectCfg(DirectRLEnvCfg):
    random_reference_start = True
    reset_joint_perturb_rad = 0.03
    active_reset_fraction = 0.0
    active_reset_frame_range = (450, 850)
    early_reset_fraction = 0.0
    early_reset_frame_range = (0, 50)
    initialize_hands_from_pd_replay = True
    reward_profile = "marvin_tcp_priority"
    observation_position_noise_rad = 0.0
    observation_velocity_noise_rad_s = 0.0
    audit_contacts_each_physics_step = False
    decimation = 10
    episode_length_s = 10.0
    action_space = 14
    observation_space = 1460
    state_space = 0
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        render_interval=10,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False),
    )
    scene: MarvinWaveSceneCfg = MarvinWaveSceneCfg(num_envs=64, env_spacing=3.0)
