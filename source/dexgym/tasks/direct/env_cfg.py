"""Direct joint target tracking configuration."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from dexgym.assets.cartpole import JointTrackingPhysicsCfg, JointTrackingSceneCfg


@configclass
class JointTrackingDirectCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_space = 1
    observation_space = 4  # cart position, velocity, target, error
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2, physics=JointTrackingPhysicsCfg())
    scene: JointTrackingSceneCfg = JointTrackingSceneCfg(num_envs=16, env_spacing=4.0, clone_in_fabric=True)
