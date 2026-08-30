from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG


@configclass
class DirectShadowHandPoseEnvCfg(DirectRLEnvCfg):
    """Minimal DirectTask used to validate the full training data path."""

    episode_length_s = 5.0
    decimation = 2
    action_scale = 0.35
    velocity_scale = 0.1
    action_space = 20
    observation_space = 80
    state_space = 0

    sim = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene = InteractiveSceneCfg(num_envs=1024, env_spacing=1.5, replicate_physics=True)
    robot: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
