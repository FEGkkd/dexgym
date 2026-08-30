from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG


@configclass
class ShadowHandReorientationDirectEnvCfg(DirectRLEnvCfg):
    """DirectTask configuration for object orientation reorientation."""

    seed = 0

    episode_length_s = 10.0
    decimation = 2
    action_scale = 0.35
    velocity_scale = 0.2
    action_space = 20
    observation_space = 84
    state_space = 0

    sim = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene = InteractiveSceneCfg(num_envs=1024, env_spacing=1.0, replicate_physics=True)
    robot: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.SphereCfg(
            radius=0.045,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.18, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    actuated_joint_names = [
        "robot0_WRJ1", "robot0_WRJ0", "robot0_FFJ3", "robot0_FFJ2", "robot0_FFJ1",
        "robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1", "robot0_RFJ3", "robot0_RFJ2",
        "robot0_RFJ1", "robot0_LFJ4", "robot0_LFJ3", "robot0_LFJ2", "robot0_LFJ1",
        "robot0_THJ4", "robot0_THJ3", "robot0_THJ2", "robot0_THJ1", "robot0_THJ0",
    ]

    object_start_pos = (0.0, -0.18, 0.56)
    object_fall_height = 0.30
    orientation_reward_scale = 5.0
    position_reward_scale = 1.0
    action_penalty_scale = 0.01
