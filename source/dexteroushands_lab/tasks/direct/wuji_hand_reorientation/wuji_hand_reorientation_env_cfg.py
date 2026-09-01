"""Configuration for the Wuji Hand reorientation DirectTask."""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from ....assets.wuji_hand import (
    WUJI_HAND_CONTACT_FRICTION,
    WUJI_ISAAC_JOINT_NAMES,
    WUJI_OBJECT_FRICTION,
    WUJI_OBJECT_MASS_KG,
    WUJI_OBJECT_SIZE_M,
    make_wuji_hand_cfg,
)
from .task_contract import WUJI_CRITIC_OBSERVATION_DIM


@configclass
class WujiHandReorientationDirectEnvCfg(DirectRLEnvCfg):
    """DirectTask configuration with an imported native Wuji Hand asset."""

    seed = 0
    # Use a 50 ms policy period for the fixed-base native USD task.
    # Isaac Sim uses 120 Hz PhysX and six substeps per policy action.
    episode_length_s = 50.0
    decimation = 6
    action_scale = 0.5
    ema_alpha = 0.5
    warmup_time_s = 0.4
    action_space = 20
    observation_history_length = 3
    observation_frame_dim = 69
    observation_noise_enabled = True
    observation_space = observation_history_length * observation_frame_dim
    # DirectRLEnv exposes this as the optional critic gym observation space.
    state_space = WUJI_CRITIC_OBSERVATION_DIM

    sim = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            enable_enhanced_determinism=True,
        ),
    )
    scene = InteractiveSceneCfg(num_envs=256, env_spacing=1.0, replicate_physics=True)
    robot: ArticulationCfg = make_wuji_hand_cfg().replace(prim_path="/World/envs/env_.*/WujiHand")
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(WUJI_OBJECT_SIZE_M, WUJI_OBJECT_SIZE_M, WUJI_OBJECT_SIZE_M),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 0.85)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0
            ),
            physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
                static_friction=WUJI_OBJECT_FRICTION,
                dynamic_friction=WUJI_OBJECT_FRICTION,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=WUJI_OBJECT_MASS_KG),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0700, 0.0300, 0.5400), rot=(0.0, 0.0, 0.0, 1.0)
        ),
    )

    # The native WUJI USD does not author a PhysX material on its imported
    # collision prototypes. Apply the configured WUJI hand contact friction
    # as a stage-local inherited material in the DirectTask.
    wuji_hand_physics_material = sim_utils.PhysxRigidBodyMaterialCfg(
        static_friction=WUJI_HAND_CONTACT_FRICTION,
        dynamic_friction=WUJI_HAND_CONTACT_FRICTION,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )

    # One filtered PhysX sensor gives per-body hand/cube contact forces.
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/WujiHand/.*",
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
        history_length=1,
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=128,
    )

    actuated_joint_names = list(WUJI_ISAAC_JOINT_NAMES)
    object_start_pos = (0.0700, 0.0300, 0.5400)
    object_fall_height = 0.20

    initial_joint_pos = (0.06, 0.0, 0.0, 0.0) * 5

    tag_in_palm_pos = (0.0262, 0.0, -0.0563)
    tag_in_palm_quat_xyzw = (0.0, 0.70710678, 0.0, 0.70710678)

    success_threshold = 0.2
    success_hold_steps = 5
    goal_switch_delay = 20
    cage_margin = 0.01
    cage_z_upper_margin = 0.03
    cage_max_outside_steps = 10
    cage_counter_decay = 0.5
    contact_force_threshold = 1.0e-6
