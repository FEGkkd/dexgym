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
class WujiCurriculumCfg:
    enabled: bool = True
    success_count_threshold: int = 3
    delta_per_loop: float = 0.08
    adaptive_episode_enabled: bool = True
    adaptive_target_steps: int = 800
    adaptive_inc_rate: float = 0.1
    adaptive_dec_rate: float = 0.2
    adaptive_min_scale: float = 0.05
    adaptive_drop_gamma: float = 1.0


@configclass
class WujiDomainRandomizationCfg:
    enabled: bool = True
    object_com_range: tuple[float, float] = (-0.003, 0.003)
    object_mass_scale: tuple[float, float] = (0.4, 1.6)
    object_size_scale: tuple[float, float] = (0.85, 1.15)
    hand_friction_scale: tuple[float, float] = (0.7, 1.3)
    contact_soft_width: tuple[float, float] = (2.0, 5.0)
    contact_finger_width: tuple[float, float] = (1.0, 2.0)
    contact_dmin: tuple[float, float] = (0.5, 1.0)
    pd_stiffness_scale: tuple[float, float] = (0.75, 1.5)
    pd_damping_scale: tuple[float, float] = (0.5, 2.0)
    joint_damping_scale: tuple[float, float] = (0.3, 3.0)
    joint_armature_scale: tuple[float, float] = (0.75, 1.3)
    joint_friction_scale: tuple[float, float] = (0.5, 2.0)
    encoder_bias_range: tuple[float, float] = (-0.01, 0.01)
    link_mass_scale: tuple[float, float] = (0.4, 1.5)
    link_inertia_scale: tuple[float, float] = (0.4, 1.5)


@configclass
class WujiDisturbanceCfg:
    enabled: bool = True
    interval_s: tuple[float, float] = (0.6, 1.8)
    speed_range: tuple[float, float] = (0.05, 0.15)
    warmup_time_s: float = 3.0
    warmup_frac: float = 0.05
    rampup_frac: float = 0.80


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

    # Compatibility switches used by train/play tooling.  The nested groups
    # below carry the tunable parameters; both gates must be enabled for a
    # feature to affect the environment.
    enable_curriculum = True
    enable_domain_randomization = True
    enable_disturbance = True
    curriculum = WujiCurriculumCfg()
    domain_randomization = WujiDomainRandomizationCfg()
    disturbance = WujiDisturbanceCfg()

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
