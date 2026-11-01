"""Shared DirectRLEnv configuration for grasp and 30 degree cap turning."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from .task_contract import ACTION_DIM, actor_dim, critic_dim


@configclass
class BimanualBottleDirectEnvCfg(DirectRLEnvCfg):
    seed = 0
    episode_length_s = 5.0
    decimation = 10
    action_space = ACTION_DIM
    observation_space = actor_dim()
    state_space = critic_dim()
    task_kind = "grasp_hold"
    # The hard-pitch mimic backend is experimental and requires geometric
    # pregrasp resets. Legacy stable states violate its axial constraint.
    thread_backend = "legacy_d6"
    evaluation_mode = False
    stable_eval_bank_path = None
    stable_train_bank_path = None
    no_right_cap_contact_counterfactual = False
    stable_reset_probability = 0.65
    wrist_translation_scale = 0.002
    wrist_rotation_scale = 0.02
    finger_increment_scale = 0.05
    cap_goal_rad = 0.5235987755982988
    # Engineering acceptance: 30 degrees at the 3 mm example pitch travels
    # 0.25 mm; allow at most 20% error (0.05 mm) in that travel. This is a
    # simulation criterion, not a measured tolerance of the user's bottle.
    cap_pitch_tolerance_m = 0.00005
    cap_constraint_tolerance_m = 0.001
    thread_axial_stiffness = 500.0
    sim = SimulationCfg(
        dt=0.002, render_interval=decimation,
        physics=PhysxCfg(solver_type=1, min_position_iteration_count=8,
                         max_position_iteration_count=8,
                         enable_enhanced_determinism=True),
    )
    scene = InteractiveSceneCfg(num_envs=16, env_spacing=1.5, replicate_physics=True)


@configclass
class GraspHoldDirectEnvCfg(BimanualBottleDirectEnvCfg):
    task_kind = "grasp_hold"
    stable_reset_probability = 0.65


@configclass
class CapTurn30DirectEnvCfg(BimanualBottleDirectEnvCfg):
    task_kind = "cap_turn_30"
    stable_reset_probability = 1.0
