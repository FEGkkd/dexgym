"""Configuration for the DexTrack Allegro Hand tracking DirectTask."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from ....assets.allegro_hand import make_allegro_fly_cfg
from .task_contract import ALLEGRO_ACTION_DIM, ALLEGRO_OBSERVATION_DIM


@configclass
class AllegroHandTrackDirectEnvCfg(DirectRLEnvCfg):
    """Single-trajectory GRAB tracking configuration."""

    seed = 0
    episode_length_s = 8.0
    decimation = 6
    action_scale = 0.10
    reference_fps = 20.0
    action_space = ALLEGRO_ACTION_DIM
    observation_space = ALLEGRO_OBSERVATION_DIM
    state_space = 0

    sim = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=1,
            enable_enhanced_determinism=True,
        ),
    )
    scene = InteractiveSceneCfg(num_envs=64, env_spacing=1.25, replicate_physics=True)
    robot: ArticulationCfg = make_allegro_fly_cfg().replace(
        prim_path="/World/envs/env_.*/AllegroHand"
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 0.85)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
                static_friction=0.4,
                dynamic_friction=0.35,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    reference_path: str = os.environ.get("DEXTRACK_REFERENCE_PATH", "")
    residual_limit: float = 0.25
    object_goal_pos = (0.0, 0.0, 0.42)
    object_goal_rot_xyzw = (0.0, 0.0, 0.0, 1.0)

    hand_reward_scale = 1.0
    object_position_reward_scale = 2.0
    object_orientation_reward_scale = 0.25
    contact_reward_scale = 0.25
    action_penalty_scale = 0.01
    residual_penalty_scale = 0.01
    success_reward = 2.0
    object_loss_distance = 0.60
    object_loss_height = 0.20

    @property
    def resolved_reference_path(self) -> str:
        """Return the configured reference path or the first GRAB sample."""

        if self.reference_path:
            return self.reference_path
        root = os.environ.get("DEXTRACK_DATA_ROOT")
        if not root:
            raise RuntimeError(
                "DEXTRACK_DATA_ROOT or DEXTRACK_REFERENCE_PATH is required for "
                "Allegro tracking."
            )
        from pathlib import Path

        candidates = sorted(Path(root).glob("data/*.npy"))
        if not candidates:
            raise FileNotFoundError(f"no GRAB .npy references found under {root}")
        return str(candidates[0].resolve())
