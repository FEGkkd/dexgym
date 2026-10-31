"""Isaac Lab InteractiveScene configuration for the standalone bimanual fixture."""

from __future__ import annotations

import math
from pathlib import Path

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.wuji_hand2_beta import make_wuji_hand2_beta_cfg


def make_bimanual_bottle_scene_cfg(
    *, num_envs: int, asset_dir: Path, bottle: BottleConfig,
    variant: str = "beta1", contact_debug: bool = False,
    bottle_tilt_radians: float = 0.0,
    bottle_center_z: float = 0.5,
    wrist_control_mode: str = "floating_wrench_pd",
    left_initial_pose: tuple[float, ...] | None = None,
    right_initial_pose: tuple[float, ...] | None = None,
):
    """Build a scene without registering an RL task or policy observation."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils.configclass import configclass

    left_orientation = (0.0, -0.7071067811865475, 0.0, 0.7071067811865476)
    right_orientation = (0.0, 0.7071067811865475, 0.0, 0.7071067811865476)
    left_pose = left_initial_pose or (-0.30, 0.0, 0.62, *left_orientation)
    right_pose = right_initial_pose or (0.30, 0.0, 0.62, *right_orientation)
    if len(left_pose) != 7 or len(right_pose) != 7:
        raise ValueError("Each initial wrist pose must have xyz and xyzw quaternion")
    bottle_orientation = (
        0.0, math.sin(bottle_tilt_radians / 2),
        0.0, math.cos(bottle_tilt_radians / 2),
    )
    cap_position = (
        bottle.cap_initial_z * math.sin(bottle_tilt_radians),
        0.0,
        bottle_center_z + bottle.cap_initial_z * math.cos(bottle_tilt_radians),
    )
    bottle_path = asset_dir / "bottle.usda"
    cap_path = asset_dir / "cap.usda"
    if not bottle_path.is_file() or not cap_path.is_file():
        raise FileNotFoundError("Generate bottle.usda and cap.usda before scene creation")

    @configclass
    class BimanualBottleSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
        )
        left_hand: ArticulationCfg = make_wuji_hand2_beta_cfg(
            "{ENV_REGEX_NS}/LeftHand", side="left", variant=variant,
            position=left_pose[:3], orientation_xyzw=left_pose[3:7],
            control_mode=wrist_control_mode,
        )
        right_hand: ArticulationCfg = make_wuji_hand2_beta_cfg(
            "{ENV_REGEX_NS}/RightHand", side="right", variant=variant,
            position=right_pose[:3], orientation_xyzw=right_pose[3:7],
            control_mode=wrist_control_mode,
        )
        bottle_body: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bottle",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(bottle_path), activate_contact_sensors=True,
                rigid_props=sim_utils.PhysxRigidBodyPropertiesCfg(
                    angular_damping=bottle.body_angular_damping,
                    max_linear_velocity=bottle.max_body_linear_velocity,
                    max_angular_velocity=bottle.max_body_angular_velocity_deg,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, bottle_center_z), rot=bottle_orientation,
            ),
        )
        bottle_cap: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cap",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(cap_path), activate_contact_sensors=True,
                rigid_props=sim_utils.PhysxRigidBodyPropertiesCfg(
                    angular_damping=bottle.body_angular_damping,
                    max_linear_velocity=bottle.max_body_linear_velocity,
                    max_angular_velocity=bottle.max_body_angular_velocity_deg,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=cap_position, rot=bottle_orientation,
            ),
        )
        cap_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cap",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
            update_period=0.0, debug_vis=contact_debug,
        )
        left_hand_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/LeftHand/l_.*",
            update_period=0.0, debug_vis=contact_debug,
        )
        right_hand_contact: ContactSensorCfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/RightHand/r_.*",
            update_period=0.0, debug_vis=contact_debug,
        )

    return BimanualBottleSceneCfg(num_envs=num_envs, env_spacing=1.5)
