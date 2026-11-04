"""External Allegro Fly hand asset adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ALLEGRO_FLY_ROOT_JOINT_NAMES = ("WRJ0x", "WRJ0y", "WRJ0z", "WRJ0rx", "WRJ0ry", "WRJ0rz")
ALLEGRO_FLY_FINGER_JOINT_NAMES = tuple(f"joint_{index}" for index in range(16))
ALLEGRO_FLY_TIP_BODY_NAMES = ("link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip")
ALLEGRO_FLY_PALM_BODY_NAME = "palm_link"
ALLEGRO_FLY_ARTICULATION_JOINT_NAMES = ALLEGRO_FLY_ROOT_JOINT_NAMES + ALLEGRO_FLY_FINGER_JOINT_NAMES
ALLEGRO_FLY_ASSET_RELATIVE_PATH = Path("urdf/allegro_hand_description_right_fly.urdf")


def allegro_description_root() -> Path:
    raw_root = os.environ.get("ALLEGRO_DESCRIPTION_ROOT")
    if not raw_root:
        raise RuntimeError("ALLEGRO_DESCRIPTION_ROOT is required for Allegro Fly assets.")
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ALLEGRO_DESCRIPTION_ROOT does not exist: {root}")
    return root


def allegro_hand_asset_path() -> Path:
    root = allegro_description_root()
    configured = os.environ.get("ALLEGRO_ASSET_PATH")
    path = Path(configured).expanduser() if configured else ALLEGRO_FLY_ASSET_RELATIVE_PATH
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Allegro Fly asset does not exist: {path}")
    return path


def allegro_hand_usd_path() -> str:
    """Resolve an explicitly supplied converted USD asset."""

    path = allegro_hand_asset_path()
    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise RuntimeError(f"Allegro asset is {path.suffix}, not USD: {path}")
    return str(path)


def make_allegro_fly_cfg() -> Any:
    """Build an articulation config, converting the external URDF when needed."""

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    asset_path = allegro_hand_asset_path()
    root_properties = sim_utils.ArticulationRootPropertiesCfg(
        fix_root_link=False,
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
    )
    if asset_path.suffix.lower() == ".urdf":
        spawn_cfg = sim_utils.UrdfFileCfg(
            asset_path=str(asset_path),
            fix_base=False,
            merge_fixed_joints=False,
            self_collision=True,
            activate_contact_sensors=True,
            articulation_props=root_properties,
        )
    else:
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(asset_path),
            activate_contact_sensors=True,
            articulation_props=root_properties,
        )
    return ArticulationCfg(
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.50),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={name: 0.0 for name in ALLEGRO_FLY_ARTICULATION_JOINT_NAMES},
        ),
        actuators={
            "root_and_fingers": ImplicitActuatorCfg(
                joint_names_expr=list(ALLEGRO_FLY_ARTICULATION_JOINT_NAMES),
                stiffness=20.0,
                damping=1.0,
            )
        },
        soft_joint_pos_limit_factor=1.0,
    )


def validate_allegro_fly_dof_contract(robot: Any) -> None:
    actual_joints = tuple(robot.joint_names)
    missing_joints = [name for name in ALLEGRO_FLY_ARTICULATION_JOINT_NAMES if name not in actual_joints]
    if missing_joints:
        raise RuntimeError(f"Allegro Fly articulation is missing joints: {missing_joints}")
    actual_bodies = set(getattr(robot, "body_names", ()))
    required_bodies = {ALLEGRO_FLY_PALM_BODY_NAME, *ALLEGRO_FLY_TIP_BODY_NAMES}
    missing_bodies = sorted(required_bodies - actual_bodies)
    if missing_bodies:
        raise RuntimeError(f"Allegro Fly articulation is missing bodies: {missing_bodies}")
    limits = getattr(getattr(robot, "data", None), "joint_limits", None)
    if limits is not None:
        import torch

        value = getattr(limits, "torch", limits)
        if value.numel() == 0 or not torch.isfinite(value).all():
            raise RuntimeError("Allegro Fly articulation produced invalid joint limits")
