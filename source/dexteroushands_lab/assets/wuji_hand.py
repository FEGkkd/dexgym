"""Official WUJI Hand USD asset adapter for Isaac Sim.

The task is trained with Isaac Sim/PhysX. The official pre-built USD is the
physics source of truth; WUJI_DESCRIPTION_ROOT selects the description checkout.
The task contract is defined by the native USD joint and body names.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# The official USD exposes these names directly. They are ordered in the
# same finger-major order as the 20-DOF policy contract.
WUJI_ISAAC_JOINT_NAMES = tuple(
    f"right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)

WUJI_ISAAC_BODY_NAMES = (
    "right_palm_link",
    *(f"right_finger{finger}_link{link}" for finger in range(1, 6) for link in range(1, 5)),
    *(f"right_finger{finger}_tip_link" for finger in range(1, 6)),
)
WUJI_ISAAC_TIP_BODY_NAMES = tuple(
    f"right_finger{finger}_tip_link" for finger in range(1, 6)
)
WUJI_ISAAC_NON_TIP_BODY_NAMES = tuple(
    name for name in WUJI_ISAAC_BODY_NAMES if name not in WUJI_ISAAC_TIP_BODY_NAMES
)

WUJI_HAND_USD_RELATIVE_PATH = Path("hand/body/usd/right/wujihand.usd")
WUJI_HAND_CONTACT_FRICTION = 0.7
WUJI_OBJECT_SIZE_M = 0.054
WUJI_OBJECT_MASS_KG = 0.12
WUJI_OBJECT_FRICTION = 0.3

# Isaac Lab pose buffers use xyzw. The official demo starts at 0.06 rad
# on the first joint of each finger and zero elsewhere.
WUJI_REORIENT_ROOT_POS = (0.0, 0.0, 0.5)
WUJI_REORIENT_ROOT_QUAT_XYZW = (0.0, -0.70710678, 0.0, 0.70710678)
WUJI_REORIENT_DEFAULT_JOINT_POS = {
    name: (0.06 if name.endswith("_joint1") else 0.0)
    for name in WUJI_ISAAC_JOINT_NAMES
}


def wuji_description_root() -> Path:
    """Return the external Wuji description checkout selected by the user."""

    raw_root = os.environ.get("WUJI_DESCRIPTION_ROOT")
    if not raw_root:
        raise RuntimeError(
            "WUJI_DESCRIPTION_ROOT is required for Wuji tasks; "
            "point it at the wuji-description repository."
        )
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"WUJI_DESCRIPTION_ROOT does not exist: {root}")
    return root


def wuji_hand_usd_path() -> str:
    """Return the official native USD from the external description checkout."""

    path = wuji_description_root() / WUJI_HAND_USD_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Official WUJI Hand USD does not exist: {path}")
    return str(path)


def make_wuji_hand_cfg() -> ArticulationCfg:
    """Build the Isaac articulation config from the task-ready Wuji USD asset."""

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=wuji_hand_usd_path(),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # The official USD contains /root_joint as a fixed articulation
                # root. Isaac Lab resolves it when no relative root is supplied.
                fix_root_link=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=WUJI_REORIENT_ROOT_POS,
            rot=WUJI_REORIENT_ROOT_QUAT_XYZW,
            joint_pos=WUJI_REORIENT_DEFAULT_JOINT_POS,
        ),
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=list(WUJI_ISAAC_JOINT_NAMES),
                # None preserves the official USD drive values.
                stiffness=None,
                damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
