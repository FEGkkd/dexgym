"""Official WUJI Hand USD asset adapter for Isaac Sim.

The task is trained with Isaac Sim/PhysX. The official pre-built USD is the
physics source of truth; WUJI_DESCRIPTION_ROOT selects the description checkout.
The MuJoCo checkout is used only as a semantic policy-contract reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

WUJI_MJCF_ACTUATOR_NAMES = (
    "r_THJ0", "r_THJ1", "r_THJ2", "r_THJ3",
    "r_FFJ0", "r_FFJ1", "r_FFJ2", "r_FFJ3",
    "r_MFJ0", "r_MFJ1", "r_MFJ2", "r_MFJ3",
    "r_RFJ0", "r_RFJ1", "r_RFJ2", "r_RFJ3",
    "r_LFJ0", "r_LFJ1", "r_LFJ2", "r_LFJ3",
)

# The official USD exposes these names directly. They are ordered in the
# same finger-major order as the 20-DOF policy contract.
WUJI_ISAAC_JOINT_NAMES = tuple(
    f"right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)

# Compatibility alias for task/config code written before the switch to the
# official native USD.
WUJI_URDF_JOINT_NAMES = WUJI_ISAAC_JOINT_NAMES

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


def wuji_hand_urdf_path() -> str:
    """Return the official right-hand URDF used only for tip-frame inspection."""

    path = wuji_description_root() / "hand/body/urdf/right.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"Wuji Hand URDF does not exist: {path}")
    return str(path)



def wuji_hand_usd_path() -> str:
    """Return the official pre-built USD, or an explicit asset override."""

    override = os.environ.get("WUJI_ISAAC_ASSET_PATH")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"WUJI_ISAAC_ASSET_PATH does not exist: {path}")
        return str(path)

    candidates = []
    configured_root = os.environ.get("WUJI_DESCRIPTION_ROOT")
    if configured_root:
        candidates.append(
            Path(configured_root).expanduser().resolve()
            / "hand/body/usd/right/wujihand.usd"
        )

    # Keep a convenient sibling-checkout fallback for local development. The
    # explicit environment variable remains the reproducible path in CI.
    project_root = Path(__file__).resolve().parents[3]
    candidates.extend([
        project_root.parent / "isaaclab-sim/wuji_hand_description/usd/right/wujihand.usd",
        project_root.parent / "wuji-hand-description/usd/right/wujihand.usd",
    ])
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "Official WUJI Hand USD was not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


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
