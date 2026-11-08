"""Standalone 20-DoF Wuji Hand 2.1 articulation for Isaac Lab Newton."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

from dexgym.data.wuji_retarget import ROBOT_RIGHT_JOINT_NAMES

from . import OFFICIAL_USD

JOINT_ORDER = ROBOT_RIGHT_JOINT_NAMES

_FLEX = "r_(thumb_cmc_flex|index_finger_mcp_flex|middle_finger_mcp_flex|ring_finger_mcp_flex|pinky_mcp_flex)"
_ABD = "r_(thumb_cmc_abd|index_finger_mcp_abd|middle_finger_mcp_abd|ring_finger_mcp_abd|pinky_mcp_abd)"
_MIDDLE = "r_(thumb_mcp|index_finger_pip|middle_finger_pip|ring_finger_pip|pinky_pip)"
_DISTAL = "r_(thumb_ip|index_finger_dip|middle_finger_dip|ring_finger_dip|pinky_dip)"

# Explicit SI gains from the official USD's config.yaml. The authored USD
# fallback drives are 50/2 for every joint and do not preserve these groups.
OFFICIAL_STIFFNESS = {_FLEX: 2.0, _ABD: 2.0, _MIDDLE: 1.0, _DISTAL: 0.8}
OFFICIAL_DAMPING = {_FLEX: 0.05, _ABD: 0.05, _MIDDLE: 0.03, _DISTAL: 0.03}
OFFICIAL_EFFORT_LIMIT = {
    "r_thumb_cmc_(flex|abd)": 0.6,
    "r_thumb_(mcp|ip)": 0.3,
    "r_(index_finger|middle_finger|ring_finger|pinky)_mcp_flex": 2.0,
    "r_(index_finger|middle_finger|ring_finger|pinky)_mcp_abd": 0.2,
    "r_(index_finger|middle_finger|ring_finger|pinky)_(pip|dip)": 0.3,
}


def make_robot_cfg(prim_path: str = "{ENV_REGEX_NS}/RightHand") -> ArticulationCfg:
    """Load official right-hand USD with its published SI drive configuration."""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(usd_path=str(OFFICIAL_USD), variants={"Physics": "PhysX"}),
        joint_ordering=JOINT_ORDER,
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["r_.*"],
                stiffness=OFFICIAL_STIFFNESS,
                damping=OFFICIAL_DAMPING,
                joint_effort_limit=OFFICIAL_EFFORT_LIMIT,
            )
        },
    )
