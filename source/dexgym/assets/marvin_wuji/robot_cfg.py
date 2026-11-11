"""Isaac Lab 3.0 articulation configuration for a locally converted MJCF."""

import json
import os
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

ASSET_ROOT = Path(__file__).parent
GENERATED = ASSET_ROOT / "hand2_2" / "generated"
_METADATA = json.loads((GENERATED / "metadata.json").read_text())
SOURCE_SHA256 = _METADATA["source_sha256"]
SOURCE_COMMIT = "5d4c345842f197bffeae7d7e08c4c3718230f9e0"  # Arm source only.
SOURCE_RELATIVE_PATH = "src/whole_body_mujoco_driver/model/dual_arm_marvin.xml"
PACKAGED_USD = GENERATED / "usd" / "dual_arm_marvin.usda"
NO_HAND_CONTACT_USD = GENERATED / "usd" / "dual_arm_marvin_no_hand_contacts.usda"
PACKAGED_KEYFRAMES = GENERATED / "metadata.json"
PACKAGED_MJCF = GENERATED / "dual_arm_marvin.xml"
LEGACY_MJCF = ASSET_ROOT / "source_model" / "dual_arm_marvin.xml"
LEGACY_USD = ASSET_ROOT / "usd" / "dual_arm_marvin.usda"


def _hand_joint_order(side: str) -> tuple[str, ...]:
    """The source MJCF orders each hand thumb, index, middle, ring, pinky."""
    return (
        f"{side}_thumb_cmc_flex",
        f"{side}_thumb_cmc_abd",
        f"{side}_thumb_mcp",
        f"{side}_thumb_ip",
        *(
            f"{side}_{finger}_{axis}"
            for finger in ("index_finger", "middle_finger", "ring_finger", "pinky")
            for axis in ("mcp_flex", "mcp_abd", "pip", "dip")
        ),
    )


JOINT_ORDER = (
    *(f"Joint{i}_L" for i in range(1, 8)),
    *_hand_joint_order("l"),
    *(f"Joint{i}_R" for i in range(1, 8)),
    *_hand_joint_order("r"),
)


def resolve_source_mjcf(path: str | Path | None = None) -> Path:
    """Resolve an explicit model override or the byte-identical packaged source."""
    candidate = path or os.environ.get("DEXGYM_MARVIN_MJCF") or PACKAGED_MJCF
    result = Path(candidate).expanduser().resolve()
    if not result.is_file() or result.name != "dual_arm_marvin.xml":
        raise FileNotFoundError(f"Expected dual_arm_marvin.xml: {result}")
    return result


def make_robot_cfg(
    usd_path: str | Path | None = None,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    *,
    hand_contacts: bool = True,
) -> ArticulationCfg:
    """Load Hand2.2 on Marvin arms, with explicit SI position-drive parameters.

    All 54 drives come from the composed MJCF, avoiding legacy USD degree/radian
    gain scaling. Hand contact is enabled; callers can explicitly disable it.
    """
    usd = Path(usd_path or (PACKAGED_USD if hand_contacts else NO_HAND_CONTACT_USD)).expanduser().resolve()
    if not usd.is_file():
        raise FileNotFoundError(usd)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(usd_path=str(usd), variants={"Physics": "mujoco"}),
        joint_ordering=JOINT_ORDER,
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={name: values["kp"] for name, values in _METADATA["drives"].items()},
                damping={name: values["kd"] for name, values in _METADATA["drives"].items()},
                joint_effort_limit={name: values["effort"] for name, values in _METADATA["drives"].items()},
            )
        },
    )
