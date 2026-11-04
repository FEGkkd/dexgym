"""Isaac Lab 3.0 articulation configuration for a locally converted MJCF."""

import os
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

SOURCE_SHA256 = "6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce"
SOURCE_COMMIT = "5d4c345842f197bffeae7d7e08c4c3718230f9e0"
SOURCE_RELATIVE_PATH = "src/whole_body_mujoco_driver/model/dual_arm_marvin.xml"


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
    """Resolve the external model without assuming a user-specific checkout path."""
    candidate = path or os.environ.get("DEXGYM_MARVIN_MJCF")
    if not candidate:
        raise ValueError("Pass --mjcf or set DEXGYM_MARVIN_MJCF to the external dual_arm_marvin.xml")
    result = Path(candidate).expanduser().resolve()
    if not result.is_file() or result.name != "dual_arm_marvin.xml":
        raise FileNotFoundError(f"Expected dual_arm_marvin.xml: {result}")
    return result


def make_robot_cfg(usd_path: str | Path, prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """Retain imported per-joint USD position-drive gains and effort limits.

    Isaac Lab 3.0 resolves ``None`` stiffness/damping from the imported USD;
    every named joint is covered by one implicit actuator group for MJWarp.
    Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/how-to/import_new_asset.html
    """
    usd = Path(usd_path).expanduser().resolve()
    if not usd.is_file():
        raise FileNotFoundError(usd)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(usd_path=str(usd), variants={"Physics": "physics"}),
        joint_ordering=JOINT_ORDER,
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )
