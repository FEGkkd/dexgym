"""Machine-readable reference values from WUJI's official MuJoCo model.

The Isaac asset is the official pre-built native USD. Keeping the
official MJCF values in a small, simulator-independent module lets audits
report the difference explicitly instead of treating URDF self-consistency as
MJCF equivalence.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


WUJI_OFFICIAL_MJCF_RELATIVE_PATH = Path(
    "src/wuji_mjlab/assets/robots/wuji_hand/mjcf/right_mjlab.xml"
)
WUJI_OFFICIAL_CUBE_RELATIVE_PATH = Path(
    "src/wuji_mjlab/assets/objects/inhand_object/xmls/cube.xml"
)

WUJI_MJCF_BODY_NAMES = (
    "right_palm_link",
    *(f"right_finger{finger}_link{link}" for finger in range(1, 6) for link in range(1, 5)),
)

# The native official USD articulation traverses the wrist, then all finger links
# at each kinematic depth.  This is the order used by robot.body_names.
WUJI_MJCF_TO_ISAAC_BODY_NAME = {
    "right_palm_link": "r_wrist",
    **{
        f"right_finger{finger}_link{link}": (
            "r_thumb_" if finger == 1 else {
                2: "r_index_finger_",
                3: "r_middle_finger_",
                4: "r_ring_finger_",
                5: "r_pinky_",
            }[finger]
        )
        + {1: "proximal", 2: "proximal_abd", 3: "middle", 4: "distal"}[link]
        for finger in range(1, 6)
        for link in range(1, 5)
    },
}

WUJI_OFFICIAL_BODY_NAMES = (
    "r_wrist",
    "r_thumb_proximal",
    "r_index_finger_proximal",
    "r_middle_finger_proximal",
    "r_ring_finger_proximal",
    "r_pinky_proximal",
    "r_thumb_proximal_abd",
    "r_index_finger_proximal_abd",
    "r_middle_finger_proximal_abd",
    "r_ring_finger_proximal_abd",
    "r_pinky_proximal_abd",
    "r_thumb_middle",
    "r_index_finger_middle",
    "r_middle_finger_middle",
    "r_ring_finger_middle",
    "r_pinky_middle",
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)

WUJI_OFFICIAL_BODY_MASS = (
    0.16914,
    0.0075,
    0.0032,
    0.0032,
    0.0032,
    0.0032,
    0.0333532,
    0.033,
    0.033,
    0.033,
    0.033,
    0.0165,
    0.0165,
    0.0165,
    0.0165,
    0.0165,
    0.0206,
    0.01718,
    0.01718,
    0.01718,
    0.01718,
)

# Active collision groups in right_mjlab.xml.  The thumb link2 has two
# colliders: a capsule and a simplified soft-body mesh.
WUJI_OFFICIAL_COLLISION_GEOMS_BY_BODY = {
    "r_wrist": ("right_palm_collision",),
    "r_thumb_proximal_abd": (
        "right_finger1_link2_col",
        "right_finger1_link2_softbody_col",
    ),
    "r_thumb_middle": ("right_finger1_link3_col",),
    "r_thumb_distal": ("right_finger1_link4_col",),
    **{
        f"{body_prefix}_{suffix}": (f"right_finger{finger}_link{link}_col",)
        for finger, body_prefix in (
            (2, "r_index_finger"),
            (3, "r_middle_finger"),
            (4, "r_ring_finger"),
            (5, "r_pinky"),
        )
        for link, suffix in ((2, "proximal_abd"), (3, "middle"), (4, "distal"))
    },
}
WUJI_OFFICIAL_COLLISION_GEOM_COUNT = sum(
    len(names) for names in WUJI_OFFICIAL_COLLISION_GEOMS_BY_BODY.values()
)
WUJI_OFFICIAL_COLLISION_BODY_NAMES = tuple(WUJI_OFFICIAL_COLLISION_GEOMS_BY_BODY)

WUJI_OFFICIAL_CONTACT_EXCLUDES = (
    ("r_wrist", "r_thumb_proximal"),
    ("r_wrist", "r_thumb_proximal_abd"),
    ("r_wrist", "r_index_finger_proximal_abd"),
    ("r_wrist", "r_middle_finger_proximal_abd"),
    ("r_wrist", "r_ring_finger_proximal_abd"),
    ("r_wrist", "r_pinky_proximal_abd"),
)

WUJI_OFFICIAL_HAND_CONTACT = {
    "friction": 0.7,
    "solref": (0.02, 1.5),
    "solimp": (0.9, 0.95, 0.001),
    "priority": 1,
}
WUJI_OFFICIAL_SIMULATION = {
    "timestep": 0.01,
    "integrator": "Euler",
    "iterations": 5,
    "ls_iterations": 8,
}
WUJI_OFFICIAL_CUBE = {
    "size": (0.054, 0.054, 0.054),
    "mass": 0.12,
    "friction": 0.3,
    "condim": 3,
    "priority": 0,
    "solref": (0.015, 1.0),
    "solimp": (0.85, 0.85, 0.001),
}


@dataclass(frozen=True)
class OfficialMjcfSummary:
    """Reference counts and scalar values parsed from the official XML files."""

    body_names: tuple[str, ...]
    body_masses: tuple[float, ...]
    joint_names: tuple[str, ...]
    collision_geom_names: tuple[str, ...]
    collision_geom_count: int
    body_collision_counts: dict[str, int]
    mass_total: float


def official_mjcf_path() -> Path | None:
    """Resolve the official MJCF from an explicit env var or this checkout."""

    configured = os.environ.get("WUJI_MJCF_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.is_file() else None

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "wuji-mjlab" / WUJI_OFFICIAL_MJCF_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    return None


def parse_official_mjcf(path: str | Path) -> OfficialMjcfSummary:
    """Parse body, joint, and active collision geometry facts from MJCF."""

    root = ET.parse(path).getroot()
    body_names: list[str] = []
    body_masses: list[float] = []
    collision_names: list[str] = []
    body_collision_counts: dict[str, int] = {}
    joint_names: list[str] = []
    for body in root.findall(".//worldbody//body"):
        name = body.attrib["name"]
        body_names.append(name)
        inertial = body.find("inertial")
        body_masses.append(float(inertial.attrib["mass"]) if inertial is not None else 0.0)
        active_count = 0
        for geom in body.findall("geom"):
            if geom.attrib.get("contype", "1") == "0" and geom.attrib.get("conaffinity", "1") == "0":
                continue
            geom_name = geom.attrib.get("name")
            if geom_name:
                collision_names.append(geom_name)
            active_count += 1
        if active_count:
            body_collision_counts[name] = active_count
        joint_names.extend(joint.attrib["name"] for joint in body.findall("joint"))

    return OfficialMjcfSummary(
        body_names=tuple(body_names),
        body_masses=tuple(body_masses),
        joint_names=tuple(joint_names),
        collision_geom_names=tuple(collision_names),
        collision_geom_count=len(collision_names),
        body_collision_counts=body_collision_counts,
        mass_total=sum(body_masses),
    )


def official_reference_summary() -> OfficialMjcfSummary | None:
    """Return parsed official facts when the sibling WUJI checkout is present."""

    path = official_mjcf_path()
    return parse_official_mjcf(path) if path is not None else None
