"""Pinned official WujiHand2 Beta assets, separate from the legacy Wuji hand.

The MJCF is used only to audit names and limits. Isaac Sim loads the upstream
native USD. No MuJoCo runtime or converted asset is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


SOURCE_TAG = "v2026.9.11"
SOURCE_COMMIT = "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95"
USD_SHA256 = {
    ("beta1", "left"): "8d1b83a6130b64fd759ea3c0720d6b255da8c1e76abe5a4226788183058dbfda",
    ("beta1", "right"): "22600d3f0aeb185552bb6eeb454b5b7199966a60fc844d4a4b5e4791ec77f0ed",
    ("beta2", "left"): "f8094cff005b5b069d55cb3f36402f98aba5e5983f583e9968be2763a2d0909b",
    ("beta2", "right"): "839fc0d0d0c06a2d7d463851b087a2dfe69bc883bc240ead47230536cbd54b20",
}
FINGER_ORDER = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
ACTUATOR_GROUPS = ("THJ", "FFJ", "MFJ", "RFJ", "LFJ")
JOINT_SUFFIXES = (
    ("cmc_flex", "cmc_abd", "mcp", "ip"),
    ("mcp_flex", "mcp_abd", "pip", "dip"),
    ("mcp_flex", "mcp_abd", "pip", "dip"),
    ("mcp_flex", "mcp_abd", "pip", "dip"),
    ("mcp_flex", "mcp_abd", "pip", "dip"),
)


@dataclass(frozen=True)
class HandJoint:
    name: str
    actuator: str
    lower: float
    upper: float
    mjcf_effort_limit: float
    usd_drive_max_force: float


@dataclass(frozen=True)
class Hand2Spec:
    variant: str
    side: str
    usd_path: Path
    usd_sha256: str
    source_tag: str
    source_commit: str
    commit_verified: bool
    root_body: str
    body_names: tuple[str, ...]
    fingertip_bodies: tuple[str, ...]
    joints: tuple[HandJoint, ...]
    collision_exclusions: tuple[tuple[str, str], ...]
    requires_tactile: bool = False

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def default_pose(self) -> dict[str, float]:
        return {name: 0.0 for name in self.joint_names}

    @property
    def palm_body(self) -> str:
        """Official USD has palm geometry on the wrist rigid body, not a separate palm link."""
        return self.root_body


def _source_root(root: str | Path | None) -> Path:
    raw = root if root is not None else os.environ.get("WUJI_DESCRIPTION_ROOT")
    if not raw:
        raise RuntimeError("Set WUJI_DESCRIPTION_ROOT to the pinned official wuji-description checkout")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_hand2_spec(
    root: str | Path | None = None, *, variant: str = "beta1", side: str
) -> Hand2Spec:
    """Validate one official side/version and return its canonical joint map.

    ``beta1`` is only a default candidate for hardware without tactile pads;
    it is not a hardware identification claim.
    """
    if variant not in ("beta1", "beta2") or side not in ("left", "right"):
        raise ValueError(f"Unsupported WujiHand2 asset selection: {variant}/{side}")
    root_path = _source_root(root)
    prefix = "l" if side == "left" else "r"
    body_dir = root_path / "hand2" / f"hand2_{variant}" / "body"
    usd_filename = "wujihand2.usd" if variant == "beta1" else "wujihand2_beta2.usd"
    usd_path = body_dir / "usd" / side / usd_filename
    mjcf_path = body_dir / "mjcf" / f"{side}.xml"
    for path in (usd_path, mjcf_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    digest = sha256(usd_path.read_bytes()).hexdigest()
    if digest != USD_SHA256[(variant, side)]:
        raise ValueError(f"Official USD checksum mismatch: {usd_path}: {digest}")

    commit_verified = False
    if (root_path / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        commit = result.stdout.strip()
        if commit != SOURCE_COMMIT:
            raise ValueError(f"Expected wuji-description {SOURCE_COMMIT}, found {commit}")
        commit_verified = True

    tree = ET.parse(mjcf_path)
    body_names = tuple(node.attrib["name"] for node in tree.findall(".//body[@name]"))
    source_joints = {node.attrib["name"]: node for node in tree.findall(".//joint[@name]")}
    actuators = {node.attrib["name"]: node for node in tree.findall("./actuator/position")}
    joints: list[HandJoint] = []
    for finger, group, suffixes in zip(FINGER_ORDER, ACTUATOR_GROUPS, JOINT_SUFFIXES):
        for index, suffix in enumerate(suffixes):
            name = f"{prefix}_{finger}_{suffix}"
            actuator = f"{prefix}_{group}{index}"
            if actuators[actuator].attrib["joint"] != name:
                raise ValueError(f"Actuator {actuator} does not drive {name}")
            source = source_joints[name]
            lower, upper = map(float, source.attrib["range"].split())
            effort_low, effort_high = map(float, source.attrib["actuatorfrcrange"].split())
            if not (lower < upper and effort_low < 0 < effort_high):
                raise ValueError(f"Invalid official limits: {name}")
            # The official native USD uses 2 N m on all twenty drives at this
            # pinned tag; MJCF actuator force limits differ. Preserve the USD
            # value at runtime and report both instead of silently replacing it.
            joints.append(HandJoint(name, actuator, lower, upper,
                                    min(-effort_low, effort_high), 2.0))
    if len(source_joints) != 20 or len(actuators) != 20:
        raise ValueError("Official hand is not 20 active joints")
    expected_tips = tuple(
        f"{prefix}_{finger}_{'tip_sensor_frame' if variant == 'beta2' else 'distal'}"
        for finger in FINGER_ORDER
    )
    if any(name not in body_names for name in expected_tips):
        raise ValueError("Official fingertip body is missing")
    exclusions = tuple(
        (node.attrib["body1"], node.attrib["body2"])
        for node in tree.findall("./contact/exclude")
    )
    return Hand2Spec(
        variant, side, usd_path, digest, SOURCE_TAG, SOURCE_COMMIT, commit_verified,
        f"{prefix}_wrist", body_names, expected_tips, tuple(joints), exclusions,
    )


def make_wuji_hand2_beta_cfg(
    prim_path: str, *, side: str, variant: str = "beta1",
    friction: float = 0.7, root: str | Path | None = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.5),
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
):
    """Create an Isaac Lab 3.0 articulation with a free physics-controlled base.

    Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.sim.schemas.html
    ``fix_root_link=False`` disables the official USD's world fixed joint.
    """
    if friction < 0:
        raise ValueError("friction must be nonnegative")
    spec = load_hand2_spec(root, variant=variant, side=side)
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path=prim_path,
        articulation_root_prim_path=f"/{spec.root_body}",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_floating_wuji_hand2_beta,
            usd_path=str(spec.usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.0,
                max_angular_velocity=360.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position, rot=orientation_xyzw, joint_pos=spec.default_pose,
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=list(spec.joint_names),
                stiffness=None, damping=None,
            ),
        },
    )


def spawn_floating_wuji_hand2_beta(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Reference the official USD and move its articulation root to the wrist.

    Isaac Lab's ``fix_root_link=False`` disables the official world joint, but
    PhysX 5.4 then requires ArticulationRootAPI on the free wrist rigid body.
    The override lives on the scene layer; upstream USD bytes remain intact.
    """
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdPhysics, UsdShade

    # The upstream colliders are instanced. Isaac Lab's recursive material
    # binding cannot author on instance proxies; bind to rigid-body ancestors.
    prim = sim_utils.spawn_from_usd(
        prim_path, cfg.replace(physics_material=None),
        translation=translation, orientation=orientation, **kwargs,
    )
    side = Path(cfg.usd_path).parent.name
    if side not in ("left", "right"):
        raise ValueError(f"Unexpected official Wuji USD side: {cfg.usd_path}")
    root_body = f"{'l' if side == 'left' else 'r'}_wrist"
    fixed_joint = prim.GetChild("root_joint")
    wrist = prim.GetChild(root_body)
    if not fixed_joint or not wrist or not wrist.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Official USD root structure changed: {cfg.usd_path}")
    UsdPhysics.Joint(fixed_joint).CreateJointEnabledAttr(False)
    fixed_joint.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(wrist)
    if cfg.articulation_props is not None:
        sim_utils.modify_articulation_root_properties(
            str(wrist.GetPath()), cfg.articulation_props.replace(fix_root_link=None)
        )
    if cfg.physics_material is not None:
        material = UsdShade.Material.Define(
            prim.GetStage(), str(prim.GetPath()) + "/ContactMaterial"
        )
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(cfg.physics_material.static_friction)
        api.CreateDynamicFrictionAttr(cfg.physics_material.dynamic_friction)
        for body in Usd.PrimRange(prim):
            if body.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdShade.MaterialBindingAPI.Apply(body).Bind(
                    material, materialPurpose="physics"
                )
    return prim
