"""Reproducible bottle-relative bimanual initial states, independent of RL.

Generated ``stable_pregrasp`` states are *candidates*. Only the gravity-on
PhysX hold fixture may mark one as validated and place it in a bank.
All poses use metres and xyzw quaternions in the env-0 world frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Literal

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.wuji_hand2_beta import BETA1_TIP_MARKER_MASS_KG
from dexteroushands_lab.assets.wrist_6dof import _quat_mul, _quat_normalize


Pose7 = tuple[float, float, float, float, float, float, float]
Velocity6 = tuple[float, float, float, float, float, float]
Mode = Literal["stable_pregrasp", "weak_contact", "near_contact"]
ZERO_VEL: Velocity6 = (0.0,) * 6
LEFT_LOCAL_QUAT = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
RIGHT_LOCAL_QUAT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))


def _rotate(q, v):
    x, y, z, w = _quat_normalize(q)
    cx = y * v[2] - z * v[1]
    cy = z * v[0] - x * v[2]
    cz = x * v[1] - y * v[0]
    return (
        v[0] + 2 * (w * cx + y * cz - z * cy),
        v[1] + 2 * (w * cy + z * cx - x * cz),
        v[2] + 2 * (w * cz + x * cy - y * cx),
    )


def compose_pose(parent: Pose7, child: Pose7) -> Pose7:
    """Compose child expressed in parent axes into the env-0 world frame."""
    translated = _rotate(parent[3:7], child[:3])
    quaternion = _quat_normalize(_quat_mul(parent[3:7], child[3:7]))
    return (*tuple(parent[i] + translated[i] for i in range(3)), *quaternion)


@dataclass(frozen=True)
class BodyInit:
    root_pose: Pose7
    root_velocity: Velocity6 = ZERO_VEL


@dataclass(frozen=True)
class HandInit(BodyInit):
    joint_position: tuple[float, ...] = ()  # canonical order from Hand2Spec
    joint_velocity: tuple[float, ...] = ()
    wrist_target_pose: Pose7 = (0, 0, 0, 0, 0, 0, 1)
    joint_target: tuple[float, ...] = ()


@dataclass(frozen=True)
class GraspInitSample:
    mode: Mode
    seed: int
    source: str
    left: HandInit
    right: HandInit
    bottle: BodyInit
    cap: BodyInit
    thread_initial_engagement_m: float
    thread_initial_angle_rad: float
    metadata: dict[str, object]
    validated: bool = False
    validation: dict[str, object] | None = None

    def with_validation(self, metrics: dict[str, object]) -> "GraspInitSample":
        if self.mode != "stable_pregrasp" or metrics.get("passed") is not True:
            raise ValueError("Only a passing stable_pregrasp PhysX hold may be validated")
        if metrics.get("fixture") != "gravity_on_no_support":
            raise ValueError("A gravity-on no-support fixture is required")
        limits = {
            "duration_s": (1.0, math.inf),
            "hold_displacement_m": (0.0, 0.02),
            "hold_rotation_rad": (0.0, math.radians(10)),
            "left_only_displacement_m": (0.0, 0.02),
            "left_contact_ratio": (0.8, 1.0),
            "right_contact_ratio": (0.8, 1.0),
            "left_only_contact_ratio": (0.8, 1.0),
            "ground_contact_ratio": (0.0, 0.0),
        }
        for name, (lower, upper) in limits.items():
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing finite PhysX validation metric: {name}")
            if name in ("ground_contact_ratio",):
                valid = value == 0.0
            elif name in ("duration_s",):
                valid = value >= lower
            elif name.endswith("contact_ratio"):
                valid = lower < value <= upper
            else:
                valid = lower <= value < upper
            if not valid:
                raise ValueError(f"Grasp validation metric failed: {name}={value}")
        peak = metrics.get("max_contact_peak_n")
        guard = metrics.get("max_peak_force_guard_n")
        if (not isinstance(peak, (int, float)) or not isinstance(guard, (int, float))
                or not math.isfinite(peak) or not math.isfinite(guard)
                or guard <= 0 or guard > 100.0 or peak < 0 or peak >= guard):
            raise ValueError("Contact peak failed the configured finite force guard")
        separation = metrics.get("initial_min_separation_m")
        depth_guard = metrics.get("max_initial_penetration_guard_m")
        if (not isinstance(separation, (int, float))
                or not isinstance(depth_guard, (int, float))
                or not math.isfinite(separation) or not math.isfinite(depth_guard)
                or depth_guard <= 0 or depth_guard > 0.002
                or separation < -depth_guard):
            raise ValueError("Initial contact penetration exceeds the 2 mm hard guard")
        thumb_time = metrics.get("left_thumb_contact_time_s")
        other_count = metrics.get("left_other_finger_count")
        if (not isinstance(thumb_time, (int, float)) or not math.isfinite(thumb_time)
                or thumb_time < 0.05 or not isinstance(other_count, int)
                or other_count < 2):
            raise ValueError("Left grasp lacks thumb and two other finger contacts")
        if metrics.get("support_released") is not True:
            raise ValueError("Candidate support must be removed before acceptance")
        if metrics.get("reset_replay_passed") is not True:
            raise ValueError("An independently reset one-second replay is required")
        return replace(self, validated=True, validation=metrics)


class BimanualGraspInitializer:
    """Generate candidates using rigid transforms relative to bottle and cap."""

    def __init__(self, bottle: BottleConfig, left_spec, right_spec):
        if left_spec.side != "left" or right_spec.side != "right":
            raise ValueError("Use official left and right hand specs in matching roles")
        if left_spec.variant != right_spec.variant:
            raise ValueError("Left and right hand variants must match")
        self.bottle = bottle
        self.left_spec = left_spec
        self.right_spec = right_spec

    def generate(
        self, *, mode: Mode, seed: int,
        bottle_pose: Pose7 = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
    ) -> GraspInitSample:
        if mode not in ("stable_pregrasp", "weak_contact", "near_contact"):
            raise ValueError(f"Unknown grasp initialization mode: {mode}")
        if len(bottle_pose) != 7 or not all(math.isfinite(x) for x in bottle_pose):
            raise ValueError("Bottle pose must contain seven finite values")
        bottle_pose = (*bottle_pose[:3], *_quat_normalize(bottle_pose[3:7]))
        rng = random.Random(seed)
        cap_pose = compose_pose(bottle_pose, (0, 0, self.bottle.cap_initial_z, 0, 0, 0, 1))
        # Approximate candidates from measured default index distal positions in
        # the pinned Beta 1 USD. GPU contact/hold validation decides acceptance.
        offset = {"stable_pregrasp": 0.0, "weak_contact": 0.007,
                  "near_contact": 0.025}[mode]
        left_local = (-0.110 - offset + rng.uniform(-0.002, 0.002),
                      0.030 + rng.uniform(-0.003, 0.003), -0.034 + rng.uniform(-0.002, 0.002),
                      *LEFT_LOCAL_QUAT)
        right_local = (0.110 + offset + rng.uniform(-0.002, 0.002),
                       -0.030 + rng.uniform(-0.003, 0.003), 0.036 + rng.uniform(-0.002, 0.002),
                       *RIGHT_LOCAL_QUAT)
        left_pose = compose_pose(bottle_pose, left_local)
        right_pose = compose_pose(cap_pose, right_local)
        grip = {"stable_pregrasp": 0.55, "weak_contact": 0.25,
                "near_contact": 0.0}[mode]

        def hand(spec, pose):
            joints = tuple(
                max(j.lower, min(j.upper, j.upper * grip)) if not j.name.endswith("_abd")
                else max(j.lower, min(j.upper, 0.65 if j.name.endswith("thumb_cmc_abd") else 0.0))
                for j in spec.joints
            )
            return HandInit(pose, ZERO_VEL, joints, (0.0,) * 20, pose, joints)

        metadata = {
            "schema": "dexgym-bimanual-grasp-init-v1",
            "coordinates": "env0_world_z_up_xyzw; hand offsets bottle/cap local",
            "units": "m,rad,s,N",
            "asset_variant": self.left_spec.variant,
            "asset_source_commit": self.left_spec.source_commit,
            "left_usd_sha256": self.left_spec.usd_sha256,
            "right_usd_sha256": self.right_spec.usd_sha256,
            "left_asset_tree_sha256": self.left_spec.asset_tree_sha256,
            "right_asset_tree_sha256": self.right_spec.asset_tree_sha256,
            "beta1_tip_marker_mass_override_kg": (
                BETA1_TIP_MARKER_MASS_KG if self.left_spec.variant == "beta1" else None),
            "hardware_candidate_unverified": True,
            "left_joint_order": list(self.left_spec.joint_names),
            "right_joint_order": list(self.right_spec.joint_names),
            "bottle_profile": self.bottle.source_profile,
            "bottle_parameters": asdict(self.bottle),
        }
        return GraspInitSample(
            mode, seed, "bottle_relative_geometry_candidate",
            hand(self.left_spec, left_pose), hand(self.right_spec, right_pose),
            BodyInit(bottle_pose), BodyInit(cap_pose), self.bottle.initial_engagement,
            0.0, metadata,
        )


class GraspInitBank:
    """Persist only gravity-on validated samples; never promote a candidate by name."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, sample: GraspInitSample) -> None:
        if not sample.validated or sample.validation is None:
            raise ValueError("Only validated grasp initial states belong in a bank")
        sample.with_validation(sample.validation)
        existing = [asdict(prior) for prior in self.load()]
        existing.append(asdict(sample))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def load(self) -> tuple[GraspInitSample, ...]:
        if not self.path.exists():
            return ()
        result = []
        for item in json.loads(self.path.read_text(encoding="utf-8")):
            sample = grasp_sample_from_dict(item)
            if not sample.validated or sample.validation is None:
                raise ValueError("Bank contains an unvalidated sample")
            sample.with_validation(sample.validation)
            result.append(sample)
        return tuple(result)


def grasp_sample_from_dict(item: dict) -> GraspInitSample:
    """Deserialize a candidate or bank entry without promoting validation."""
    def body(data):
        return BodyInit(tuple(data["root_pose"]), tuple(data["root_velocity"]))
    def hand(data):
        return HandInit(tuple(data["root_pose"]), tuple(data["root_velocity"]),
                        tuple(data["joint_position"]), tuple(data["joint_velocity"]),
                        tuple(data["wrist_target_pose"]), tuple(data["joint_target"]))
    return GraspInitSample(
        item["mode"], item["seed"], item["source"], hand(item["left"]),
        hand(item["right"]), body(item["bottle"]), body(item["cap"]),
        item["thread_initial_engagement_m"], item["thread_initial_angle_rad"],
        item["metadata"], item.get("validated", False), item.get("validation"),
    )
