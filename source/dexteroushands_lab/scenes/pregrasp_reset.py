"""Bottle-relative, randomized geometric pregrasp Reset states.

This module has no Isaac or MuJoCo runtime dependency. `draw` proposes a
state; only an offline native-USD geometry checker may put it in the separate
PregraspResetBank. Geometric admission never asserts a gravity-on hold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import replace
from importlib.resources import files
import json
import math
from pathlib import Path
import random

from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.wrist_pose_control import _exp_rotation
from dexteroushands_lab.assets.wrist_6dof import _quat_mul, _quat_normalize
from dexteroushands_lab.assets.wuji_hand2_beta import BETA1_TIP_MARKER_MASS_KG
from dexteroushands_lab.scenes.grasp_init import (
    BodyInit, GraspInitSample, HandInit, Pose7, ZERO_VEL, compose_pose,
    grasp_sample_from_dict, _reframe_sample,
)


TEMPLATE_IDS = ("left_cylindrical_pregrasp", "right_cap_tripod_pregrasp")


def has_pregrasp_geometry_evidence(sample: GraspInitSample) -> bool:
    """Require both the source-mesh and native-USD geometry checks."""
    evidence = sample.metadata.get("geometry_evidence")
    return (isinstance(evidence, dict) and
            isinstance(evidence.get("official_mjcf"), dict) and
            isinstance(evidence.get("native_physx"), dict) and
            evidence["official_mjcf"].get("qualified") is True and
            evidence["native_physx"].get("qualified") is True and
            evidence["official_mjcf"].get("backend") ==
            "official_beta1_MJCF_mesh_and_engineering_cylinder_proxy" and
            evidence["native_physx"].get("backend") ==
            "official_native_USD_PhysX_one_step_contact")


@dataclass(frozen=True)
class PregraspTemplate:
    template_id: str
    object_frame: str
    wrist_local_pose_xyzw: Pose7
    joint_center_rad: dict[str, float]
    wrist_translation_halfwidth_m: tuple[float, float, float]
    wrist_rotation_halfwidth_rad: tuple[float, float, float]
    joint_halfwidth_rad: float

    @classmethod
    def load(cls, template_id: str) -> "PregraspTemplate":
        if template_id not in TEMPLATE_IDS:
            raise ValueError(f"Unknown pregrasp template: {template_id}")
        data = json.loads(files("dexteroushands_lab.scenes").joinpath(
            f"presets/{template_id}.json").read_text(encoding="utf-8"))
        if data["template_id"] != template_id:
            raise ValueError("Pregrasp template ID does not match its resource")
        result = cls(
            template_id, data["object_frame"],
            tuple(data["wrist_local_pose_xyzw"]), dict(data["joint_center_rad"]),
            tuple(data["wrist_translation_halfwidth_m"]),
            tuple(data["wrist_rotation_halfwidth_rad"]),
            float(data["joint_halfwidth_rad"]),
        )
        if (result.object_frame not in ("bottle", "cap")
                or len(result.wrist_local_pose_xyzw) != 7
                or len(result.wrist_translation_halfwidth_m) != 3
                or len(result.wrist_rotation_halfwidth_rad) != 3
                or result.joint_halfwidth_rad <= 0
                or any(not math.isfinite(x) or x < 0 for x in (
                    *result.wrist_translation_halfwidth_m,
                    *result.wrist_rotation_halfwidth_rad))):
            raise ValueError("Invalid pregrasp template ranges")
        return result


class PregraspResetGenerator:
    """Draw finite candidate states from two pinned anatomical templates.

    The four left long digits and the right thumb/index/middle are sampled
    independently. No joint is clamped: an out-of-range draw is rejected.
    `draw` is not a collision certificate. Use the offline checker and bank.
    """

    def __init__(self, bottle: BottleConfig, left_spec, right_spec):
        if (left_spec.side, right_spec.side) != ("left", "right"):
            raise ValueError("Pregrasp roles require official left/right assets")
        if left_spec.variant != right_spec.variant:
            raise ValueError("Hand variants must match")
        if left_spec.variant != "beta1":
            raise ValueError("These collision-screened pregrasp templates require Beta 1")
        if bottle.source_profile != "engineering_example" or not math.isclose(
                bottle.bottle_radius, .035, abs_tol=1e-9):
            raise ValueError("These pregrasp templates target the 70 mm engineering bottle")
        self.bottle = bottle
        self.specs = (left_spec, right_spec)
        self.templates = tuple(PregraspTemplate.load(name) for name in TEMPLATE_IDS)
        for template, spec in zip(self.templates, self.specs):
            if set(template.joint_center_rad) != set(spec.joint_names):
                raise ValueError(f"Template joint map differs from {spec.side} canonical order")
            for joint in spec.joints:
                center = template.joint_center_rad[joint.name]
                if not math.isfinite(center) or not joint.lower <= center <= joint.upper:
                    raise ValueError(f"Template center exceeds official limit: {joint.name}")

    def draw(self, *, seed: int, bottle_pose: Pose7 = (0, 0, .5, 0, 0, 0, 1)) -> GraspInitSample:
        """Propose a deterministic candidate; no geometry/hold claim is made."""
        if (len(bottle_pose) != 7 or any(not math.isfinite(x) for x in bottle_pose)
                or bottle_pose[2] < self.bottle.bottle_height / 2 + .05):
            raise ValueError("Bottle pose is nonfinite or too close to the ground")
        bottle_pose = (*bottle_pose[:3], *_quat_normalize(bottle_pose[3:7]))
        cap_pose = compose_pose(bottle_pose, (0, 0, self.bottle.cap_initial_z, 0, 0, 0, 1))
        rng = random.Random(seed)
        hands = []
        draws = {}
        for template, spec in zip(self.templates, self.specs):
            offset = tuple(rng.uniform(-r, r) for r in template.wrist_translation_halfwidth_m)
            rotation = tuple(rng.uniform(-r, r) for r in template.wrist_rotation_halfwidth_rad)
            local = template.wrist_local_pose_xyzw
            local_pose = (*tuple(a + b for a, b in zip(local[:3], offset)),
                          *_quat_normalize(_quat_mul(local[3:7], _exp_rotation(rotation))))
            pose = compose_pose(bottle_pose if template.object_frame == "bottle" else cap_pose,
                                local_pose)
            values = []
            for joint in spec.joints:
                center = template.joint_center_rad[joint.name]
                value = center + rng.uniform(-template.joint_halfwidth_rad,
                                             template.joint_halfwidth_rad)
                if not joint.lower <= value <= joint.upper:
                    raise ValueError(f"Randomized joint exceeds official limit: {joint.name}")
                values.append(value)
            q = tuple(values)
            hands.append(HandInit(pose, ZERO_VEL, q, (0.0,) * 20, pose, q))
            draws[spec.side] = {"wrist_translation_local_m": list(offset),
                                "wrist_rotation_local_rad": list(rotation),
                                "joint_position_rad": list(q)}
        metadata = {
            "schema": "dexgym-bimanual-pregrasp-reset-v1",
            "coordinates": "env0_world_z_up_xyzw; wrists in bottle/cap local frames",
            "units": "m,rad,s,N",
            "asset_variant": self.specs[0].variant,
            "asset_source_commit": self.specs[0].source_commit,
            "left_usd_sha256": self.specs[0].usd_sha256,
            "right_usd_sha256": self.specs[1].usd_sha256,
            "left_asset_tree_sha256": self.specs[0].asset_tree_sha256,
            "right_asset_tree_sha256": self.specs[1].asset_tree_sha256,
            "beta1_tip_marker_mass_override_kg": (
                BETA1_TIP_MARKER_MASS_KG if self.specs[0].variant == "beta1" else None),
            "hardware_candidate_unverified": True,
            "left_joint_order": list(self.specs[0].joint_names),
            "right_joint_order": list(self.specs[1].joint_names),
            "bottle_profile": self.bottle.source_profile,
            "bottle_parameters": asdict(self.bottle),
            "wrist_control_mode": "pose_command_wrist",
            "pregrasp_template_ids": list(TEMPLATE_IDS),
            "template_ranges": {t.template_id: {
                "wrist_translation_halfwidth_m": list(t.wrist_translation_halfwidth_m),
                "wrist_rotation_halfwidth_rad": list(t.wrist_rotation_halfwidth_rad),
                "joint_halfwidth_rad": t.joint_halfwidth_rad,
            } for t in self.templates},
            "random_draw": draws,
            "geometry_qualified": False,
            "hold_validated": False,
        }
        return GraspInitSample(
            "geometry_pregrasp", seed, "randomized_anatomical_pregrasp_proposal",
            hands[0], hands[1], BodyInit(bottle_pose), BodyInit(cap_pose),
            self.bottle.initial_engagement, 0.0, metadata,
        )


def validate_pregrasp_state(sample: GraspInitSample, bottle: BottleConfig,
                            left_spec, right_spec) -> None:
    """Check exact Reset state agreement before any collision computation."""
    if sample.mode != "geometry_pregrasp" or sample.validated or sample.validation is not None:
        raise ValueError("Pregrasp state must not claim a one-second hold")
    if sample.metadata.get("pregrasp_template_ids") != list(TEMPLATE_IDS):
        raise ValueError("Unknown pregrasp topology")
    if sample.metadata.get("bottle_parameters") != asdict(bottle):
        raise ValueError("Bottle parameters differ from pregrasp template")
    if sample.metadata.get("wrist_control_mode") != "pose_command_wrist":
        raise ValueError("Randomized pregrasp requires pose-command wrists")
    if sample.metadata.get("geometry_qualified") is True and not has_pregrasp_geometry_evidence(sample):
        raise ValueError("Geometry-qualified pregrasp lacks both backend certificates")
    expected_cap = compose_pose(sample.bottle.root_pose,
                                (0, 0, bottle.cap_initial_z, 0, 0, 0, 1))
    if (math.dist(sample.cap.root_pose[:3], expected_cap[:3]) > 1e-6
            or abs(sum(a*b for a, b in zip(sample.cap.root_pose[3:7],
                                            expected_cap[3:7]))) < 1-1e-6
            or sample.thread_initial_angle_rad != 0
            or not math.isclose(sample.thread_initial_engagement_m,
                                bottle.initial_engagement, abs_tol=1e-8)):
        raise ValueError("Cap pose and thread engagement disagree with bottle frame")
    for hand, spec in ((sample.left, left_spec), (sample.right, right_spec)):
        if (len(hand.joint_position) != 20 or len(hand.joint_target) != 20
                or tuple(hand.joint_position) != tuple(hand.joint_target)
                or hand.root_pose != hand.wrist_target_pose
                or hand.root_velocity != ZERO_VEL
                or tuple(hand.joint_velocity) != (0.0,) * 20):
            raise ValueError("Pregrasp physical and controller states differ")
        if sample.metadata.get(f"{spec.side}_joint_order") != list(spec.joint_names):
            raise ValueError("Pregrasp joint order differs from official USD")
        for value, joint in zip(hand.joint_position, spec.joints):
            if not math.isfinite(value) or not joint.lower <= value <= joint.upper:
                raise ValueError(f"Pregrasp violates official limit: {joint.name}")


class PregraspResetBank:
    """Separate geometry-qualified bank; never writes the held-grasp bank."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def add(self, sample: GraspInitSample) -> None:
        if (sample.mode != "geometry_pregrasp" or sample.validated
                or sample.metadata.get("geometry_qualified") is not True
                or not has_pregrasp_geometry_evidence(sample)):
            raise ValueError("Only geometry-qualified, unheld pregrasps enter this bank")
        rows = [asdict(item) for item in self.load()]
        rows.append(asdict(sample))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        temporary.replace(self.path)

    def load(self) -> tuple[GraspInitSample, ...]:
        if not self.path.exists():
            return ()
        samples = tuple(grasp_sample_from_dict(item) for item in json.loads(
            self.path.read_text(encoding="utf-8")))
        if any(sample.mode != "geometry_pregrasp" or sample.validated or
               sample.metadata.get("geometry_qualified") is not True or
               not has_pregrasp_geometry_evidence(sample) for sample in samples):
            raise ValueError("Pregrasp bank contains an unqualified or held sample")
        return samples

    def select(self, *, seed: int, bottle_position_xyz=(0.0, 0.0, 0.5),
               bottle_yaw_rad: float = 0.0) -> GraspInitSample:
        """Pick a screened local preshape and rigidly reframe the full assembly.

        Translation and world yaw preserve internal geometry and collision
        gaps. Pitch/roll and new joint jitter require a fresh offline screen.
        The world floor is guarded separately after translation.
        """
        samples = self.load()
        if not samples:
            raise ValueError("Pregrasp Reset bank is empty")
        if (len(bottle_position_xyz) != 3 or
                any(not math.isfinite(x) for x in bottle_position_xyz) or
                bottle_position_xyz[2] < .2 or
                not math.isfinite(bottle_yaw_rad) or
                abs(bottle_yaw_rad) > math.pi):
            raise ValueError("Rigid pregrasp augmentation must stay above the floor")
        rng = random.Random(seed)
        original = samples[rng.randrange(len(samples))]
        yaw = bottle_yaw_rad
        target: Pose7 = (*bottle_position_xyz, 0.0, 0.0,
                         math.sin(yaw/2), math.cos(yaw/2))
        transformed = _reframe_sample(original, target)
        return replace(transformed,
                       source="rigid_world_transform_of_geometry_qualified_pregrasp",
                       metadata={**transformed.metadata,
                                 "geometry_qualified": True,
                                 "rigid_transform_preserves_collision_clearance": True,
                                 "runtime_selection_seed": seed,
                                 "runtime_yaw_rad": yaw,
                                 "requested_bottle_pose_xyzw": list(target),
                                 "rigid_reframe_requires_new_physx_validation": False})

    def draw_runtime(self, *, seed: int, center_xyz=(0.0, 0.0, 0.5),
                     xy_halfwidth_m: float = .02, z_halfwidth_m: float = .01,
                     yaw_halfwidth_rad: float = math.pi) -> GraspInitSample:
        """Randomize global pose within bounds while retaining screened gaps.

        Local wrist/joint variation comes from the bank's independently
        screened samples. This method only applies a rigid world yaw and
        translation, so it does not invent unscreened local geometry.
        """
        if (len(center_xyz) != 3 or any(not math.isfinite(x) for x in center_xyz)
                or not 0 <= xy_halfwidth_m <= .25
                or not 0 <= z_halfwidth_m <= .1
                or not 0 <= yaw_halfwidth_rad <= math.pi):
            raise ValueError("Invalid runtime pregrasp randomization range")
        rng = random.Random(seed)
        target = (center_xyz[0] + rng.uniform(-xy_halfwidth_m, xy_halfwidth_m),
                  center_xyz[1] + rng.uniform(-xy_halfwidth_m, xy_halfwidth_m),
                  center_xyz[2] + rng.uniform(-z_halfwidth_m, z_halfwidth_m))
        yaw = rng.uniform(-yaw_halfwidth_rad, yaw_halfwidth_rad)
        return self.select(seed=seed, bottle_position_xyz=target,
                           bottle_yaw_rad=yaw)
