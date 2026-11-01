"""Candidate local hand shapes from screened natural Wuji pregrasps."""

from dataclasses import replace
import math
import random

from dexteroushands_lab.assets.bottle.mimic_contract import mimic_reset_geometry
from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.scenes.grasp_init import (
    ZERO_VEL, _rotate, _inverse_pose, compose_pose,
)


LEVELS = {
    "A": {"left_inward": (0., .002), "right_inward": (0., .003),
          "left_blend": (.18, .22), "right_blend": (.08, .12)},
    "B": {"left_inward": (0., .001), "right_inward": (0., .002),
          "left_blend": (.04, .10), "right_blend": (0., .08)},
    "C": {"left_inward": (0., 0.), "right_inward": (0., 0.),
          "left_blend": (0., 0.), "right_blend": (0., 0.)},
}


def make_candidate(stable, natural, *, level: str, seed: int,
                   left_inward: float | None = None,
                   right_inward: float | None = None,
                   left_blend: float | None = None,
                   right_blend: float | None = None):
    """Close natural preshape locally while retaining its opposed tripod.

    This proposes a local relative grasp sample. No collision or hold claim
    transfers from the natural template; both geometry and reach require a
    fresh check before admission to the training bank.
    """
    if level not in LEVELS or not stable.validated or natural.validated:
        raise ValueError("Expected a validated PGS state and unheld natural template")
    if stable.metadata.get("solver_type") != "pgs":
        raise ValueError("PGS Mimic source state required")
    geometry = mimic_reset_geometry(natural.bottle.root_pose, natural.cap.root_pose,
                                     natural.thread_initial_angle_rad, BottleConfig())
    if (abs(geometry["pitch_error_m"]) > 5e-5 or
            abs(geometry["twist_error_rad"]) > .01):
        raise ValueError("Natural source screw state is incompatible with Mimic")
    rng = random.Random(seed)
    supplied = {"left_inward": left_inward, "right_inward": right_inward,
                "left_blend": left_blend, "right_blend": right_blend}
    params = {}
    for name, value in supplied.items():
        lo, hi = LEVELS[level][name]
        value = rng.uniform(lo, hi) if value is None else value
        if not 0 <= value <= (0.005 if "inward" in name else .6):
            raise ValueError(f"{name} is outside the candidate search range")
        params[name] = value
    axis = _rotate(natural.bottle.root_pose[3:], (0., 0., 1.))

    def modify_hand(name):
        hand = getattr(natural, name)
        target = getattr(stable, name)
        origin = natural.bottle.root_pose if name == "left" else natural.cap.root_pose
        vector = tuple(hand.root_pose[i] - origin[i] for i in range(3))
        projection = sum(vector[i] * axis[i] for i in range(3))
        radial = tuple(vector[i] - projection * axis[i] for i in range(3))
        norm = math.sqrt(sum(x*x for x in radial))
        if norm < .01:
            raise ValueError("Hand wrist lacks a radial direction")
        inward = params[f"{name}_inward"]
        pose = (*tuple(hand.root_pose[i] - inward * radial[i] / norm
                       for i in range(3)), *hand.root_pose[3:])
        blend = params[f"{name}_blend"]
        joints = tuple((1 - blend) * a + blend * b
                       for a, b in zip(hand.joint_position, target.joint_position))
        return replace(hand, root_pose=pose, root_velocity=ZERO_VEL,
                       wrist_target_pose=pose, joint_position=joints,
                       joint_velocity=(0.,) * 20, joint_target=joints)

    left = modify_hand("left")
    right = modify_hand("right")
    metadata = {key: value for key, value in natural.metadata.items()
                if key != "geometry_evidence"}
    metadata.update({
        "schema": "dexgym-phase1b-local-pregrasp-candidate-v1",
        "geometry_qualified": False,
        "hold_validated": False,
        "phase1b_reset_level": level,
        "local_shape_seed": seed,
        "derived_from_natural_pregrasp_family": True,
        "natural_source_seed": natural.seed,
        "stable_reference_seed": stable.seed,
        **params,
    })
    return replace(natural, mode="geometry_pregrasp", seed=seed,
                   source="local_closure_of_official_screened_natural_pregrasp",
                   left=left, right=right,
                   metadata=metadata, validated=False, validation=None)


def make_hybrid_candidate(stable, natural, *, seed: int,
                          left_open_blend: float,
                          left_outward_m: float,
                          right_inward_m: float,
                          level: str = "A"):
    """Pair recoverable left contact with a genuinely opposed right pregrasp.

    This is still an unqualified candidate. The left finger state is opened
    from a physically checked PGS support, while the right hand comes from an
    independent natural local shape with its cap-relative pose preserved.
    """
    if (level not in LEVELS or not stable.validated or natural.validated
            or stable.metadata.get("solver_type") != "pgs"
            or not 0 <= left_open_blend <= .3
            or not 0 <= left_outward_m <= .002
            or not 0 <= right_inward_m <= .005):
        raise ValueError("Invalid hybrid pregrasp source or local perturbation")
    axis = _rotate(stable.bottle.root_pose[3:], (0., 0., 1.))

    def radial_unit(pose, origin):
        vector = tuple(pose[i] - origin[i] for i in range(3))
        axial = sum(vector[i] * axis[i] for i in range(3))
        radial = tuple(vector[i] - axial * axis[i] for i in range(3))
        norm = math.sqrt(sum(x*x for x in radial))
        if norm < .01:
            raise ValueError("Hybrid wrist radial direction is undefined")
        return tuple(x / norm for x in radial)

    left_direction = radial_unit(stable.left.root_pose, stable.bottle.root_pose)
    left_pose = (*tuple(stable.left.root_pose[i] + left_outward_m * left_direction[i]
                        for i in range(3)), *stable.left.root_pose[3:])
    left_q = tuple((1 - left_open_blend) * a + left_open_blend * b
                   for a, b in zip(stable.left.joint_position, natural.left.joint_position))
    left = replace(stable.left, root_pose=left_pose, wrist_target_pose=left_pose,
                   root_velocity=ZERO_VEL, joint_position=left_q,
                   joint_target=left_q, joint_velocity=(0.,) * 20)
    right_local = compose_pose(_inverse_pose(natural.cap.root_pose),
                               natural.right.root_pose)
    right_pose = compose_pose(stable.cap.root_pose, right_local)
    right_direction = radial_unit(right_pose, stable.cap.root_pose)
    right_pose = (*tuple(right_pose[i] - right_inward_m * right_direction[i]
                         for i in range(3)), *right_pose[3:])
    right = replace(natural.right, root_pose=right_pose,
                    wrist_target_pose=right_pose, root_velocity=ZERO_VEL,
                    joint_velocity=(0.,) * 20)
    metadata = {key: value for key, value in stable.metadata.items()
                if key not in ("geometry_evidence", "source_probe", "source_replay",
                               "random_draw")}
    metadata.update({
        "schema": "dexgym-phase1b-hybrid-pregrasp-candidate-v1",
        "geometry_qualified": False, "hold_validated": False,
        "phase1b_reset_level": level, "local_shape_seed": seed,
        "left_open_blend": left_open_blend,
        "left_outward_m": left_outward_m,
        "right_inward_m": right_inward_m,
        "right_natural_source_seed": natural.seed,
        "stable_reference_seed": stable.seed,
        "right_local_tripod_from_natural_family": True,
    })
    return replace(stable, mode="geometry_pregrasp", seed=seed,
                   source="hybrid_left_recoverable_right_natural_tripod_proposal",
                   left=left, right=right, metadata=metadata,
                   validated=False, validation=None)
