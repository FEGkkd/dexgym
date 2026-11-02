"""Live WujiHand2 fingertip marker poses and bottle-axis contact diagnostics.

The official Beta 1 and Beta 2 USDs have five fixed ``*_tip`` rigid-body frames. Their
``body_link_pose_w`` values come from the current PhysX articulation update;
distal body origins and stale USD stage transforms are never substituted.
"""

from __future__ import annotations

import math

from dexteroushands_lab.assets.wuji_hand2_beta import FINGER_ORDER
from dexteroushands_lab.assets.wrist_6dof import _quat_mul, _quat_normalize


def _rotate(quat_xyzw, vector):
    q = _quat_normalize(quat_xyzw)
    rotated = _quat_mul(_quat_mul(q, (*vector, 0.0)),
                        (-q[0], -q[1], -q[2], q[3]))
    return rotated[:3]


def bottle_axis_radial_distance(point_w, bottle_pose_xyzw):
    """Return point distance from bottle-local +Z, irrespective of world tilt."""
    axis = _rotate(bottle_pose_xyzw[3:7], (0.0, 0.0, 1.0))
    offset = tuple(a - b for a, b in zip(point_w, bottle_pose_xyzw[:3]))
    axial = sum(a*b for a, b in zip(offset, axis))
    radial = tuple(a - axial*b for a, b in zip(offset, axis))
    return math.sqrt(sum(x*x for x in radial)), axial


class FingertipSiteTracker:
    """Resolve all five official tip-frame indices once and require live data."""

    def __init__(self, hand, spec):
        if spec.variant not in ("beta1", "beta2"):
            raise ValueError("Unsupported WujiHand2 fingertip variant")
        prefix = "l" if spec.side == "left" else "r"
        self.hand = hand
        self.spec = spec
        self.names = tuple(f"{prefix}_{digit}_tip" for digit in FINGER_ORDER)
        missing = [name for name in self.names if name not in hand.body_names]
        if missing:
            raise RuntimeError(f"Missing live PhysX fingertip frames: {missing}")
        self.indices = tuple(hand.body_names.index(name) for name in self.names)
        self.colliders = dict(zip(FINGER_ORDER, spec.fingertip_bodies))

    def observe(self, env_id, bottle_pose_xyzw, bottle_radius, raw_contacts=()):
        if bottle_radius <= 0:
            raise ValueError("Bottle radius must be positive")
        poses = self.hand.data.body_link_pose_w.torch[env_id, self.indices, :3]
        positions = poses.detach().cpu().tolist()
        if len(positions) != 5 or any(not math.isfinite(x) for p in positions for x in p):
            raise RuntimeError("Live PhysX fingertip positions are missing or nonfinite")
        output = {}
        for digit, name, position in zip(FINGER_ORDER, self.names, positions):
            radial, axial = bottle_axis_radial_distance(position, bottle_pose_xyzw)
            contacts = [item for item in raw_contacts
                        if item.get("category") in ("left_bottle", "right_bottle")
                        and any(f"_{digit}_" in body for body in
                                (item.get("source_body", ""), item.get("partner_body", "")))]
            output[digit] = {
                "site_body": name,
                "distal_collision_body": self.colliders[digit],
                "site_position_w_m": position,
                "radial_distance_to_bottle_axis_m": radial,
                "axial_position_bottle_local_m": axial,
                "signed_distance_to_cylindrical_surface_m": radial - bottle_radius,
                "contact_points": contacts,
            }
        return output
