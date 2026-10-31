"""Bottle-local contact coverage and force distribution without a finger-count rule.

Contact-point radial directions express where the hand touches the cylindrical
body. They do not assume the backend normal's source/partner sign convention.
Friction values are reported as raw PhysX pair-level diagnostics, not as a
calibrated grasp-stability certificate.
"""

from __future__ import annotations

import math

from dexteroushands_lab.scenes.fingertip_sites import _rotate


def analyze_bottle_contacts(contacts, bottle_pose_xyzw, *, force_floor_n=0.05):
    if force_floor_n < 0:
        raise ValueError("Force floor must be nonnegative")
    center = bottle_pose_xyzw[:3]
    x_axis = _rotate(bottle_pose_xyzw[3:7], (1.0, 0.0, 0.0))
    y_axis = _rotate(bottle_pose_xyzw[3:7], (0.0, 1.0, 0.0))
    z_axis = _rotate(bottle_pose_xyzw[3:7], (0.0, 0.0, 1.0))
    accepted = []
    friction_pairs = {}
    for contact in contacts:
        if contact.get("category") != "left_bottle":
            continue
        force = float(contact["normal_force_n"])
        if not math.isfinite(force) or force < force_floor_n:
            continue
        delta = tuple(float(a)-float(b) for a,b in zip(contact["point_w_m"],center))
        x = sum(a*b for a,b in zip(delta,x_axis))
        y = sum(a*b for a,b in zip(delta,y_axis))
        radius = math.hypot(x,y)
        if radius < 1e-6:
            continue
        body = next((part for part in (contact.get("source_body",""),
                                        contact.get("partner_body",""))
                     if part.startswith("l_")), "unknown")
        accepted.append({"body":body,"angle_rad":math.atan2(y,x),
                         "normal_force_n":force,
                         "radial_unit_xy":(x/radius,y/radius),
                         "point_w_m":contact["point_w_m"],
                         "normal_w":contact["normal_w"]})
        friction = contact.get("pair_friction_force_vector_w_n")
        if friction is not None:
            key=(contact.get("source_path"),contact.get("partner_path"))
            friction_pairs[key]=tuple(float(v) for v in friction)
    angles=sorted(item["angle_rad"] % (2*math.pi) for item in accepted)
    gaps=([b-a for a,b in zip(angles,angles[1:])]
          + [angles[0]+2*math.pi-angles[-1]] if angles else [])
    total=sum(item["normal_force_n"] for item in accepted)
    weighted=(sum(item["normal_force_n"]*item["radial_unit_xy"][axis]
                  for item in accepted) for axis in (0,1))
    lateral=tuple(weighted)
    axial_friction=sum(sum(a*b for a,b in zip(force,z_axis))
                       for force in friction_pairs.values())
    return {"active_contact_count":len(accepted),
            "contact_bodies":sorted(set(item["body"] for item in accepted)),
            "maximum_angular_gap_rad":max(gaps) if gaps else 2*math.pi,
            "surrounds_bottle_axis":len(angles)>=2 and max(gaps)<=math.pi+1e-3,
            "radial_force_imbalance_ratio":(
                math.hypot(*lateral)/total if total else None),
            "total_normal_force_n":total,
            "reported_pair_axial_friction_sum_n":axial_friction,
            "contact_distribution":accepted}
