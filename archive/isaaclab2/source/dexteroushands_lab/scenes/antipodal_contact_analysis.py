"""Bottle-local contact coverage and force distribution without a finger-count rule.

Contact-point radial directions express where the hand touches the cylindrical
body. They do not assume the backend normal's source/partner sign convention.
Friction values are reported as raw PhysX pair-level diagnostics, not as a
calibrated grasp-stability certificate.
"""

from __future__ import annotations

import math

from dexteroushands_lab.scenes.fingertip_sites import _rotate


def analyze_bottle_contacts(contacts, bottle_pose_xyzw, *, force_floor_n=0.05,
                            static_friction=None, bottle_height=None,
                            contact_axial_margin_m=.02):
    if force_floor_n < 0:
        raise ValueError("Force floor must be nonnegative")
    if static_friction is not None and static_friction < 0:
        raise ValueError("Static friction must be nonnegative")
    if bottle_height is not None and not bottle_height > 2*contact_axial_margin_m:
        raise ValueError("Bottle height must exceed twice the axial margin")
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
                         "axial_position_m":sum(a*b for a,b in zip(delta,z_axis)),
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
    sidewall = [item for item in accepted if bottle_height is None or
                abs(item["axial_position_m"]) <=
                bottle_height/2-contact_axial_margin_m]
    thumb = [item for item in sidewall if "_thumb_" in item["body"]]
    opposing = [item for item in sidewall if "_thumb_" not in item["body"]]
    pair_angles = [math.acos(max(-1.0, min(1.0,
                   sum(a*b for a,b in zip(left["radial_unit_xy"],
                                          right["radial_unit_xy"])))))
                   for left in thumb for right in opposing]
    required_angle = (math.pi-2*math.atan(static_friction)
                      if static_friction is not None else None)
    return {"active_contact_count":len(accepted),
            "contact_bodies":sorted(set(item["body"] for item in accepted)),
            "maximum_angular_gap_rad":max(gaps) if gaps else 2*math.pi,
            "surrounds_bottle_axis":len(angles)>=2 and max(gaps)<=math.pi+1e-3,
            "maximum_thumb_other_angle_rad":max(pair_angles, default=None),
            "frictional_opposition_required_angle_rad":required_angle,
            "frictional_opposition":(required_angle is not None and
                                     bool(pair_angles) and
                                     max(pair_angles)>=required_angle),
            "radial_force_imbalance_ratio":(
                math.hypot(*lateral)/total if total else None),
            "total_normal_force_n":total,
            "reported_pair_axial_friction_sum_n":axial_friction,
            "contact_distribution":accepted}


def analyze_cylindrical_power_grasp(
    contacts, bottle_pose_xyzw, *, bottle_radius_m=.035,
    force_floor_n=.05, axial_band_m=(-.025, .065),
    radius_tolerance_m=.003, max_four_finger_half_arc_rad=math.radians(70),
    min_thumb_opposition_rad=math.radians(130),
    minimum_loaded_long_digits=4,
):
    """Classify *measured* opposed bottle-body contact.

    By default all four long digits must share one cylinder side and the
    thumb must lie across its axis. The image-specific series requests three
    simultaneous loaded long digits and checks four-digit placement against
    official geometry separately. Only force-bearing middle/upper sidewall
    points count. The unsupported PhysX hold and Reset replay are separate.
    """
    if bottle_radius_m <= 0 or force_floor_n < 0 or radius_tolerance_m <= 0:
        raise ValueError("Power-grasp radius, force and tolerance must be positive")
    if minimum_loaded_long_digits not in (3, 4):
        raise ValueError("Require three or four loaded long fingers")
    if not axial_band_m[0] < axial_band_m[1]:
        raise ValueError("Power-grasp axial band must be increasing")
    names = ("thumb", "index", "middle", "ring", "pinky")
    body_terms = ("_thumb_", "_index_finger_", "_middle_finger_",
                  "_ring_finger_", "_pinky_")
    center = bottle_pose_xyzw[:3]
    axes = tuple(_rotate(bottle_pose_xyzw[3:7], axis) for axis in
                 ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    vectors = {name: [0., 0.] for name in names}
    digit_forces = {name: 0. for name in names}
    loaded_force = 0.
    total_force = 0.
    for contact in contacts:
        if contact.get("category") != "left_bottle":
            continue
        force = float(contact["normal_force_n"])
        if not math.isfinite(force) or force < force_floor_n:
            continue
        total_force += force
        offset = tuple(float(a)-float(b) for a,b in
                       zip(contact["point_w_m"], center))
        local = tuple(sum(a*b for a,b in zip(offset, axis)) for axis in axes)
        radius = math.hypot(*local[:2])
        if (not axial_band_m[0] <= local[2] <= axial_band_m[1]
                or abs(radius-bottle_radius_m) > radius_tolerance_m):
            continue
        loaded_force += force
        body = next((part for part in
                     (contact.get("source_body", ""), contact.get("partner_body", ""))
                     if part.startswith("l_")), "")
        digit = next((name for name, term in zip(names, body_terms)
                      if term in body), None)
        if digit is None:
            continue  # The palm may assist, but cannot stand in for a digit.
        unit = (local[0]/radius, local[1]/radius)
        vectors[digit][0] += force*unit[0]
        vectors[digit][1] += force*unit[1]
        digit_forces[digit] += force
    loaded_digits = [name for name in names if digit_forces[name] > 0]
    units = {}
    for name in loaded_digits:
        length = math.hypot(*vectors[name])
        if length > 1e-8:
            units[name] = tuple(component/length for component in vectors[name])
    four = names[1:]
    loaded_four = [name for name in four if name in units]
    four_mean = (tuple(sum(units[name][axis] for name in loaded_four)/len(loaded_four)
                       for axis in (0, 1))
                 if len(loaded_four) >= minimum_loaded_long_digits else None)
    mean_length = math.hypot(*four_mean) if four_mean is not None else 0.
    group_direction = (tuple(component/mean_length for component in four_mean)
                       if mean_length > 1e-8 else None)
    spread = (max(math.acos(max(-1., min(1., sum(
        units[name][i]*group_direction[i] for i in (0, 1))))) for name in loaded_four)
              if group_direction is not None else None)
    opposition = (math.acos(max(-1., min(1., sum(
        units["thumb"][i]*group_direction[i] for i in (0, 1)))))
                  if group_direction is not None and "thumb" in units else None)
    in_band_ratio = loaded_force/total_force if total_force else 0.
    passed = ("thumb" in units and len(loaded_four) >= minimum_loaded_long_digits
              and spread is not None and opposition is not None
              and spread <= max_four_finger_half_arc_rad
              and opposition >= min_thumb_opposition_rad
              and in_band_ratio >= .8)
    return {"valid_topology": passed,
            "loaded_digits": loaded_digits,
            "digit_normal_force_n": digit_forces,
            "in_band_sidewall_force_ratio": in_band_ratio,
            "four_finger_half_arc_rad": spread,
            "thumb_to_four_finger_angle_rad": opposition,
            "four_finger_direction_xy": group_direction,
            "minimum_loaded_long_digits": minimum_loaded_long_digits,
            "axial_band_m": axial_band_m}


class CylindricalPowerGraspSeries:
    """Measure sustained, five-digit opposed contact over an unassisted hold."""

    def __init__(self, *, required_ratio=.8, bottle_radius_m=.035,
                 minimum_loaded_long_digits=4,
                 max_four_finger_half_arc_rad=math.radians(70)):
        if not 0 < required_ratio <= 1:
            raise ValueError("Power-grasp contact ratio must lie in (0, 1]")
        if minimum_loaded_long_digits not in (3, 4):
            raise ValueError("Require three or four loaded long fingers")
        self.required_ratio = required_ratio
        self.bottle_radius_m = bottle_radius_m
        self.minimum_loaded_long_digits = minimum_loaded_long_digits
        self.max_four_finger_half_arc_rad = max_four_finger_half_arc_rad
        self.steps = 0
        self.valid_steps = 0
        self.digit_steps = {name: 0 for name in
                            ("thumb", "index", "middle", "ring", "pinky")}
        self.peak_digit_force_n = {name: 0. for name in self.digit_steps}

    def add(self, contacts, bottle_pose_xyzw):
        observation = analyze_cylindrical_power_grasp(
            contacts, bottle_pose_xyzw, bottle_radius_m=self.bottle_radius_m,
            minimum_loaded_long_digits=self.minimum_loaded_long_digits,
            max_four_finger_half_arc_rad=self.max_four_finger_half_arc_rad)
        self.steps += 1
        self.valid_steps += int(observation["valid_topology"])
        for name in self.digit_steps:
            force = observation["digit_normal_force_n"][name]
            self.digit_steps[name] += int(force > 0.)
            self.peak_digit_force_n[name] = max(self.peak_digit_force_n[name], force)
        return observation

    def summary(self):
        ratios = {name: count/self.steps if self.steps else 0.
                  for name, count in self.digit_steps.items()}
        valid_ratio = self.valid_steps/self.steps if self.steps else 0.
        return {"step_count": self.steps,
                "valid_topology_ratio": valid_ratio,
                "digit_contact_ratio": ratios,
                "peak_digit_normal_force_n": self.peak_digit_force_n.copy(),
                "minimum_loaded_long_digits": self.minimum_loaded_long_digits,
                "passed": self.steps > 0 and valid_ratio >= self.required_ratio
                and ratios["thumb"] >= self.required_ratio
                and sum(ratios[name] >= self.required_ratio for name in
                        ("index", "middle", "ring", "pinky")) >=
                    self.minimum_loaded_long_digits}


class ImageCylindricalContactSeries(CylindricalPowerGraspSeries):
    """Measured thumb plus three simultaneous long-digit contacts.

    The unloaded long digit may change between substeps as the compliant hand
    shares load. Each of the four must also carry force in at least 5% of
    substeps. Official collision geometry must independently certify that all
    four distal pads lie on the same bottle side. The original five-digit
    series remains unchanged.
    """

    def __init__(self, *, required_ratio=.8, bottle_radius_m=.035):
        super().__init__(required_ratio=required_ratio,
                         bottle_radius_m=bottle_radius_m,
                         minimum_loaded_long_digits=3,
                         max_four_finger_half_arc_rad=math.radians(45))

    def summary(self):
        result = super().summary()
        ratios = result["digit_contact_ratio"]
        result["passed"] = (
            self.steps > 0 and result["valid_topology_ratio"] >= self.required_ratio
            and ratios["thumb"] >= self.required_ratio
            and all(ratios[name] >= .05 for name in
                    ("index", "middle", "ring", "pinky")))
        result["minimum_each_long_digit_ratio"] = .05
        return result
