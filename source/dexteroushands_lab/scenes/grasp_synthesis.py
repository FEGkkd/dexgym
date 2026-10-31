"""Support-assisted grasp formation followed by a gravity-on release test.

The support is a bounded world-frame wrench on the bottle during synthesis.
It is completely absent during acceptance. No object or hand root is written
between episode boundaries, and finger motion uses the normal position drives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from dexteroushands_lab.assets.wrist_6dof import _orientation_error
from dexteroushands_lab.scenes.grasp_init import (BodyInit, HandInit,
                                                  GraspInitSample, _rotate)
from dexteroushands_lab.scenes.fingertip_sites import FingertipSiteTracker
from dexteroushands_lab.scenes.antipodal_contact_analysis import (
    CylindricalPowerGraspSeries, ImageCylindricalContactSeries,
    analyze_bottle_contacts,
    analyze_cylindrical_power_grasp)


def _clip(vector, limit):
    length = math.sqrt(sum(x*x for x in vector))
    scale = min(1.0, limit / length) if length else 1.0
    return tuple(float(x * scale) for x in vector)


def interpolate_joint_waypoints(start, waypoints, fraction):
    """Interpolate a staged, position-driven finger closure without pose writes."""
    points = (tuple(start), *(tuple(point) for point in waypoints))
    if not waypoints or any(len(point) != len(points[0]) for point in points):
        raise ValueError("Closing waypoints must have the same joint count as start")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Closing fraction must be within [0, 1]")
    segment = min(int(fraction * len(waypoints)), len(waypoints) - 1)
    blend = min(1.0, fraction * len(waypoints) - segment)
    return tuple((1.0 - blend) * a + blend * b
                 for a, b in zip(points[segment], points[segment + 1]))


@dataclass(frozen=True)
class SynthesisLimits:
    approach_s: float = 0.35
    close_s: float = 0.45
    settle_s: float = 0.2
    release_s: float = 1.0
    start_clearance_m: float = 0.08
    max_penetration_m: float = 0.002
    max_contact_force_n: float = 100.0
    max_precontact_finger_speed_rad_s: float = 8.5
    max_finger_speed_rad_s: float = 25.0
    max_support_force_n: float = 10.0
    max_support_torque_nm: float = 0.4
    contact_axial_margin_m: float = 0.02


class GraspSynthesisFixture:
    """Drive a single left grasp candidate through a reproducible trajectory."""

    def __init__(self, fixture, limits: SynthesisLimits | None = None,
                 pose_compensation: bool = False, visualizer=None,
                 require_power_grasp: bool = False,
                 ramp_fixed_support: bool = False):
        if fixture.num_envs != 1:
            raise ValueError("Synthesis currently requires exactly one environment")
        self.fixture = fixture
        self.limits = limits or SynthesisLimits()
        self.pose_compensation = pose_compensation
        self.tip_tracker = FingertipSiteTracker(fixture.left, fixture.specs[0])
        self.visualizer = visualizer
        self.require_power_grasp = require_power_grasp
        if ramp_fixed_support and fixture.temporary_support_joint is None:
            raise ValueError("A fixed support joint is needed for staged support release")
        self.ramp_fixed_support = ramp_fixed_support

    def _power_series(self):
        return CylindricalPowerGraspSeries(
            bottle_radius_m=self.fixture.config.bottle_radius)

    def _image_series(self):
        return ImageCylindricalContactSeries(
            bottle_radius_m=self.fixture.config.bottle_radius)

    def _support_wrench(self, target_position):
        bottle = self.fixture.bottle.data
        pos = bottle.root_pose_w.torch[0, :3].detach().cpu().tolist()
        vel = bottle.root_vel_w.torch[0, :6].detach().cpu().tolist()
        error = tuple(a - b for a, b in zip(target_position, pos))
        force = tuple(1000.0*e - 20.0*v for e, v in zip(error, vel[:3]))
        force = (force[0], force[1], force[2] +
                 (self.fixture.config.bottle_mass + self.fixture.config.cap_mass)*9.81)
        torque = tuple(-0.06*v for v in vel[3:6])
        return (_clip(force, self.limits.max_support_force_n),
                _clip(torque, self.limits.max_support_torque_nm))

    def _finger_target(self, canonical_targets):
        import torch
        asset = self.fixture.left
        names = self.fixture.specs[0].joint_names
        target = [canonical_targets[names.index(name)] for name in asset.joint_names]
        asset.set_joint_position_target_index(
            target=torch.tensor([target], dtype=torch.float32, device=asset.device),
            env_ids=torch.tensor([0], dtype=torch.long, device=asset.device),
        )

    def _contact_snapshot(self):
        details = [{**item,
                    "body": next((body for body in
                                  (item["source_body"], item["partner_body"])
                                  if body.startswith("l_")), "unknown")}
                   for item in self.fixture.monitor.raw_contact_points(0, self.fixture.dt)
                   if item["category"] == "left_bottle"]
        minimum = min((item["separation_m"] for item in details), default=None)
        return minimum, sorted(details, key=lambda x: x["separation_m"])

    @staticmethod
    def _left_finger_contact_times(contact_summary, dt):
        times = {}
        for pair, detail in contact_summary.get("bodies", {}).items():
            hand_body = next((body for body in pair.split("|") if body.startswith("l_")), None)
            if hand_body is None:
                continue
            digit = hand_body.split("_")[1]
            times[digit] = max(times.get(digit, 0.0), detail["samples"] * dt)
        return times

    @staticmethod
    def _opposition_contact_passed(times):
        return (times.get("thumb", 0.0) >= 0.05 and
                sum(time >= 0.05 for digit, time in times.items()
                    if digit != "thumb") >= 2)

    def _record(self, t, phase, minimum, details, support):
        f = self.fixture
        bottle_pose = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        bottle_vel = f.bottle.data.root_vel_w.torch[0].detach().cpu().tolist()
        left_pose = f.left.data.root_pose_w.torch[0].detach().cpu().tolist()
        left_vel = f.left.data.root_vel_w.torch[0].detach().cpu().tolist()
        left_joint = f.left.data.joint_pos.torch[0].detach().cpu().tolist()
        left_joint_vel = f.left.data.joint_vel.torch[0].detach().cpu().tolist()
        left_joint_target = f.left.data.joint_pos_target.torch[0].detach().cpu().tolist()
        applied_joint_torque = f.left.data.applied_torque.torch[0].detach().cpu().tolist()
        bottle_contacts = f.monitor.raw_contact_points(0, f.dt)
        tips = self.tip_tracker.observe(0, bottle_pose, f.config.bottle_radius,
                                        bottle_contacts)
        if self.visualizer is not None:
            self.visualizer.update(tips, bottle_pose, f.config.bottle_radius,
                                   f.config.bottle_height, details[:64])
        coverage = self._coverage(details, bottle_pose)
        power_contact = analyze_cylindrical_power_grasp(
            details, bottle_pose, bottle_radius_m=f.config.bottle_radius)
        signed_normal_resultant = [0.0, 0.0, 0.0]
        signed_normal_moment = [0.0, 0.0, 0.0]
        friction_pairs = {}
        for item in details:
            force = [item["signed_normal_force_scalar_n"] * n
                     for n in item["normal_w"]]
            arm = [item["point_w_m"][i] - bottle_pose[i] for i in range(3)]
            moment = (arm[1]*force[2] - arm[2]*force[1],
                      arm[2]*force[0] - arm[0]*force[2],
                      arm[0]*force[1] - arm[1]*force[0])
            signed_normal_resultant = [a+b for a,b in zip(signed_normal_resultant, force)]
            signed_normal_moment = [a+b for a,b in zip(signed_normal_moment, moment)]
            if item.get("pair_friction_force_vector_w_n") is not None:
                friction_pairs[(item["source_path"], item["partner_path"])] = (
                    item["pair_friction_force_vector_w_n"])
        limits = {joint.name: joint.usd_drive_max_force for joint in f.specs[0].joints}
        saturation = sum(abs(torque) >= .99*limits[name] for name, torque in
                         zip(f.left.joint_names, applied_joint_torque))
        tip_sites = {item["site_body"]: item["site_position_w_m"]
                     for item in tips.values()}
        controller = f.controllers[0][0]
        commanded = (controller.wrench(left_pose[:3], left_pose[3:7],
                                       left_vel[:3], left_vel[3:6])
                     if hasattr(controller, "wrench") else None)
        composed = f.left.permanent_wrench_composer.out_force_b.torch[
            0, f.root_body_ids[0]].detach().cpu().tolist()
        contact = f.monitor.summary(0).get("left_bottle", {})
        return {"t_s": t, "phase": phase, "bottle_pose_w": bottle_pose,
                "bottle_velocity_w": bottle_vel, "left_pose_w": left_pose,
                "left_velocity_w": left_vel, "left_joint_position_rad": left_joint,
                "left_joint_velocity_rad_s": left_joint_vel,
                "isaaclab_applied_torque_buffer_nm": applied_joint_torque,
                "left_joint_position_target_rad": left_joint_target,
                "left_joint_target_error_rad": [a-b for a,b in
                                                zip(left_joint_target, left_joint)],
                "left_joint_torque_near_limit_count": saturation,
                "reported_signed_normal_resultant_w_n": signed_normal_resultant,
                "reported_signed_normal_moment_about_bottle_w_nm": signed_normal_moment,
                "reported_pair_friction_vectors_w_n": list(friction_pairs.values()),
                "official_tip_sites_w_m": tip_sites,
                "fingertip_observations": tips,
                "contact_coverage": coverage,
                "power_contact_analysis": power_contact,
                "wrist_target_w": (*controller.target_pos, *controller.target_quat),
                "wrist_commanded_wrench": commanded,
                "wrist_composed_force_body": composed,
                "minimum_separation_m": minimum,
                "contact_peak_n": contact.get("peak_force_n", 0.0),
                "contact_pairs": details[:64],
                "support_wrench": support,
                "temporary_support_joint_enabled": bool(
                    f.temporary_support_joint and
                    f.temporary_support_joint.GetJointEnabledAttr().Get())}

    def _capture(self, template, target_joints, source,
                 right_target_joints=None, support_kind=None):
        f = self.fixture
        def body(asset):
            pose = tuple(float(x) for x in asset.data.root_pose_w.torch[0].detach().cpu())
            velocity = tuple(float(x) for x in asset.data.root_vel_w.torch[0].detach().cpu())
            return BodyInit(pose, velocity)
        def hand(asset, spec, controller, targets):
            root = body(asset)
            order = [asset.joint_names.index(name) for name in spec.joint_names]
            joint = tuple(float(asset.data.joint_pos.torch[0, i]) for i in order)
            speed = tuple(float(asset.data.joint_vel.torch[0, i]) for i in order)
            target_pose = (*controller.target_pos, *controller.target_quat)
            return HandInit(root.root_pose, root.root_velocity, joint, speed,
                            target_pose, tuple(targets))
        left = hand(f.left, f.specs[0], f.controllers[0][0], target_joints)
        right = hand(f.right, f.specs[1], f.controllers[1][0],
                     (template.right.joint_target if right_target_joints is None
                      else right_target_joints))
        bottle = body(f.bottle)
        cap = body(f.cap)
        thread = f.couplers[0].observe(bottle.root_pose[:3], bottle.root_pose[3:7],
                                       cap.root_pose[:3], cap.root_pose[3:7])
        engagement = f.config.initial_engagement - (thread.axial - f.config.cap_initial_z)
        return replace(template, left=left, right=right, bottle=bottle, cap=cap,
                       thread_initial_engagement_m=engagement,
                       thread_initial_angle_rad=thread.theta_unwrapped,
                       source=source,
                       metadata={**template.metadata,
                                 "wrist_control_mode": f.control_mode,
                                 "requested_grasp_topology": (
                                     "cylindrical_power_grasp_v1"
                                     if self.require_power_grasp else None),
                                 "thread_coupler_state": f.couplers[0].snapshot_state(),
                                 "synthesis": {
                           "limits": asdict(self.limits), "support": (
                               support_kind if support_kind is not None else
                               "fixed_then_bounded_wrench" if self.ramp_fixed_support else
                               "temporary_world_fixed_joint" if f.temporary_support_joint
                               else "bounded_bottle_wrench"),
                           "support_removed_before_acceptance": True}})

    def _coverage(self, details, bottle_pose):
        return analyze_bottle_contacts(
            details, bottle_pose, static_friction=self.fixture.config.static_friction,
            bottle_height=self.fixture.config.bottle_height,
            contact_axial_margin_m=self.limits.contact_axial_margin_m)

    def run_left(self, *, wrist_goal, joint_goal, seed: int = 0,
                 pregrasp_joints=None, start_at_goal: bool = False,
                 joint_waypoints=None):
        """Return all measurements and an unvalidated release-state candidate."""
        f, limits = self.fixture, self.limits
        template = f.initializer.generate(mode="stable_pregrasp", seed=seed)
        fixed_wrist = f.control_mode == "fixed_wrist"
        stationary_start = fixed_wrist or start_at_goal
        start = (wrist_goal if stationary_start else
                 (wrist_goal[0] - limits.start_clearance_m, *wrist_goal[1:]))
        opened = tuple(pregrasp_joints) if pregrasp_joints is not None else (0.0,) * 20
        if len(opened) != 20 or len(joint_goal) != 20:
            raise ValueError("Synthesis requires twenty independent finger targets")
        waypoints = (tuple(tuple(point) for point in joint_waypoints)
                     if joint_waypoints is not None else (tuple(joint_goal),))
        if not waypoints or any(len(point) != 20 for point in waypoints):
            raise ValueError("Synthesis requires twenty joints per waypoint")
        if any(abs(a-b) > 1e-8 for a, b in zip(waypoints[-1], joint_goal)):
            raise ValueError("The final closing waypoint must equal joint_goal")
        left = replace(template.left, root_pose=start, wrist_target_pose=start,
                       joint_position=opened, joint_target=opened)
        right_pose = (tuple(float(x) for x in f.right.data.root_pose_w.torch[0])
                      if fixed_wrist else
                      (template.right.root_pose[0] + 0.4, *template.right.root_pose[1:]))
        right = replace(template.right, root_pose=right_pose, wrist_target_pose=right_pose,
                        joint_position=opened, joint_target=opened)
        f.reset_manager.reset([0], [replace(template, left=left, right=right)])
        if f.temporary_support_joint:
            f.temporary_support_joint.GetJointEnabledAttr().Set(True)
        initial_bottle = tuple(float(x) for x in f.bottle.data.root_pose_w.torch[0, :3])
        trace = []
        min_separation = None
        max_force = 0.0
        abort = None
        elapsed = 0.0
        target_joints = opened
        release_sample = None
        release_start_pose = None
        release_initial_separation = None
        release_coverage_steps = 0
        release_frictional_steps = 0
        release_step_count = 0
        power_series = self._power_series()
        image_series = self._image_series()
        for phase, duration in (("approach", limits.approach_s),
                                ("close", limits.close_s),
                                ("settle", limits.settle_s),
                                ("release", limits.release_s)):
            if phase == "settle" and self.ramp_fixed_support:
                f.temporary_support_joint.GetJointEnabledAttr().Set(False)
            if phase == "release":
                release_sample = self._capture(
                    template, target_joints, "support_assisted_release_start")
                release_start_pose = tuple(float(x) for x in
                    f.bottle.data.root_pose_w.torch[0].detach().cpu())
                f.monitor.reset([0])
                if f.temporary_support_joint:
                    f.temporary_support_joint.GetJointEnabledAttr().Set(False)
            steps = math.ceil(duration / f.dt)
            for step in range(steps):
                fraction = (step + 1) / steps
                if phase == "approach" and not stationary_start:
                    desired_x = start[0] + fraction*(wrist_goal[0] - start[0])
                    controller = f.controllers[0][0]
                    delta_x = desired_x - controller.target_pos[0]
                    controller.command((delta_x, 0.0, 0.0), (0.0, 0.0, 0.0))
                if phase == "close":
                    target_joints = interpolate_joint_waypoints(
                        opened, waypoints, fraction)
                    self._finger_target(target_joints)
                if (phase == "settle" and self.pose_compensation and
                        f.temporary_support_joint and step % max(1, round(.05/f.dt)) == 0):
                    actual = f.left.data.root_pose_w.torch[0].detach().cpu().tolist()
                    controller = f.controllers[0][0]
                    position_error = tuple(wrist_goal[i] - actual[i] for i in range(3))
                    world_rotation_error = _orientation_error(wrist_goal[3:7], actual[3:7])
                    q = controller.target_quat
                    local_rotation_error = _rotate((-q[0], -q[1], -q[2], q[3]),
                                                   world_rotation_error)
                    controller.command(tuple(.5*x for x in position_error),
                                       tuple(.25*x for x in local_rotation_error))
                support = (None if phase == "release" or
                           (f.temporary_support_joint and
                            not (self.ramp_fixed_support and phase == "settle"))
                           else self._support_wrench(initial_bottle))
                if phase == "settle" and support is not None:
                    # Ramp the fixture away before the measured hold. A
                    # large residual support force cannot hide a weak grasp.
                    support = tuple(tuple((1.0 - fraction)*x for x in vector)
                                    for vector in support)
                f.step(bottle_support_wrenches={0: support} if support else None)
                minimum, details = self._contact_snapshot()
                if phase == "release":
                    bottle_pose = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
                    coverage = self._coverage(details, bottle_pose)
                    power_series.add(details, bottle_pose)
                    image_series.add(details, bottle_pose)
                    release_step_count += 1
                    release_coverage_steps += int(coverage["surrounds_bottle_axis"])
                    release_frictional_steps += int(coverage["frictional_opposition"])
                if phase == "release" and step == 0:
                    release_initial_separation = minimum
                contact = f.monitor.summary(0).get("left_bottle", {})
                max_force = max(max_force, contact.get("peak_force_n", 0.0))
                if minimum is not None:
                    min_separation = minimum if min_separation is None else min(min_separation, minimum)
                elapsed += f.dt
                if step % max(1, steps // 20) == 0 or step == steps - 1:
                    trace.append(self._record(elapsed, phase, minimum, details, support))
                if (minimum is not None and minimum < -limits.max_penetration_m):
                    abort = f"penetration {minimum:.6f} m in {phase}"
                if max_force >= limits.max_contact_force_n:
                    abort = f"contact peak {max_force:.3f} N in {phase}"
                finger_speed = float(f.left.data.joint_vel.torch[0].abs().max())
                speed_guard = (limits.max_precontact_finger_speed_rad_s
                               if phase == "approach" and stationary_start
                               else limits.max_finger_speed_rad_s)
                if finger_speed > speed_guard:
                    reason = f"finger speed {finger_speed:.3f} rad/s in {phase}"
                    abort = f"{abort}; {reason}" if abort else reason
                if abort:
                    if not trace or abs(trace[-1]["t_s"] - elapsed) > 1e-12:
                        trace.append(self._record(elapsed, phase + "_abort",
                                                  minimum, details, support))
                    break
            if abort:
                break
        bottle_end = f.bottle.data.root_pose_w.torch[0, :3].detach().cpu().tolist()
        displacement = math.dist(release_start_pose[:3], bottle_end) if release_start_pose else None
        summary = f.monitor.summary(0)
        left = summary.get("left_bottle", {})
        finger_time = self._left_finger_contact_times(left, f.dt)
        coverage_ratio = (release_coverage_steps / release_step_count
                          if release_step_count else 0.0)
        frictional_ratio = (release_frictional_steps / release_step_count
                            if release_step_count else 0.0)
        if release_start_pose:
            current = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
            dot = min(1.0, abs(sum(a*b for a, b in zip(release_start_pose[3:7],
                                                       current[3:7]))))
            rotation = 2.0 * math.acos(dot)
        else:
            rotation = None
        hold_passed = (
            abort is None and release_sample is not None
            and release_initial_separation is not None
            and release_initial_separation >= -limits.max_penetration_m
            and displacement is not None and displacement < 0.02
            and rotation is not None and rotation < math.radians(10)
            and left.get("contact_ratio", 0.0) > 0.8
            # Cylindrical frictional pinches need not cover 180 degrees on
            # every substep. The gravity-on hold is the actual stability
            # proof; require sustained thumb and long-finger contact as well.
            and finger_time.get("thumb", 0.0) > 0.8*limits.release_s
            and max((time for digit, time in finger_time.items()
                     if digit != "thumb"), default=0.0) > 0.8*limits.release_s
            and self._opposition_contact_passed(finger_time)
            and summary.get("bottle_ground", {}).get("contact_ratio", 0.0) == 0
            and max_force < limits.max_contact_force_n
            and (not self.require_power_grasp or power_series.summary()["passed"]))
        replay = self.replay_left(release_sample) if hold_passed else None
        return {"seed": seed, "wrist_goal": wrist_goal, "joint_goal": joint_goal,
                "aborted": abort, "minimum_separation_m": min_separation,
                "maximum_contact_force_n": max_force,
                "bottle_displacement_m": displacement,
                "bottle_rotation_rad": rotation,
                "release_initial_min_separation_m": release_initial_separation,
                "release_contact_ratio": left.get("contact_ratio", 0.0),
                "release_finger_contact_time_s": finger_time,
                "release_opposing_contact_coverage_ratio": coverage_ratio,
                "release_frictional_opposition_ratio": frictional_ratio,
                "release_power_grasp": power_series.summary(),
                "release_image_grasp": image_series.summary(),
                "release_passed": bool(hold_passed),
                "reset_replay": replay,
                "contact_summary": summary, "trace": trace,
                "candidate": release_sample}

    def replay_left(self, sample, *, wrist_target_offset_xyz=(0.0, 0.0, 0.0)):
        """Restore the release state and repeat the unassisted hold.

        A nonzero wrist offset is an explicitly labeled diagnostic command,
        not a replay of the persisted sample or bank admission evidence.
        """
        f = self.fixture
        try:
            if f.temporary_support_joint:
                f.temporary_support_joint.GetJointEnabledAttr().Set(False)
            f.reset_manager.reset([0], [sample])
            if any(wrist_target_offset_xyz):
                if f.control_mode != "pose_command_wrist":
                    raise ValueError("Wrist-offset diagnostic needs pose-command control")
                controller = f.controllers[0][0]
                controller.set_target(
                    tuple(a+b for a,b in zip(controller.target_pos,
                                             wrist_target_offset_xyz)),
                    controller.target_quat)
        except (ValueError, RuntimeError) as exc:
            return {"passed": False, "reset_error": str(exc)}
        start = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        first_sep = None
        minimum = None
        trace = []
        coverage_steps = 0
        frictional_steps = 0
        step_count = 0
        power_series = self._power_series()
        image_series = self._image_series()
        max_thread_lateral = 0.0
        for step in range(math.ceil(self.limits.release_s / f.dt)):
            f.step()
            bottle_pose = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
            cap_pose = f.cap.data.root_pose_w.torch[0].detach().cpu().tolist()
            axis = _rotate(bottle_pose[3:7], (0.0, 0.0, 1.0))
            delta = tuple(a-b for a, b in zip(cap_pose[:3], bottle_pose[:3]))
            axial = sum(a*b for a, b in zip(delta, axis))
            max_thread_lateral = max(max_thread_lateral, math.sqrt(sum(
                (d-axial*a)**2 for d, a in zip(delta, axis))))
            sep, details = self._contact_snapshot()
            coverage = self._coverage(
                details, f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist())
            power_series.add(details,
                f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist())
            image_series.add(details,
                f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist())
            step_count += 1
            coverage_steps += int(coverage["surrounds_bottle_axis"])
            frictional_steps += int(coverage["frictional_opposition"])
            if step == 0:
                first_sep = sep
            if sep is not None:
                minimum = sep if minimum is None else min(minimum, sep)
            if step % max(1, round(0.05 / f.dt)) == 0:
                trace.append(self._record((step + 1)*f.dt, "reset_replay",
                                          sep, details, None))
        end = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        displacement = math.dist(start[:3], end[:3])
        dot = min(1.0, abs(sum(a*b for a, b in zip(start[3:7], end[3:7]))))
        rotation = 2.0 * math.acos(dot)
        summary = f.monitor.summary(0)
        left = summary.get("left_bottle", {})
        finger_time = self._left_finger_contact_times(left, f.dt)
        peak_force = left.get("peak_force_n", 0.0)
        coverage_ratio = coverage_steps / step_count if step_count else 0.0
        frictional_ratio = frictional_steps / step_count if step_count else 0.0
        return {"passed": first_sep is not None and first_sep >=
                -self.limits.max_penetration_m and displacement < 0.02 and
                rotation < math.radians(10) and left.get("contact_ratio", 0.0) > 0.8 and
                minimum is not None and minimum >= -self.limits.max_penetration_m and
                finger_time.get("thumb", 0.0) > 0.8*self.limits.release_s and
                max((time for digit, time in finger_time.items()
                     if digit != "thumb"), default=0.0) > 0.8*self.limits.release_s and
                self._opposition_contact_passed(finger_time) and
                peak_force < self.limits.max_contact_force_n and
                (not self.require_power_grasp or power_series.summary()["passed"]) and
                summary.get("bottle_ground", {}).get("contact_ratio", 0.0) == 0,
                "initial_min_separation_m": first_sep,
                "minimum_separation_m": minimum,
                "bottle_displacement_m": displacement,
                "bottle_rotation_rad": rotation,
                "left_contact_ratio": left.get("contact_ratio", 0.0),
                "left_finger_contact_time_s": finger_time,
                "opposing_contact_coverage_ratio": coverage_ratio,
                "frictional_opposition_ratio": frictional_ratio,
                "power_grasp": power_series.summary(),
                "image_grasp": image_series.summary(),
                "left_contact_peak_n": peak_force,
                "max_thread_lateral_error_m": max_thread_lateral,
                "wrist_target_offset_xyz_m": tuple(wrist_target_offset_xyz),
                "contact_summary": summary, "trace": trace}
