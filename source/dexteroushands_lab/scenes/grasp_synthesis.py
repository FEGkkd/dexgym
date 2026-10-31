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


def _clip(vector, limit):
    length = math.sqrt(sum(x*x for x in vector))
    scale = min(1.0, limit / length) if length else 1.0
    return tuple(float(x * scale) for x in vector)


@dataclass(frozen=True)
class SynthesisLimits:
    approach_s: float = 0.35
    close_s: float = 0.45
    settle_s: float = 0.2
    release_s: float = 1.0
    start_clearance_m: float = 0.08
    max_penetration_m: float = 0.002
    max_contact_force_n: float = 100.0
    max_support_force_n: float = 10.0
    max_support_torque_nm: float = 0.4


class GraspSynthesisFixture:
    """Drive a single left grasp candidate through a reproducible trajectory."""

    def __init__(self, fixture, limits: SynthesisLimits | None = None,
                 pose_compensation: bool = False):
        if fixture.num_envs != 1:
            raise ValueError("Synthesis currently requires exactly one environment")
        self.fixture = fixture
        self.limits = limits or SynthesisLimits()
        self.pose_compensation = pose_compensation

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
        details = [{"body": item["partner_body"],
                    "separation_m": item["separation_m"],
                    "normal_force_n": item["normal_force_n"],
                    "point_w_m": item["point_w_m"],
                    "normal_w": item["normal_w"]}
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
        applied_joint_torque = f.left.data.applied_torque.torch[0].detach().cpu().tolist()
        tip_sites = {}
        for digit in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"):
            name = f"l_{digit}_tip"
            if name in f.left.body_names:
                tip_sites[name] = f.left.data.body_link_pose_w.torch[
                    0, f.left.body_names.index(name), :3].detach().cpu().tolist()
        controller = f.controllers[0][0]
        commanded = controller.wrench(left_pose[:3], left_pose[3:7],
                                      left_vel[:3], left_vel[3:6])
        composed = f.left.permanent_wrench_composer.out_force_b.torch[
            0, f.root_body_ids[0]].detach().cpu().tolist()
        contact = f.monitor.summary(0).get("left_bottle", {})
        return {"t_s": t, "phase": phase, "bottle_pose_w": bottle_pose,
                "bottle_velocity_w": bottle_vel, "left_pose_w": left_pose,
                "left_velocity_w": left_vel, "left_joint_position_rad": left_joint,
                "left_joint_velocity_rad_s": left_joint_vel,
                "isaaclab_applied_torque_buffer_nm": applied_joint_torque,
                "official_tip_sites_w_m": tip_sites,
                "wrist_target_w": (*controller.target_pos, *controller.target_quat),
                "wrist_commanded_wrench": commanded,
                "wrist_composed_force_body": composed,
                "minimum_separation_m": minimum,
                "contact_peak_n": contact.get("peak_force_n", 0.0),
                "contact_pairs": details[:10],
                "support_wrench": support,
                "temporary_support_joint_enabled": bool(
                    f.temporary_support_joint and
                    f.temporary_support_joint.GetJointEnabledAttr().Get())}

    def _capture(self, template, target_joints, source):
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
                     template.right.joint_target)
        bottle = body(f.bottle)
        cap = body(f.cap)
        thread = f.couplers[0].observe(bottle.root_pose[:3], bottle.root_pose[3:7],
                                       cap.root_pose[:3], cap.root_pose[3:7])
        return replace(template, left=left, right=right, bottle=bottle, cap=cap,
                       thread_initial_angle_rad=thread.theta_unwrapped,
                       source=source,
                       metadata={**template.metadata, "synthesis": {
                           "limits": asdict(self.limits), "support": (
                               "temporary_world_fixed_joint" if f.temporary_support_joint
                               else "bounded_bottle_wrench"),
                           "support_removed_before_acceptance": True}})

    def run_left(self, *, wrist_goal, joint_goal, seed: int = 0,
                 pregrasp_joints=None):
        """Return all measurements and an unvalidated release-state candidate."""
        f, limits = self.fixture, self.limits
        template = f.initializer.generate(mode="stable_pregrasp", seed=seed)
        start = (wrist_goal[0] - limits.start_clearance_m, *wrist_goal[1:])
        opened = tuple(pregrasp_joints) if pregrasp_joints is not None else (0.0,) * 20
        if len(opened) != 20 or len(joint_goal) != 20:
            raise ValueError("Synthesis requires twenty independent finger targets")
        left = replace(template.left, root_pose=start, wrist_target_pose=start,
                       joint_position=opened, joint_target=opened)
        right_pose = (template.right.root_pose[0] + 0.4, *template.right.root_pose[1:])
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
        for phase, duration in (("approach", limits.approach_s),
                                ("close", limits.close_s),
                                ("settle", limits.settle_s),
                                ("release", limits.release_s)):
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
                if phase == "approach":
                    desired_x = start[0] + fraction*(wrist_goal[0] - start[0])
                    controller = f.controllers[0][0]
                    delta_x = desired_x - controller.target_pos[0]
                    controller.command((delta_x, 0.0, 0.0), (0.0, 0.0, 0.0))
                if phase == "close":
                    target_joints = tuple(a + fraction*(b-a)
                                          for a, b in zip(opened, joint_goal))
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
                support = (None if phase == "release" or f.temporary_support_joint
                           else self._support_wrench(initial_bottle))
                if phase == "settle" and support is not None:
                    # Ramp the fixture away before the measured hold. A
                    # large residual support force cannot hide a weak grasp.
                    support = tuple(tuple((1.0 - fraction)*x for x in vector)
                                    for vector in support)
                f.step(bottle_support_wrenches={0: support} if support else None)
                minimum, details = self._contact_snapshot()
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
                if abort:
                    break
            if abort:
                break
        bottle_end = f.bottle.data.root_pose_w.torch[0, :3].detach().cpu().tolist()
        displacement = math.dist(release_start_pose[:3], bottle_end) if release_start_pose else None
        summary = f.monitor.summary(0)
        left = summary.get("left_bottle", {})
        finger_time = self._left_finger_contact_times(left, f.dt)
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
            and self._opposition_contact_passed(finger_time)
            and summary.get("bottle_ground", {}).get("contact_ratio", 0.0) == 0
            and max_force < limits.max_contact_force_n)
        replay = self.replay_left(release_sample) if hold_passed else None
        return {"seed": seed, "wrist_goal": wrist_goal, "joint_goal": joint_goal,
                "aborted": abort, "minimum_separation_m": min_separation,
                "maximum_contact_force_n": max_force,
                "bottle_displacement_m": displacement,
                "bottle_rotation_rad": rotation,
                "release_initial_min_separation_m": release_initial_separation,
                "release_contact_ratio": left.get("contact_ratio", 0.0),
                "release_finger_contact_time_s": finger_time,
                "release_passed": bool(hold_passed),
                "reset_replay": replay,
                "contact_summary": summary, "trace": trace,
                "candidate": release_sample}

    def replay_left(self, sample):
        """Restore the release state and repeat the unassisted one-second hold."""
        f = self.fixture
        try:
            if f.temporary_support_joint:
                f.temporary_support_joint.GetJointEnabledAttr().Set(False)
            f.reset_manager.reset([0], [sample])
        except (ValueError, RuntimeError) as exc:
            return {"passed": False, "reset_error": str(exc)}
        start = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        first_sep = None
        minimum = None
        trace = []
        for step in range(math.ceil(self.limits.release_s / f.dt)):
            f.step()
            sep, details = self._contact_snapshot()
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
        return {"passed": first_sep is not None and first_sep >=
                -self.limits.max_penetration_m and displacement < 0.02 and
                rotation < math.radians(10) and left.get("contact_ratio", 0.0) > 0.8 and
                minimum is not None and minimum >= -self.limits.max_penetration_m and
                self._opposition_contact_passed(finger_time) and
                peak_force < self.limits.max_contact_force_n and
                summary.get("bottle_ground", {}).get("contact_ratio", 0.0) == 0,
                "initial_min_separation_m": first_sep,
                "minimum_separation_m": minimum,
                "bottle_displacement_m": displacement,
                "bottle_rotation_rad": rotation,
                "left_contact_ratio": left.get("contact_ratio", 0.0),
                "left_finger_contact_time_s": finger_time,
                "left_contact_peak_n": peak_force,
                "contact_summary": summary, "trace": trace}
