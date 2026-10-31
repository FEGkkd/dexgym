"""Right-cap formation while a validated pose-command left grasp holds the bottle.

All contact and acceptance measurements come from the live official USD
articulations and PhysX. The offline MJCF proposal is only a starting pose.
No support wrench or fixture joint is used in this stage.
"""

from __future__ import annotations

from dataclasses import replace
import math

from dexteroushands_lab.scenes.grasp_init import HandInit, _rotate
from dexteroushands_lab.scenes.grasp_synthesis import GraspSynthesisFixture


def _hand_digit_times(summary, prefix, dt):
    times = {}
    for pair, detail in summary.get("bodies", {}).items():
        hand = next((name for name in pair.split("|") if name.startswith(prefix)), None)
        if hand is None:
            continue
        digit = hand.split("_")[1]
        times[digit] = max(times.get(digit, 0.0), detail["samples"] * dt)
    return times


def _right_tripod_held(times, hold_s, ratio=.8):
    """Loaded thumb, index and middle contact for the requested hold interval."""
    return all(times.get(name, 0.0) > ratio * hold_s
               for name in ("thumb", "index", "middle"))


def _staggered_close_targets(names, opened, closed, fraction, index_delay_fraction):
    """Let thumb/middle establish cap contact before the index approaches."""
    if not 0 <= fraction <= 1 or not 0 <= index_delay_fraction < 1:
        raise ValueError("Closure fraction or index delay is outside [0, 1]")
    index_fraction = min(1.0, max(0.0,
        (fraction - index_delay_fraction) / (1 - index_delay_fraction)))
    return tuple(a + (index_fraction if name.startswith("r_index_finger_")
                      else fraction) * (b - a)
                 for name, a, b in zip(names, opened, closed))


def _digit_join_targets(names, initial, goal, fraction, digit):
    """Change one long finger while preserving all other loaded targets."""
    if not 0 <= fraction <= 1:
        raise ValueError("Digit join fraction must be in [0, 1]")
    if digit not in ("index", "middle"):
        raise ValueError("Only index and middle tripod joins are supported")
    prefix = f"r_{digit}_finger_"
    return tuple(a + fraction * (goal[name] - a)
                 if name.startswith(prefix) else a
                 for name, a in zip(names, initial))


def _index_join_targets(names, initial, index_goal, fraction):
    """Backwards-compatible index-only target interpolation."""
    return _digit_join_targets(names, initial, index_goal, fraction, "index")


def _right_radial_opposition(points, cap_pose, cap_height):
    """Require loaded thumb/long-finger contacts with opposed radial normals.

    A cap rim point is accepted when its force is predominantly radial. Pure
    top pressing is rejected by the radial/axial normal checks.
    """
    axis = _rotate(cap_pose[3:7], (0.0, 0.0, 1.0))
    sides = {"thumb": [], "other": []}
    for item in points:
        if item["category"] != "right_cap" or item["normal_force_n"] < .05:
            continue
        body = next((name for name in (item["source_body"], item["partner_body"])
                     if name.startswith("r_")), "")
        offset = tuple(a-b for a,b in zip(item["point_w_m"], cap_pose[:3]))
        axial = sum(a*b for a,b in zip(offset, axis))
        if abs(axial) > cap_height/2 + .0002:
            continue
        normal = item["normal_w"]
        n_axial = sum(a*b for a,b in zip(normal, axis))
        radial = tuple(n - n_axial*a for n,a in zip(normal, axis))
        length = math.sqrt(sum(x*x for x in radial))
        if length < .8 or abs(n_axial) > .6:
            continue
        unit = tuple(x/length for x in radial)
        sides["thumb" if "thumb" in body else "other"].append(unit)
    best_dot = min((sum(a*b for a,b in zip(left,right))
                    for left in sides["thumb"] for right in sides["other"]),
                   default=None)
    return best_dot is not None and best_dot < -.3, best_dot


class BimanualGraspSynthesisFixture:
    """A one-env, gravity-on, unsupported right-cap contact experiment."""

    def __init__(self, fixture, *, max_penetration_m=.002,
                 max_contact_force_n=100.0, hold_s=1.0,
                 require_right_tripod=False):
        if fixture.num_envs != 1 or fixture.control_mode != "pose_command_wrist":
            raise ValueError("Right-cap synthesis needs one pose-command wrist environment")
        if fixture.temporary_support_joint is not None:
            raise ValueError("Right-cap synthesis cannot have temporary support")
        self.fixture = fixture
        self.max_penetration_m = max_penetration_m
        self.max_contact_force_n = max_contact_force_n
        self.hold_s = hold_s
        self.require_right_tripod = require_right_tripod
        self.left_capture = GraspSynthesisFixture(fixture)

    def _right_target(self, canonical):
        import torch
        f = self.fixture
        order = [f.specs[1].joint_names.index(name) for name in f.right.joint_names]
        target = torch.tensor([[canonical[i] for i in order]],
                              device=f.right.device, dtype=torch.float32)
        ids = torch.tensor([0], device=f.right.device, dtype=torch.long)
        f.right.set_joint_position_target_index(target=target, env_ids=ids)

    def _left_target(self, canonical):
        import torch
        f = self.fixture
        order = [f.specs[0].joint_names.index(name) for name in f.left.joint_names]
        target = torch.tensor([[canonical[i] for i in order]],
                              device=f.left.device, dtype=torch.float32)
        ids = torch.tensor([0], device=f.left.device, dtype=torch.long)
        f.left.set_joint_position_target_index(target=target, env_ids=ids)

    def _points(self):
        return [p for p in self.fixture.monitor.raw_contact_points(
            0, self.fixture.dt) if p["category"] in (
                "left_bottle", "right_cap", "left_cap", "right_bottle",
                "hand_hand", "bottle_ground", "cap_ground")]

    def _snapshot(self, t, phase):
        f = self.fixture
        points = self._points()
        summary = f.monitor.summary(0)
        return {"t_s": t, "phase": phase,
                "left_root_pose_w": f.left.data.root_pose_w.torch[0].detach().cpu().tolist(),
                "right_root_pose_w": f.right.data.root_pose_w.torch[0].detach().cpu().tolist(),
                "right_wrist_target_pose_w": list(f.controllers[1][0].target_pose),
                "bottle_pose_w": f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist(),
                "cap_pose_w": f.cap.data.root_pose_w.torch[0].detach().cpu().tolist(),
                "right_joint_order": list(f.right.joint_names),
                "right_joint_position_rad": f.right.data.joint_pos.torch[0].detach().cpu().tolist(),
                "right_joint_velocity_rad_s": f.right.data.joint_vel.torch[0].detach().cpu().tolist(),
                "right_applied_torque_nm": f.right.data.applied_torque.torch[0].detach().cpu().tolist(),
                "right_cap_contact_points": [p for p in points if p["category"] == "right_cap"],
                "cross_contact_points": [p for p in points if p["category"] in
                                         ("left_cap", "right_bottle", "hand_hand")],
                "minimum_separation_m": min((p["separation_m"] for p in points), default=None),
                "right_cap_contact_ratio": summary.get("right_cap", {}).get("contact_ratio", 0),
                "left_bottle_contact_ratio": summary.get("left_bottle", {}).get("contact_ratio", 0)}

    def _measure_hold(self, phase):
        f = self.fixture
        f.monitor.reset([0])
        start = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        first_sep = None
        minimum = None
        peak = 0.0
        opposing_steps = 0
        normal_dots = []
        max_thread_lateral = 0.0
        trace = []
        steps = math.ceil(self.hold_s / f.dt)
        for step in range(steps):
            f.step()
            points = self._points()
            bottle_pose = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
            cap_pose = f.cap.data.root_pose_w.torch[0].detach().cpu().tolist()
            opposing, dot = _right_radial_opposition(
                points, cap_pose, f.config.cap_height)
            opposing_steps += int(opposing)
            if dot is not None:
                normal_dots.append(dot)
            axis = _rotate(bottle_pose[3:7], (0.0, 0.0, 1.0))
            delta = tuple(a-b for a,b in zip(cap_pose[:3], bottle_pose[:3]))
            axial = sum(a*b for a,b in zip(delta, axis))
            max_thread_lateral = max(max_thread_lateral,
                math.sqrt(sum((d-axial*a)**2 for d,a in zip(delta, axis))))
            separation = min((p["separation_m"] for p in points), default=None)
            if step == 0:
                first_sep = separation
            if separation is not None:
                minimum = separation if minimum is None else min(minimum, separation)
            if points:
                peak = max(peak, *(p["normal_force_n"] for p in points))
            if step % max(1, round(.05/f.dt)) == 0 or step == steps-1:
                trace.append(self._snapshot((step+1)*f.dt, phase))
        end = f.bottle.data.root_pose_w.torch[0].detach().cpu().tolist()
        displacement = math.dist(start[:3], end[:3])
        qdot = min(1.0, abs(sum(a*b for a,b in zip(start[3:7], end[3:7]))))
        rotation = 2*math.acos(qdot)
        summary = f.monitor.summary(0)
        left = summary.get("left_bottle", {})
        right = summary.get("right_cap", {})
        opposing_ratio = opposing_steps/steps
        lt = _hand_digit_times(left, "l_", f.dt)
        rt = _hand_digit_times(right, "r_", f.dt)
        passed = (displacement < .02 and rotation < math.radians(10)
                  and left.get("contact_ratio", 0) > .8
                  and right.get("contact_ratio", 0) > .8
                  and lt.get("thumb", 0) > .8*self.hold_s
                  and max((v for k,v in lt.items() if k != "thumb"), default=0)
                      > .8*self.hold_s
                  and rt.get("thumb", 0) > .8*self.hold_s
                  and max((v for k,v in rt.items() if k != "thumb"), default=0)
                      > .8*self.hold_s
                  and (not self.require_right_tripod
                       or _right_tripod_held(rt, self.hold_s))
                  and opposing_ratio > .8
                  and summary.get("bottle_ground", {}).get("contact_ratio", 0) == 0
                  and summary.get("cap_ground", {}).get("contact_ratio", 0) == 0
                  and minimum is not None and minimum >= -self.max_penetration_m
                  and peak < self.max_contact_force_n)
        return {"passed": bool(passed), "duration_s": steps*f.dt,
                "initial_min_separation_m": first_sep,
                "minimum_separation_m": minimum,
                "peak_contact_force_n": peak,
                "bottle_displacement_m": displacement,
                "bottle_rotation_rad": rotation,
                "left_contact_ratio": left.get("contact_ratio", 0),
                "right_contact_ratio": right.get("contact_ratio", 0),
                "left_finger_contact_time_s": lt,
                "right_finger_contact_time_s": rt,
                "right_tripod_held": _right_tripod_held(rt, self.hold_s),
                "right_radial_opposition_ratio": opposing_ratio,
                "right_min_radial_normal_dot": min(normal_dots) if normal_dots else None,
                "max_thread_lateral_error_m": max_thread_lateral,
                "cross_contact_ratios": {name: summary.get(name, {}).get("contact_ratio", 0)
                                         for name in ("left_cap", "right_bottle", "hand_hand")},
                "contact_summary": summary, "trace": trace}

    def replay(self, sample):
        self.fixture.reset_manager.reset([0], [sample])
        return self._measure_hold("reset_replay")

    def derive_temporal_snapshot(self, sample, *, parent_result_sha256,
                                 offset_s):
        """Capture and validate a later *physical* state of one grasp family.

        These states share the same grasp geometry and seed; their provenance
        explicitly records the parent and elapsed unsupported time.
        """
        if not 0 < offset_s <= .5:
            raise ValueError("Temporal snapshot offset must be in (0, 0.5] s")
        f = self.fixture
        f.reset_manager.reset([0], [sample])
        f.monitor.reset([0])
        minimum = None
        peak = 0.0
        for _ in range(math.ceil(offset_s/f.dt)):
            f.step()
            points = self._points()
            if points:
                minimum = min(minimum if minimum is not None else math.inf,
                              *(p["separation_m"] for p in points))
                peak = max(peak, *(p["normal_force_n"] for p in points))
            if (minimum is not None and minimum < -self.max_penetration_m
                    or peak >= self.max_contact_force_n):
                return {"passed": False, "aborted": "snapshot prefix failed physical guards",
                        "formation_minimum_separation_m": minimum,
                        "formation_peak_contact_force_n": peak,
                        "candidate": None, "unsupported_hold": None,
                        "reset_replay": None}
        captured = self.left_capture._capture(
            sample, sample.left.joint_target,
            "unsupported_bimanual_temporal_snapshot", sample.right.joint_target,
            support_kind="none")
        captured = replace(captured, metadata={**captured.metadata,
            "temporal_snapshot": {"offset_s": offset_s,
                                   "parent_result_sha256": parent_result_sha256,
                                   "same_grasp_family": True}})
        prefix_summary = f.monitor.summary(0)
        hold = self._measure_hold("temporal_snapshot_hold")
        replay = self.replay(captured) if hold["passed"] else None
        return {"passed": bool(hold["passed"] and replay and replay["passed"]),
                "aborted": None,
                "formation_minimum_separation_m": minimum,
                "formation_peak_contact_force_n": peak,
                "formation_contact_summary": prefix_summary,
                "candidate": captured, "unsupported_hold": hold,
                "reset_replay": replay}

    def run(self, left_sample, right_proposal, *, close_s=1.2,
            settle_s=.3, approach_m=0.0, approach_s=.2,
            approach_down_m=0.0, follow_cap_during_formation=False,
            middle_mcp_delta_rad=0.0, index_pip_delta_rad=0.0,
            index_delay_fraction=0.0):
        f = self.fixture
        if (not left_sample.validated or
                left_sample.validation.get("scope") != "left_only" or
                left_sample.validation.get("wrist_control_mode") != "pose_command_wrist"):
            raise ValueError("Right formation requires a PhysX-validated pose-left sample")
        if not 0 <= approach_m <= .01 or not 0 <= approach_down_m <= .01:
            raise ValueError("Right wrist X/Z approach must be in [0, 10] mm")
        if not 0 <= index_delay_fraction < 1:
            raise ValueError("Index closure delay must be in [0, 1)")
        q = right_proposal["wrist_quaternion_wxyz"]
        pose = (*right_proposal["wrist_position_w_m"], *q[1:4], q[0])
        names = f.specs[1].joint_names
        pregrasp = tuple(right_proposal["pregrasp_joints_rad"][n] for n in names)
        closed = tuple(right_proposal["joints_rad"][n] +
                       (middle_mcp_delta_rad if n == "r_middle_finger_mcp_flex" else 0.0) +
                       (index_pip_delta_rad if n == "r_index_finger_pip" else 0.0)
                       for n in names)
        if (not -.1 <= middle_mcp_delta_rad <= .1
                or not -.1 <= index_pip_delta_rad <= .1) or any(
            not joint.lower <= target <= joint.upper
            for joint,target in zip(f.specs[1].joints, closed)
        ):
            raise ValueError("Finger adjustment exceeds the directed 0.1 rad or joint limit")
        right = HandInit(pose, (0.0,)*6, pregrasp, (0.0,)*20, pose, pregrasp)
        candidate = replace(left_sample, right=right, validated=False, validation=None,
                            source="pose_left_bank_plus_right_cap_synthesis",
                            metadata={**left_sample.metadata,
                                      "right_cap_proposal_index": right_proposal["index"],
                                      "right_wrist_dx_m": right_proposal.get("right_wrist_dx_m", 0.0),
                                      "right_wrist_dy_m": right_proposal.get("right_wrist_dy_m", 0.0),
                                      "right_cap_proposal_backend": "official_beta1_MJCF_geometry_only"})
        cap_reference = candidate.cap.root_pose[:3]
        print("BIMANUAL_STAGE reset_start", flush=True)
        f.reset_manager.reset([0], [candidate])
        print("BIMANUAL_STAGE reset_done", flush=True)
        f.step()
        print("BIMANUAL_STAGE first_step_done", flush=True)
        initial = self._snapshot(f.dt, "initial")
        print("BIMANUAL_STAGE first_snapshot_done", flush=True)
        trace = [initial]
        elapsed = f.dt
        minimum = initial["minimum_separation_m"]
        peak = max((p["normal_force_n"] for p in self._points()), default=0.0)
        abort = None
        follow_latched = False
        last_cap_delta = (0.0, 0.0, 0.0)
        if minimum is not None and minimum < -self.max_penetration_m:
            abort = f"initial penetration {minimum:.6f} m"
        for phase, seconds in (("close", close_s), ("approach", approach_s),
                               ("settle", settle_s)):
            print(f"BIMANUAL_STAGE {phase}_start", flush=True)
            if abort:
                break
            steps = max(1, math.ceil(seconds/f.dt))
            for step in range(steps):
                fraction = (step+1)/steps
                if phase == "close":
                    self._right_target(_staggered_close_targets(
                        names, pregrasp, closed, fraction, index_delay_fraction))
                if follow_cap_during_formation or (phase == "approach" and
                    (approach_m or approach_down_m)):
                    controller = f.controllers[1][0]
                    cap_now = f.cap.data.root_pose_w.torch[0, :3].detach().cpu().tolist()
                    # Follow lateral drift only. Chasing cap Z while a finger
                    # presses its top feeds the contact-induced downward
                    # motion back into the wrist command.
                    cap_delta = (cap_now[0]-cap_reference[0],
                                 cap_now[1]-cap_reference[1], 0.0)
                    if math.dist(cap_delta, (0.0,)*3) > .02:
                        abort = "cap moved beyond the 20 mm formation tracking bound"
                        break
                    if follow_cap_during_formation:
                        if not follow_latched:
                            last_cap_delta = cap_delta
                        else:
                            cap_delta = last_cap_delta
                    scale = (fraction if phase == "approach" else
                             1.0 if phase == "settle" else 0.0)
                    target = tuple(p + (d if follow_cap_during_formation else 0.0)
                                   - (scale*approach_m if i == 0 else
                                      scale*approach_down_m if i == 2 else 0.0)
                                   for i,(p,d) in enumerate(zip(pose[:3], cap_delta)))
                    controller.set_target(target, pose[3:7])
                f.step()
                elapsed += f.dt
                points = self._points()
                if follow_cap_during_formation and any(
                    p["category"] == "right_cap" and p["normal_force_n"] >= .05
                    for p in points
                ):
                    follow_latched = True
                sep = min((p["separation_m"] for p in points), default=None)
                if sep is not None:
                    minimum = sep if minimum is None else min(minimum, sep)
                peak = max(peak, *(p["normal_force_n"] for p in points)) if points else peak
                if step % max(1, round(.05/f.dt)) == 0 or step == steps-1:
                    trace.append(self._snapshot(elapsed, phase))
                if sep is not None and sep < -self.max_penetration_m:
                    abort = f"penetration {sep:.6f} m in {phase}"
                if peak >= self.max_contact_force_n:
                    abort = f"contact peak {peak:.3f} N in {phase}"
                if abort:
                    break
            print(f"BIMANUAL_STAGE {phase}_done", flush=True)
        release = None
        replay = None
        captured = None
        formation_summary = f.monitor.summary(0)
        if abort is None:
            captured = self.left_capture._capture(
                candidate, left_sample.left.joint_target,
                "unsupported_bimanual_right_cap_release_start", closed,
                support_kind="none")
            captured = replace(captured, metadata={**captured.metadata,
                               "right_cap_synthesis": {
                                   "proposal_index": right_proposal["index"],
                                   "approach_m": approach_m,
                                   "approach_down_m": approach_down_m,
                                   "middle_mcp_delta_rad": middle_mcp_delta_rad,
                                   "index_pip_delta_rad": index_pip_delta_rad,
                                   "index_delay_fraction": index_delay_fraction,
                                   "close_s": close_s,
                                   "settle_s": settle_s,
                                   "follow_cap_during_formation": follow_cap_during_formation,
                                   "follow_cap_latched_on_contact": follow_latched,
                                   "require_right_tripod": self.require_right_tripod,
                                   "support_used": False}})
            release = self._measure_hold("unsupported_hold")
            if release["passed"]:
                replay = self.replay(captured)
        return {"seed": left_sample.seed,
                "right_proposal_index": right_proposal["index"],
                "aborted": abort,
                "formation_minimum_separation_m": minimum,
                "formation_peak_contact_force_n": peak,
                "formation_contact_summary": formation_summary,
                "formation_trace": trace,
                "unsupported_hold": release,
                "reset_replay": replay,
                "candidate": captured,
                "passed": bool(release and release["passed"] and replay and replay["passed"])}

    def join_index(self, existing_sample, right_proposal, *, close_s=.6):
        """Add the index after resetting an already validated thumb-middle hold."""
        return self.join_digit(existing_sample, right_proposal,
                               digit="index", close_s=close_s)

    def join_digit(self, existing_sample, right_proposal, *, digit, close_s=.6):
        """Add one missing long finger to a validated, unsupported two-digit hold.

        This is an unsupported PhysX trajectory. It never alters bottle/cap
        pose outside Reset and never treats the existing two-digit bank as a
        validated tripod bank.
        """
        if (not existing_sample.validated or
                existing_sample.validation.get("scope") != "bimanual" or
                not self.require_right_tripod):
            raise ValueError("Digit joining needs a validated bimanual source and tripod guard")
        if digit not in ("index", "middle"):
            raise ValueError("Tripod join digit must be index or middle")
        if not .1 <= close_s <= 1.5:
            raise ValueError("Digit joining duration must be in [0.1, 1.5] s")
        f = self.fixture
        names = f.specs[1].joint_names
        initial = existing_sample.right.joint_target
        goal = right_proposal["joints_rad"]
        if any(not joint.lower <= goal[name] <= joint.upper
               for name, joint in zip(names, f.specs[1].joints)
               if name.startswith(f"r_{digit}_finger_")):
            raise ValueError("Digit join target exceeds official joint limits")
        f.reset_manager.reset([0], [existing_sample])
        f.step()
        trace = [self._snapshot(f.dt, f"{digit}_join_start")]
        points = self._points()
        minimum = min((p["separation_m"] for p in points), default=None)
        peak = max((p["normal_force_n"] for p in points), default=0.0)
        abort = None
        if minimum is not None and minimum < -self.max_penetration_m:
            abort = f"initial penetration {minimum:.6f} m"
        steps = math.ceil(close_s / f.dt)
        for step in range(steps):
            if abort:
                break
            fraction = (step + 1) / steps
            self._right_target(_digit_join_targets(
                names, initial, goal, fraction, digit))
            f.step()
            points = self._points()
            sep = min((p["separation_m"] for p in points), default=None)
            if sep is not None:
                minimum = sep if minimum is None else min(minimum, sep)
            peak = max(peak, *(p["normal_force_n"] for p in points)) if points else peak
            if step % max(1, round(.05 / f.dt)) == 0 or step == steps - 1:
                trace.append(self._snapshot((step + 2) * f.dt, f"{digit}_join"))
            if sep is not None and sep < -self.max_penetration_m:
                abort = f"penetration {sep:.6f} m during {digit} join"
            if peak >= self.max_contact_force_n:
                abort = f"contact peak {peak:.3f} N during {digit} join"
        target = _digit_join_targets(names, initial, goal, 1.0, digit)
        formation_summary = f.monitor.summary(0)
        captured = None
        hold = None
        replay = None
        if abort is None:
            captured = self.left_capture._capture(
                existing_sample, existing_sample.left.joint_target,
                f"unsupported_bimanual_{digit}_join", target, support_kind="none")
            captured = replace(captured, validated=False, validation=None,
                metadata={**captured.metadata,
                          "right_cap_proposal_index": right_proposal["index"],
                          "right_cap_synthesis": {
                              "joined_digit": digit,
                              "digit_join_duration_s": close_s,
                              "require_right_tripod": True,
                              "support_used": False}})
            hold = self._measure_hold("index_join_hold")
            if hold["passed"]:
                replay = self.replay(captured)
        return {"seed": existing_sample.seed,
                "right_proposal_index": right_proposal["index"],
                "aborted": abort,
                "formation_minimum_separation_m": minimum,
                "formation_peak_contact_force_n": peak,
                "formation_trace": trace,
                "formation_contact_summary": formation_summary,
                "unsupported_hold": hold, "reset_replay": replay,
                "candidate": captured,
                "passed": bool(hold and hold["passed"] and replay and replay["passed"])}

    def join_left_index(self, existing_sample, *, joint_name, delta_rad, close_s=.6):
        """Load a second left long finger while preserving a verified right tripod."""
        if (not existing_sample.validated or
                existing_sample.validation.get("scope") != "bimanual" or
                existing_sample.validation.get("right_grasp_topology") !=
                "thumb_index_middle_tripod" or not self.require_right_tripod):
            raise ValueError("Left index join requires a validated right tripod source")
        if joint_name not in ("l_index_finger_mcp_abd",
                              "l_index_finger_mcp_flex",
                              "l_index_finger_pip"):
            raise ValueError("Unsupported left index join joint")
        if not 0 < abs(delta_rad) <= .1 or not .1 <= close_s <= 1.5:
            raise ValueError("Left index joint change or duration exceeds bounds")
        f = self.fixture
        names = f.specs[0].joint_names
        index = names.index(joint_name)
        initial = existing_sample.left.joint_target
        goal = list(initial)
        goal[index] += delta_rad
        joint = f.specs[0].joints[index]
        if not joint.lower <= goal[index] <= joint.upper:
            raise ValueError("Left index target exceeds official limit")
        f.reset_manager.reset([0], [existing_sample])
        f.step()
        trace = [self._snapshot(f.dt, "left_index_join_start")]
        points = self._points()
        minimum = min((p["separation_m"] for p in points), default=None)
        peak = max((p["normal_force_n"] for p in points), default=0.0)
        abort = None
        if minimum is not None and minimum < -self.max_penetration_m:
            abort = f"initial penetration {minimum:.6f} m"
        steps = math.ceil(close_s/f.dt)
        for step in range(steps):
            if abort:
                break
            fraction = (step+1)/steps
            target = tuple(a + fraction*(b-a) for a,b in zip(initial, goal))
            self._left_target(target)
            f.step()
            points = self._points()
            sep = min((p["separation_m"] for p in points), default=None)
            if sep is not None:
                minimum = sep if minimum is None else min(minimum, sep)
            peak = max(peak, *(p["normal_force_n"] for p in points)) if points else peak
            if step % max(1, round(.05/f.dt)) == 0 or step == steps-1:
                trace.append(self._snapshot((step+2)*f.dt, "left_index_join"))
            if sep is not None and sep < -self.max_penetration_m:
                abort = f"penetration {sep:.6f} m during left index join"
            if peak >= self.max_contact_force_n:
                abort = f"contact peak {peak:.3f} N during left index join"
        formation_summary = f.monitor.summary(0)
        captured = hold = replay = None
        if abort is None:
            captured = self.left_capture._capture(
                existing_sample, tuple(goal), "unsupported_bimanual_left_index_join",
                existing_sample.right.joint_target, support_kind="none")
            captured = replace(captured, validated=False, validation=None,
                metadata={**captured.metadata,
                          "left_grasp_synthesis": {
                              "joined_digit": "index",
                              "joint_name": joint_name,
                              "delta_rad": delta_rad,
                              "join_duration_s": close_s,
                              "require_index_middle": True,
                              "support_used": False}})
            hold = self._measure_hold("left_index_join_hold")
            hold["left_index_middle_held"] = all(
                hold["left_finger_contact_time_s"].get(digit, 0.0) >
                .8*self.hold_s for digit in ("index", "middle"))
            if hold["passed"] and hold["left_index_middle_held"]:
                replay = self.replay(captured)
                replay["left_index_middle_held"] = all(
                    replay["left_finger_contact_time_s"].get(digit, 0.0) >
                    .8*self.hold_s for digit in ("index", "middle"))
        return {"seed": existing_sample.seed,
                "aborted": abort,
                "formation_minimum_separation_m": minimum,
                "formation_peak_contact_force_n": peak,
                "formation_trace": trace,
                "formation_contact_summary": formation_summary,
                "unsupported_hold": hold, "reset_replay": replay,
                "candidate": captured,
                "passed": bool(hold and hold["passed"] and
                               hold["left_index_middle_held"] and replay and
                               replay["passed"] and replay["left_index_middle_held"])}
