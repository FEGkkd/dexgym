"""The same frame, action decoder and fingertip objective serve real and imagined transitions."""
import math

import torch
from torch import nn
from .hand_shape import adjacent_finger_closure, shape_cost



DT = .02
HISTORY = 5
DEFAULT_ARM_RESIDUAL_DEG = 6.
HAND_STEP_RAD = .1
HAND_DEADZONE = .1
ACTION_MODE = "arm_ref_residual6_hand_command_delta_v1"
REWARD_VERSION = "s02_handdelta_fixed_wrist_shape_v3"
SHAPE_WEIGHT = .60


def hand_increment(action):
    bounded = action.clamp(-1., 1.)
    return HAND_STEP_RAD * bounded.sign() * (bounded.abs() - HAND_DEADZONE).relu() / (1. - HAND_DEADZONE)


def action_contract():
    return dict(action_mode=ACTION_MODE, hand_step_rad=HAND_STEP_RAD, hand_deadzone=HAND_DEADZONE)


def check_action_contract(archive):
    expected = dict(**action_contract(), residual_bound_deg=DEFAULT_ARM_RESIDUAL_DEG)
    if any(archive.get(k) != v for k, v in expected.items()):
        raise ValueError("Incompatible S0-2 arm/hand action contract")

# Original action scale, also the physical unit of the residual penalty.
BASE_ARM_RESIDUAL_DEG = 3.
# q, qd, previous command, target wrist xyz/rotation6, local target landmarks,
# reference arm, reference wrist position/rotation error, world/local tip errors,
# elbow swivel, transported anchor offset, anchor valid.
TIP_FRAME = 54 * 3 + 18 + 30 + 14 + 18 + 30 * 2 + 6 + 6 + 2
FRAME = TIP_FRAME - 30 + 2 * 5 * 4 * 3
TIP_VERSION = "s02_causal_fingertips_residual_v1"
VERSION = "s02_causal_four_point_hand_residual_v2"
MOTION_VERSION = "s02_causal_four_point_hand_residual_motion_v3"
PRECISION_VERSION = "s02_causal_four_point_hand_residual_precision_v4"
MOTION_WEIGHT = .025
WRIST_TAIL_WEIGHT = .12
ARM_LIMIT_SOFT_RAD = math.radians(3.)


def frame_features(version):
    if version in (VERSION, MOTION_VERSION, PRECISION_VERSION):
        return FRAME
    if version == TIP_VERSION:
        return TIP_FRAME
    raise ValueError(f"Unknown S0-2 observation version: {version}")


def local(pose):
    return pose.hand[..., -1, :]


def world(pose):
    return pose.wrist.unsqueeze(-2) + local(pose) @ pose.rotation.transpose(-1, -2)


def rotation6(rotation):
    return rotation[..., :2].transpose(-1, -2).flatten(-2)


def rotation_from_columns(columns):
    a, b = columns.reshape(-1, 2, 2, 3).unbind(-2)
    return torch.stack((a, b, torch.linalg.cross(a, b, dim=-1)), -1)


def observation_wrist(frame):
    return frame[:, 162:168].reshape(-1, 2, 3), rotation_from_columns(frame[:, 168:180])


def quiet_residual_weight(current, previous):
    """Per-arm smoothing weight from the four-point wrist-feedback observation."""
    wrist, rotation = observation_wrist(current)
    old_wrist, old_rotation = observation_wrist(previous)
    tips = current[:, 180:300].reshape(-1, 2, 5, 4, 3)[..., -1, :] / 10.
    old_tips = previous[:, 180:300].reshape(-1, 2, 5, 4, 3)[..., -1, :] / 10.
    position_delta = current[:, 314:320].reshape(-1, 2, 3) / 5.
    reference_to_goal = rotation_from_columns(current[:, 320:332])
    tip_offset = tips @ rotation.transpose(-1, -2)
    reference_offset = (tips @ reference_to_goal) @ rotation.transpose(-1, -2)
    reference_error = (position_delta.unsqueeze(-2) + tip_offset - reference_offset).norm(dim=-1).amax(-1)
    measured_wrist_error = current[:, 404:406] / 20.
    target_step = ((wrist - old_wrist).unsqueeze(-2) + tip_offset -
                   old_tips @ old_rotation.transpose(-1, -2)).norm(dim=-1).amax(-1)

    def below(value, low, high):
        alpha = ((high - value) / (high - low)).clamp(0., 1.)
        return alpha.square() * (3. - 2. * alpha)

    return (below(reference_error, .004, .018) * below(measured_wrist_error, .010, .026)
            * below(target_step, .001, .003))


def joint_clearance(position, lower, upper):
    return torch.minimum(position - lower, upper - position)


def quiet_arm_speed_weight(frame, arm_index):
    """Keep residual response available until the observed arm slows down."""
    speed = frame[:, 54:108][:, arm_index].reshape(-1, 2, 7).abs().amax(-1) / .1
    gate = ((.28 - speed) / .20).clamp(0., 1.)
    return gate.square() * (3. - 2. * gate)


def quiet_goal(goal, previous):
    delta = (world(goal) - world(previous)).norm(dim=-1).amax(-1)
    turn = (goal.rotation - previous.rotation).square().sum((-2, -1))
    return (delta < .001) & (turn < .00015)


def anchor_offset(geometry, anchor):
    ray, swivel, valid = anchor.split((3, 3, 1), -1)
    current = geometry.ray
    cosine = (ray * current).sum(-1, keepdim=True)
    projected = swivel - (swivel * current).sum(-1, keepdim=True) / (1. + cosine).clamp_min(.02) * (ray + current)
    ok = (valid > .5) & (cosine > .5) & (geometry.swivel.norm(dim=-1, keepdim=True) > .02)
    return (geometry.swivel - projected) * ok, ok.to(current.dtype)


def next_anchor(geometry, goal, previous, anchor):
    # Capture on convergence, retain until the operator moves the goal.
    still = quiet_goal(goal, previous)
    close = (world(geometry.pose) - world(goal)).norm(dim=-1).amax(-1) < .025
    ok = (geometry.swivel.norm(dim=-1) > .02) & close
    captured = torch.cat((geometry.ray, geometry.swivel, ok.unsqueeze(-1)), -1)
    retained = torch.where(anchor[..., 6:] > .5, anchor, captured)
    return torch.where(still.unsqueeze(-1), retained, torch.zeros_like(anchor)).detach()


class Interface(nn.Module):
    def __init__(self, fk, command, arm_margin_deg=5., *, version=VERSION, residual_bound_deg=None,
                 wrist_feedback_obs=False, arm_limit_guard_deg=0.):
        super().__init__()
        self.fk, self.command = fk, command
        self.frame_features = frame_features(version)
        self.features, self.history_frames = self.frame_features * HISTORY, HISTORY
        self.current_only, self.version = True, version
        self.four_point_hand = version != TIP_VERSION
        self.uses_elbow_anchor = version not in (MOTION_VERSION, PRECISION_VERSION)
        self.separate_tip_scores = version == PRECISION_VERSION
        self.wrist_feedback_obs = wrist_feedback_obs
        if wrist_feedback_obs and self.uses_elbow_anchor:
            raise ValueError("Wrist feedback requires the observation layout without an elbow target")
        self.register_buffer("arm_idx", command.arm.nonzero().flatten())
        if residual_bound_deg is None:
            residual_bound_deg = DEFAULT_ARM_RESIDUAL_DEG
        if residual_bound_deg != DEFAULT_ARM_RESIDUAL_DEG:
            raise ValueError("HandDelta-v2 requires a six-degree arm residual")
        self.residual_bound_deg = float(residual_bound_deg)
        self.register_buffer("residual_limit", torch.full_like(command.home[self.arm_idx], math.radians(residual_bound_deg)))
        if not math.isfinite(arm_margin_deg) or arm_margin_deg < 0:
            raise ValueError("Arm joint margin must be finite and nonnegative")
        self.arm_margin_deg = float(arm_margin_deg)
        self.register_buffer("arm_low", command.lower[self.arm_idx] + math.radians(arm_margin_deg))
        self.register_buffer("arm_high", command.upper[self.arm_idx] - math.radians(arm_margin_deg))
        if not math.isfinite(arm_limit_guard_deg) or arm_limit_guard_deg < 0.:
            raise ValueError("Residual limit guard must be finite and nonnegative")
        self.arm_limit_guard_rad = math.radians(arm_limit_guard_deg)
        if (self.arm_low + 2. * self.arm_limit_guard_rad >= self.arm_high).any():
            raise ValueError("Arm margin consumes the usable range of a joint")

    def desired(self, action, ref, previous_command):
        hand_request = previous_command + hand_increment(action)
        arms = ref + self.residual_limit * action[:, self.arm_idx].clamp(-1., 1.)
        if self.arm_limit_guard_rad:
            # Leave at least this much room when the reference already does. A reference
            # inside the guard zone can still move inward under the policy's residual.
            lower = torch.minimum(ref, self.arm_low + self.arm_limit_guard_rad)
            upper = torch.maximum(ref, self.arm_high - self.arm_limit_guard_rad)
            arms = arms.clamp(lower, upper)
        arms = arms.clamp(self.arm_low, self.arm_high)
        return hand_request.index_copy(1, self.arm_idx, arms)

    def normalized(self, action, ref, previous_command):
        return (self.desired(action, ref, previous_command) - self.command.home) / self.command.scale

    def frame(self, state, previous_command, goal, ref, anchor=None):
        q, qd = state.split(54, -1)
        batch = q.shape[0]
        q_at_ref = q.index_copy(1, self.arm_idx, ref)
        both = self.fk(torch.cat((q, q_at_ref)))
        # Extract only the physical wrist transforms of the reference configuration.
        pose = type(both.pose)(*(v[:batch] for v in both.pose))
        reference_rotation = both.pose.rotation[batch:].transpose(-1, -2) @ goal.rotation
        hand_target = goal.hand if self.four_point_hand else local(goal)
        fields = (q, qd * .1, previous_command, goal.wrist, rotation6(goal.rotation),
                  hand_target * 10., ref, (goal.wrist - both.pose.wrist[batch:]) * 5.,
                  rotation6(reference_rotation), (world(goal) - world(pose)) * 10.,
                  (local(goal) - local(pose)) * 20.)
        if self.uses_elbow_anchor:
            geometry = type(both)(pose, *(v[:batch] for v in both[1:]))
            offset, valid = anchor_offset(geometry, anchor)
            fields += (geometry.swivel * 5., offset * 10., valid)
        elif self.wrist_feedback_obs:
            local_rotation = pose.rotation.transpose(-1, -2) @ goal.rotation
            skew = local_rotation - local_rotation.transpose(-1, -2)
            angle_error = .5 * torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), -1)
            position_error = (goal.wrist - pose.wrist).unsqueeze(-2) @ pose.rotation
            ideal_tip = world(pose._replace(hand=goal.hand))
            arm_error = (world(goal) - ideal_tip).norm(dim=-1).amax(-1)
            fields += (position_error * 20., angle_error * 2., arm_error * 20.)
        else:
            # Keep the actor history layout for checkpoint continuation.
            fields += (q.new_zeros(batch, 14),)
        return torch.cat([v.reshape(batch, -1) for v in fields], -1)


class Actor(nn.Module):
    def __init__(self, command, *, version=VERSION, hand_prior=None, hand_correction_deg=None,
                 arm_feedback_gain=0., hand_lead_steps=0., arm_hand_compensation=0.,
                 arm_lead_steps=0., hand_actuation_gain=0., arm_state_lead_steps=0.,
                 arm_feedback_damping=.0025, arm_command_blend=0., quiet_residual_blend=0.,
                 quiet_residual_speed_gate=False, arm_margin_deg=5.):
        super().__init__()
        self.frame_features = frame_features(version)
        self.hand_prior = hand_prior
        self.hand_correction_deg = hand_correction_deg
        self.arm_feedback_gain = arm_feedback_gain
        self.hand_lead_steps = hand_lead_steps
        self.arm_hand_compensation = arm_hand_compensation
        self.arm_lead_steps = arm_lead_steps
        self.hand_actuation_gain = hand_actuation_gain
        self.arm_state_lead_steps = arm_state_lead_steps
        self.arm_feedback_damping = arm_feedback_damping
        self.arm_command_blend = arm_command_blend
        self.quiet_residual_blend = quiet_residual_blend
        self.quiet_residual_speed_gate = quiet_residual_speed_gate
        if (hand_prior is not None or hand_correction_deg is not None or arm_feedback_gain or hand_lead_steps
                or arm_hand_compensation or arm_lead_steps or hand_actuation_gain or arm_state_lead_steps):
            raise ValueError("Incremental hands and learned arm residuals require a network-only action")
        index = command.arm.nonzero().flatten()
        self.register_buffer("arm_indices", index, persistent=False)
        self.register_buffer("arm_low", command.lower[index] + math.radians(arm_margin_deg), persistent=False)
        self.register_buffer("arm_high", command.upper[index] - math.radians(arm_margin_deg), persistent=False)
        self.arm_scale = math.radians(DEFAULT_ARM_RESIDUAL_DEG)
        self.register_buffer("home", command.home.clone())
        self.register_buffer("scale", command.scale.clone())
        self.register_buffer("hand", ~command.arm.clone())
        self.net = nn.Sequential(nn.Linear(self.frame_features * HISTORY, 512), nn.LayerNorm(512), nn.SiLU(),
                                 nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 54))
        nn.init.normal_(self.net[-1].weight, std=.0005)
        # The exact deadzone has zero derivative at rest. Initialize hand rows at a
        # scale that exposes both the hold and moving regions to the actor loss.
        with torch.no_grad():
            weight = self.net[-1].weight
            indices = self.hand.to(weight.device)
            weight[indices] = torch.randn_like(weight[indices]) * .018
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, history):
        current = history[..., -self.frame_features:]
        measured = current[..., :54]
        command = current[:, 108:162]
        action = self.net(history).tanh()
        if self.arm_command_blend:
            index = self.arm_indices
            reference = current[:, 300:314]
            previous = (command[:, index] - reference) / self.arm_scale
            blended = torch.lerp(action[:, index], previous, self.arm_command_blend).clamp(-1., 1.)
            action = action.index_copy(1, index, blended)
        if self.quiet_residual_blend:
            index = self.arm_indices
            previous_frame = history[..., -2 * self.frame_features:-self.frame_features]
            previous_reference = previous_frame[:, 300:314]
            previous_residual = (command[:, index] - previous_reference) / self.arm_scale
            lower, upper = self.arm_low, self.arm_high
            clearance = torch.minimum(joint_clearance(measured[:, index], lower, upper),
                                      joint_clearance(current[:, 300:314], lower, upper))
            limit_release = ((clearance - math.radians(1.)) / math.radians(2.)).clamp(0., 1.)
            weight = self.quiet_residual_blend * quiet_residual_weight(current, previous_frame)
            if self.quiet_residual_speed_gate:
                weight = weight * quiet_arm_speed_weight(current, index)
            weight = weight.repeat_interleave(7, dim=-1) * limit_release
            # Keep the reference free to move. Smooth the offset relative to each tick's reference.
            filtered = torch.lerp(action[:, index], previous_residual.clamp(-1., 1.), weight)
            action = action.index_copy(1, index, filtered)
        return action


def robust(value):
    return torch.sqrt(1. + value.square()) - 1.


def precision_reward(world_cost, wrist_cost, motion_cost):
    world_score = .60 / (1. + world_cost / .60)
    wrist_score = .40 / (1. + wrist_cost / .40)
    return 1.4 * (world_score + wrist_score) / (1. + motion_cost)


def dual_tip_score(local_error, world_error):
    """Balanced per-hand tracking of physical tips in both specified coordinate systems."""
    def score(error,radius):
        closeness=(1.+(error/radius).square()).rsqrt()
        return .5*closeness.mean(-1)+.5*closeness.topk(2,dim=-1,largest=False).values.mean(-1)
    local,placed=score(local_error,.004), score(world_error,.007)
    both=2.*local*placed/(local+placed).clamp_min(1.e-6)
    return .75*both.mean(-1)+.25*both.amin(-1)


def uncorrective_motion(displacement, previous_error):
    """Remove only a bounded approach toward an outstanding target from the jitter cost."""
    error=previous_error.detach()
    distance=error.norm(dim=-1,keepdim=True)
    direction=error/distance.clamp_min(1.e-6)
    approach=(displacement*direction).sum(-1,keepdim=True).clamp_min(0.)
    budget=(.25*(distance-.0015).clamp_min(0.)).clamp_max(.0035)
    return displacement-direction*torch.minimum(approach,budget)


def tip_pairs(points):
    cross = (points[:, 1, None] - points[:, 0, :, None]).flatten(1, 2)
    thumb = (points[:, :, :1] - points[:, :, 1:]).flatten(1, 2)
    return torch.cat((cross, thumb), dim=1)


def paired_tip_errors(tips, target):
    """Absolute endpoint error, world-vector error, and desired pair spacing."""
    endpoint = (tips - target).norm(dim=-1)
    cross = (endpoint[:, 0, :, None] + endpoint[:, 1, None, :]).flatten(1, 2)
    thumb = (endpoint[:, :, :1] + endpoint[:, :, 1:]).flatten(1, 2)
    desired_pair = tip_pairs(target)
    absolute = .5 * torch.cat((cross, thumb), dim=1)
    relative = (tip_pairs(tips) - desired_pair).norm(dim=-1)
    return absolute, relative, desired_pair.square().sum(-1)


def pair_average(value, weight):
    return (value * weight).sum(-1) / weight.sum(-1).clamp_min(1.)


def weak_wrist_cost(pose, goal):
    position = (pose.wrist - goal.wrist).norm(dim=-1)
    orientation = (pose.rotation - goal.rotation).norm(dim=(-2, -1))
    return (.008 * robust((position - .020).relu() / .040).mean(-1)
            + .006 * robust((orientation - .24).relu() / .45).mean(-1))


def worst_hand_tip_error_sq(pose, goal):
    return (world(pose) - world(goal)).square().sum(-1).mean(-1).amax(-1)


def fingertip_placement(tip, target, scale):
    """Score physical endpoints and signed inter-tip vectors in the same world frame."""
    error = (tip - target).norm(dim=-1)
    absolute, vector, gap_sq = paired_tip_errors(tip, target)
    proximity = torch.exp(-gap_sq / .035**2)
    center = (tip - target).mean(dim=2).norm(dim=-1)
    corresponding = tip[:, 1] - tip[:, 0]
    desired = target[:, 1] - target[:, 0]
    near = torch.exp(-desired.square().sum(-1) / .035**2)
    relative = (corresponding - desired).norm(dim=-1)
    close_tail = (near * robust(relative / .012)).amax(-1)
    cost = (.38 * robust(error / scale).mean((1, 2))
            + .16 * robust(error.topk(2, dim=-1).values / scale).mean((1, 2))
            + .09 * pair_average(robust(absolute / scale), proximity)
            + .18 * pair_average(robust(vector / .012), proximity)
            + .10 * robust(center.amax(-1) / scale) + .16 * close_tail)
    return cost, error, absolute, vector, proximity, close_tail


def objective(geom, old, goal, previous_goal, action, cmd, prev, prev2, anchor, actuator, *,
              four_point_hand=True, residual_bound_deg=None, separate_tip_scores=False,
              fine_tracking_reward=False, quiet_arm_curvature_weight=0., world_tip_scale_m=None, adjacent_shape_weight=0.,
              dual_tip_fraction=0., correction_aware_stillness=False):
    tip, target = world(geom.pose), world(goal)
    e_local = (local(geom.pose) - local(goal)).norm(dim=-1)
    world_scale = world_tip_scale_m if world_tip_scale_m is not None else (.012 if fine_tracking_reward else .018)
    world_cost, e_world, pair_absolute, vector_error, proximity, close_tail = fingertip_placement(tip, target, world_scale)
    shape_stats = {"placement_cost": world_cost.detach(), "near_corresponding_tail_cost": close_tail.detach()}
    if four_point_hand:
        hand_cost, shape_tip, direction = shape_cost(geom.pose.hand, goal.hand)
        wrist_cost = SHAPE_WEIGHT * hand_cost
        if adjacent_shape_weight:
            closure = adjacent_finger_closure(geom.pose.hand, goal.hand)
            wrist_cost = wrist_cost + adjacent_shape_weight * closure
            shape_stats["adjacent_finger_closure_cost"] = closure.detach()
        shape_stats["common_skeleton_cost"] = hand_cost.detach()
        shape_stats["shape_tip_normalized_mm"] = 1000. * shape_tip.mean((1, 2))
        shape_stats["shape_direction_chord"] = direction.mean((1, 2, 3))
        joints = (geom.pose.hand[..., 1:3, :] - goal.hand[..., 1:3, :]).norm(dim=-1)
        shape_stats["hand_joint_local_mm"] = joints.mean((1, 2, 3)) * 1000.
    else:
        wrist_scale = .007 if fine_tracking_reward else .009
        wrist_cost = .36 * robust(e_local / wrist_scale).mean((1, 2))
        if separate_tip_scores:
            wrist_cost = wrist_cost + WRIST_TAIL_WEIGHT * robust(e_local.flatten(1).topk(2).values / wrist_scale).mean(1)
    world_cost = world_cost + weak_wrist_cost(geom.pose, goal)
    reached = torch.exp(-e_world.detach().amax(-1).square() / .025**2)
    quiet = quiet_goal(goal, previous_goal)
    still = quiet * reached
    old_tip,old_target=world(old.pose),world(previous_goal)
    relative_step=(tip-old_tip)-(target-old_target)
    if correction_aware_stillness:
        relative_step=uncorrective_motion(relative_step,old_target-old_tip)
    jitter=relative_step.norm(dim=-1).mean(-1)
    motion_cost=.075*((quiet if correction_aware_stillness else still)*robust(jitter/.0015)).mean(-1)
    if four_point_hand:
        # Once settled, discourage elbow motion even if the wrist and fingertips hold still.
        elbow_step = (geom.upper_arm - old.upper_arm).norm(dim=-1)
        motion_cost = motion_cost + .04 * (still * robust(elbow_step / .002)).mean(-1)
        shape_stats["elbow_step_mm"] = 1000. * elbow_step.mean(-1)
    accel = (cmd - 2. * prev + prev2) / actuator.slew.clamp_min(.0001)
    curvature_weight = .045 if fine_tracking_reward else .035
    arm_curvature = robust(accel[:, actuator.arm]).reshape(-1, 2, 7).mean(-1)
    motion_cost = motion_cost + curvature_weight * (arm_curvature.mean(-1)
                                                   + robust(accel[:, ~actuator.arm]).mean(-1))
    if quiet_arm_curvature_weight:
        motion_cost = motion_cost + quiet_arm_curvature_weight * (quiet * arm_curvature).mean(-1)
    if residual_bound_deg is None:
        residual_bound_deg = DEFAULT_ARM_RESIDUAL_DEG if four_point_hand else BASE_ARM_RESIDUAL_DEG
    if action.shape[-1] == actuator.arm.numel():
        requested_deg = action[:, actuator.arm].clamp(-1., 1.) * residual_bound_deg
        shape_stats["arm_residual_deg"] = requested_deg.abs().mean(-1)
    elif anchor is not None:
        raise ValueError("The elbow-anchor reward requires arm joint residuals")
    if anchor is None:
        travel = (cmd - prev).abs()
        scaled_travel = travel / actuator.slew.clamp_min(.0001)
        # An absolute increment cost adds over time, including repeated slow motion.
        motion_cost = motion_cost + MOTION_WEIGHT * (scaled_travel[:, actuator.arm].mean(-1)
                                                      + scaled_travel[:, ~actuator.arm].mean(-1))
        shape_stats["arm_command_travel_rad"] = travel[:, actuator.arm].sum(-1)
        shape_stats["hand_command_travel_rad"] = travel[:, ~actuator.arm].sum(-1)
    else:
        offset, _ = anchor_offset(geom, anchor)
        motion_cost = motion_cost + .10 * robust((offset.norm(dim=-1) - .003).relu() / .012).mean(-1)
        motion_cost = motion_cost + .012 * (requested_deg / BASE_ARM_RESIDUAL_DEG).square().mean(-1)
        shape_stats["elbow_anchor_mm"] = offset.norm(dim=-1).amax(-1) * 1000.
    # Positive scores prevent early termination from being profitable.
    reward = (precision_reward(world_cost, wrist_cost, motion_cost) if separate_tip_scores and not four_point_hand
              else 1.4 / (1. + world_cost + wrist_cost + motion_cost))
    if dual_tip_fraction:
        precise=dual_tip_score(e_local,e_world)
        reward=torch.lerp(reward,1.4*precise/(1.+motion_cost),dual_tip_fraction)
        shape_stats["dual_frame_tip_score"]=precise.detach()
    squared_velocity_change = ((cmd - 2. * prev + prev2) / DT**2).square()
    support = proximity.sum(-1)
    return reward, {**shape_stats, "tip_world_mm": e_world.mean((1, 2)) * 1000.,
                    "tip_local_mm": e_local.mean((1, 2)) * 1000.,
                    "tip_worst_mm": e_world.flatten(1).amax(-1) * 1000.,
                    "close_pair_support": support,
                    "pair_error_mm_weighted": 1000. * pair_average(vector_error, proximity),
                    "pair_endpoint_error_mm_weighted": 1000. * pair_average(pair_absolute, proximity),
                    "arm_command_accel_rms_rad_s2": squared_velocity_change[:, actuator.arm].mean(-1).sqrt(),
                    "command_curvature": accel.abs().mean(-1)}
