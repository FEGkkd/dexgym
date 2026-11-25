"""Continuous reference replay with RSI, closed-loop rewards, and Newton events."""

from math import cos, exp

import torch
from torch.nn.functional import normalize, softplus

from ..hand2_cube_supermimic.rotations import mm
from ..marvin54_s0.physics import COLLISION, NONFINITE, OVERFLOW, SUBSTEPS, TRACKING
from ..wholebody_s0.domain import Domain, IMPULSE_PROBABILITY, apply_newton_domain
from ..wholebody_s0.robot import FIRMWARE_FINGERS
from . import DOF, DT
from .bank import bridge_pose
from .control import OBSERVATION_VERSION, History, Observation
from .cross_trajectory import ReferenceSwitch, SwitchConfig
from .finger_reward import (BILATERAL_NEAR_M, FingerRewardConfig, bilateral_opposition_gate, bilateral_tip_vectors,
                            opposing_fingertip_cost, thumb_ip_indices)
from .kinematics import Geometry, Pose
from .posture import PostureReference
from .prepare import HARD_DEPTH

HOME_FRACTION = .12
BRIDGE_STEPS = 60
SEGMENT_STEPS = 128
STATIONARY_STEPS = 48
SAMPLING_RAMP_UPDATES = 1000
AUDIT_DEPTH = .0005
POSITIVE_TRACKING_REWARD_VERSION = "positive_tracking_reward_v1"
WRIST_EXECUTION_REWARD_VERSION = "actual_wrist_tracking_reward_v1"
HAND_EXECUTION_REWARD_VERSION = "actual_hand_tracking_reward_v1"
WRIST_ROTATION_REWARD_VERSION = "actual_wrist_rotation_tail_v1"
FAST_HAND_REWARD_VERSION = "fast_finger_execution_reward_v1"
RECOVERY_DISTANCE_M = .22
RECOVERY_ROTATION_RAD = 1.05
RECOVERY_ROTATION_CHORD2 = 2. * (1. - cos(RECOVERY_ROTATION_RAD))

REWARD_SCALES = {"wrist_m": .045, "wrist_chord": .25, "finger_m": .026, "tip_m": .020,
                 "relative_m": .055, "relative_chord": .29}


def squared(value):
    return value.square().sum(-1)


def chord2(measured, desired):
    return .5 * (measured - desired).square().sum(dim=(-1, -2))


def degrees(error2):
    return (2 * (error2.clamp_min(0).sqrt() / 2).clamp(max=1).asin()).rad2deg()


def pose_errors(measured, goal):
    hand_mm = squared(measured.hand - goal.hand).sqrt() * 1000.
    return {
        "wrist_mm": squared(measured.wrist - goal.wrist).sqrt() * 1000.,
        "wrist_deg": degrees(chord2(measured.rotation, goal.rotation)),
        "hand_mm": hand_mm,
        "knuckle_mm": hand_mm[..., -3:-1],
        "tip_mm": hand_mm[..., -1],
        "relative_mm": squared(measured.relative_position - goal.relative_position).sqrt() * 1000.,
        "relative_deg": degrees(chord2(measured.relative_rotation, goal.relative_rotation)),
    }


def score(error2, scale):
    ratio = error2 / scale**2
    return (.65 / (1 + ratio) + .35 / (1 + ratio / 16)).reshape(error2.shape[0], -1).mean(-1)


def tracking_cost(measured, goal):
    s = REWARD_SCALES
    hand = squared(measured.hand - goal.hand)

    def cost(error2, key):
        return torch.log1p(error2 / s[key]**2).reshape(error2.shape[0], -1).mean(-1)

    return (
        2. * cost(squared(measured.wrist - goal.wrist), "wrist_m")
        + 1.5 * cost(chord2(measured.rotation, goal.rotation), "wrist_chord")
        + 1.5 * cost(squared(measured.relative_position - goal.relative_position), "relative_m")
        + cost(chord2(measured.relative_rotation, goal.relative_rotation), "relative_chord")
        + 2. * cost(hand[..., -3:-1], "finger_m")
        + 2. * cost(hand[..., -1], "tip_m")
    ) / 10.


def precision_cost(measured, goal, precise=False, right_elbow=False):
    hand = squared(measured.hand - goal.hand)
    tip = hand[..., -1]
    # A bounded fine-scale term keeps gradients near the precision range
    # while the broad tracking cost handles the recovery range.
    cost = (
        .052 * (1. - score(squared(measured.wrist - goal.wrist), .006))
        + .030 * (1. - score(chord2(measured.rotation, goal.rotation), .040))
        + .025 * (1. - score(hand[..., -3:-1], .006))
        + .050 * (1. - score(tip, .005))
        + .050 * (1. - score(squared(measured.relative_position - goal.relative_position), .010))
        + .025 * (1. - score(chord2(measured.relative_rotation, goal.relative_rotation), .060))
    )
    if precise:
        tail = torch.log1p(tip / .009**2).amax(dim=-1).mean(-1)
        cost = cost + .11 * (1. - score(tip, .007)) + .025 * tail
    if right_elbow:
        knuckle = hand[..., -3:-1]
        knuckle_tail = torch.log1p(knuckle / .012**2).amax(dim=(-1, -2)).mean(-1)
        tip_excess = (tip - .007**2).clamp_min(0.)
        cost = (cost + .105 * (1. - score(knuckle, .008)) + .045 * knuckle_tail
                + .055 * torch.log1p(tip_excess / .008**2).amax(dim=-1).mean(-1)
                + .045 * (1. - score(squared(measured.wrist - goal.wrist), .004))
                + .023 * (1. - score(chord2(measured.rotation, goal.rotation), .032))
                + .025 * (1. - score(squared(measured.relative_position - goal.relative_position), .007))
                + .015 * (1. - score(chord2(measured.relative_rotation, goal.relative_rotation), .049)))
    return cost


def fingertip_position_tail_cost(tip_error):
    distance = torch.linalg.vector_norm(tip_error, dim=-1)
    # Start tightening before the 10 mm evaluation target. Average each hand's
    # two worst errors as well as all ten tips so a single digit cannot be ignored.
    excess = (distance - .0065).clamp_min(0.)
    tail = torch.log1p((excess / .008).square())
    # Concentrate some of the gradient near the desired percentile threshold.
    # A bounded term keeps individual hard-to-reach goals from dominating it.
    floor = 1. / (1. + exp(.0035 / .002))
    over_threshold = torch.where(excess > 0., torch.sigmoid((excess - .0035) / .002) - floor, 0.)
    return (.15 * tail.mean((1, 2)) + .11 * tail.topk(k=2, dim=-1).values.mean((1, 2))
            + .14 * over_threshold.mean((1, 2)))


def fingertip_tail_cost(tip_error, tip_rate_error, position_scale=1.):
    velocity = torch.log1p(squared(tip_rate_error) / .095**2)
    return position_scale * fingertip_position_tail_cost(tip_error) + .026 * velocity.mean((1, 2))


def wrist_precision_tail_cost(measured, goal):
    # Continue improving the difficult wrist near the 10 mm range even when
    # bounded fine-scale scores have nearly saturated. Keep side/group means.
    wrist = (torch.linalg.vector_norm(measured.wrist - goal.wrist, dim=-1) - .004).clamp_min(0.)
    relative = (torch.linalg.vector_norm(measured.relative_position - goal.relative_position,
                                        dim=-1) - .008).clamp_min(0.)
    wrist = torch.sqrt(1. + (wrist / .012).square()) - 1.
    relative = torch.sqrt(1. + (relative / .020).square()) - 1.
    return .090 * wrist.mean(-1) + .035 * wrist.amax(-1) + .070 * relative


def wrist_rotation_tail_cost(measured, goal):
    def cost(actual, desired):
        chord = torch.linalg.vector_norm((actual - desired).flatten(-2), dim=-1) / 2.**.5
        excess = (chord - .04).clamp_min(0.)
        return torch.sqrt(1. + (excess / .10).square()) - 1.

    bilateral = cost(measured.rotation, goal.rotation).mean(-1)
    relative = cost(measured.relative_rotation, goal.relative_rotation)
    return .65 * bilateral + .35 * relative


def four_point_tracking_cost(measured, goal):
    cost = torch.log1p(squared(measured.hand - goal.hand) / .008**2)
    per_finger = cost.mean(-1)
    worst = per_finger.topk(2, dim=-1).values
    return .070 * per_finger.mean((1, 2)) + .040 * worst.mean((1, 2))


def slow_finger_motion_cost(rate_error, goal_rate):
    # The gate depends only on the external target, so the policy cannot evade
    # this penalty by moving farther away from the target.
    slow = torch.exp(-squared(goal_rate) / .020**2)
    cost = slow * torch.log1p(squared(rate_error) / .055**2)
    return .016 * cost.flatten(1).mean(-1)


def still_finger_gate(goal_point_rate):
    return torch.exp(-squared(goal_point_rate).amax(dim=-1) / .045**2)


def still_finger_command_cost(request_rate, goal_point_rate, *, pseudo_huber=False):
    magnitude2 = request_rate.square()
    cost = torch.sqrt(1. + magnitude2) - 1. if pseudo_huber else torch.log1p(magnitude2)
    return (still_finger_gate(goal_point_rate) * cost.mean(-1)).mean((1, 2))


def requested_landmark_cost(requested_hand, target_hand, goal_point_rate):
    point_cost = torch.sqrt(1. + squared(requested_hand - target_hand) / .010**2) - 1.
    by_finger = still_finger_gate(goal_point_rate) * point_cost.mean(-1)
    return .7 * by_finger.mean((1, 2)) + .3 * by_finger.topk(2, dim=-1).values.mean((1, 2))


def fast_hand_tracking_cost(measured, goal, goal_rate, delayed):
    speed = torch.linalg.vector_norm(goal_rate, dim=-1).amax(-1)
    moving = ((speed - .035) / .085).clamp(0., 1.)
    point = torch.sqrt(1. + squared(measured.hand - goal.hand) / .010**2) - 1.
    digit = moving * (.50 * point.mean(-1) + .50 * point[..., -1])
    value = .70 * digit.mean((1, 2)) + .30 * digit.topk(2, dim=-1).values.mean((1, 2))
    return value if delayed is None else value * (1. + .75 * delayed)


def proximal_hand_cost(measured, goal):
    current = normalize(measured.hand.diff(dim=-2), dim=-1, eps=1e-6)
    desired = normalize(goal.hand.diff(dim=-2), dim=-1, eps=1e-6)
    root = torch.log1p(squared(measured.hand[..., 0, :] - goal.hand[..., 0, :]) / .006**2)
    first_bone = torch.log1p(squared(current[..., 0, :] - desired[..., 0, :]) / .22**2)
    # Match the change of direction at the thumb MCP as well as its two positions.
    bend = (current[:, :, 0, 0] * current[:, :, 0, 1]).sum(-1)
    bend_target = (desired[:, :, 0, 0] * desired[:, :, 0, 1]).sum(-1)
    bend_cost = torch.log1p(((bend - bend_target) / .16).square())
    return (.025 * root.mean((1, 2)) + .045 * first_bone.mean((1, 2))
            + .055 * first_bone[..., 0].mean(-1) + .10 * bend_cost.mean(-1))


def wrist_tracking_failure(wrist_dist, wrist_rot):
    return ((wrist_dist > RECOVERY_DISTANCE_M**2) | (wrist_rot > RECOVERY_ROTATION_CHORD2)).any(-1)


def positive_step_score(score):
    # Keep a positive return for continuing to track, including while recovering.
    # Contact and terminal costs are applied to this score by the environment.
    return .05 + .46 * softplus(score / .48)


class DirectTask:
    thumb_ip_reference = None
    stability = False
    elbow_swivel_weight = 0.
    elbow_envelope_weight = 0.

    def __init__(self, model, metadata, count, device, seed, robust, *, finger_reward=None, proximal=False,
                 observation_version=OBSERVATION_VERSION, positive_tracking_reward=False,
                 wrist_execution_weight=0., hand_execution_weight=0., wrist_rotation_tail_weight=0.,
                 fast_hand_weight=0.):
        self.positive_tracking_reward = positive_tracking_reward
        self.wrist_execution_weight = wrist_execution_weight
        self.hand_execution_weight = hand_execution_weight
        self.wrist_rotation_tail_weight = wrist_rotation_tail_weight
        self.fast_hand_weight = fast_hand_weight
        self.finger_reward = finger_reward or FingerRewardConfig()
        if self.finger_reward.still_command_weight:
            finger_joints = [
                [[int(model.jnt_qposadr[int(model.actuator(f"{side}_{finger}{joint}").trnid[0])])
                  for joint in range(4)] for finger in FIRMWARE_FINGERS]
                for side in "lr"
            ]
            self.finger_joints = torch.tensor(finger_joints, device=device)
        self.observe = Observation(model, metadata, version=observation_version, proximal=proximal).to(device)
        self.command = self.observe.command
        self.domain = Domain(count, device, seed, robust, self.command.arm, include_nominal=True)
        self.transfer_delta = None
        self.simulator_context = self.domain.physical
        self.home_geometry = self.observe.fk(self.command.home[None])
        self.home_pose = self.home_geometry.pose
        self.home_swivel = normalize(self.home_geometry.swivel, dim=-1, eps=1e-6)
        self.margin = .05 * (self.command.upper - self.command.lower)
        self.acceleration_budget = torch.where(self.command.arm, 16., 36.)
        self.acceleration_weight = torch.where(self.command.arm, .013 / 14, .009 / 40)
        self.posture = None

    def thumb_ip_target(self, frame, bridge):
        fraction = (1. - bridge / BRIDGE_STEPS).clamp(0., 1.)
        weight = (fraction.square() * (3. - 2. * fraction))[:, None]
        return torch.lerp(self.command.home[list(self.thumb_ip_joints)], self.thumb_ip_reference[frame], weight)

    def servo_scales(self, domain):
        return torch.where(self.command.arm, domain[:, :1], domain[:, 1:2]), domain[:, 2:3], domain[:, 3:4]

    def position_violation(self, state):
        q = state[:, :DOF]
        return torch.maximum(self.command.lower - q, q - self.command.upper).clamp_min(0).amax(dim=-1)

    def final_tracking_score(self, value, measured, goal):
        if self.wrist_rotation_tail_weight:
            value = value - self.wrist_rotation_tail_weight * wrist_rotation_tail_cost(measured, goal)
        value = positive_step_score(value) if self.positive_tracking_reward else value
        reward = (1. - self.wrist_execution_weight - self.hand_execution_weight) * value
        if self.wrist_execution_weight:
            wrist = score(squared(measured.wrist - goal.wrist), .012)
            orientation = score(chord2(measured.rotation, goal.rotation), .075)
            relative = score(squared(measured.relative_position - goal.relative_position), .015)
            relative_orientation = score(chord2(measured.relative_rotation, goal.relative_rotation), .085)
            position_quality = .38 * wrist + .22 * relative
            rotation_quality = .22 * orientation + .18 * relative_orientation
            position_relaxation = self.finger_reward.opposition_wrist_position_relaxation
            rotation_relaxation = self.finger_reward.opposition_wrist_rotation_relaxation
            if position_relaxation or rotation_relaxation:
                gate = bilateral_opposition_gate(goal)
                position_quality = position_quality + gate * position_relaxation * (.60 - position_quality)
                rotation_quality = rotation_quality + gate * rotation_relaxation * (.40 - rotation_quality)
            quality = 1.25 * (position_quality + rotation_quality)
            reward = reward + self.wrist_execution_weight * quality
        if self.hand_execution_weight:
            hand = squared(measured.hand - goal.hand)
            shape = score(hand, .008)
            tip = score(hand[..., -1], .006)
            reward = reward + self.hand_execution_weight * 1.25 * (.45 * shape + .55 * tip)
        return reward

    def tracking_reward(self, state, geometry, previous, goal, old_goal, command, previous_command,
                        *, previous_state=None, phase=None, old_phase=None, target_jump=None, requested_action=None,
                        delayed=None):
        if target_jump is not None:
            # A reference discontinuity is not an achievable target velocity.
            old_goal = Pose(*(torch.where(target_jump.reshape(-1, *([1] * (a.ndim - 1))), b, a)
                              for a, b in zip(old_goal, goal, strict=True)))
        measured, swivel, ray, _ = geometry
        refining = self.posture is not None
        point_tracking = refining and self.posture.point_tracking
        finger_reward = self.finger_reward
        use_request = finger_reward.still_command_weight > 0. or finger_reward.still_landmark_weight > 0.
        use_point_rate = point_tracking or use_request or self.fast_hand_weight
        goal_point_rate = (goal.hand - old_goal.hand) / DT if use_point_rate else None
        prior_pose = previous.pose if refining else previous
        tracking = tracking_cost(measured, goal)
        progress = (tracking_cost(prior_pose, goal) - tracking).clamp(-.5, .5)
        wrist_rate_error = (measured.wrist - prior_pose.wrist - goal.wrist + old_goal.wrist) / DT
        tip_rate_error = (measured.hand[..., -1, :] - prior_pose.hand[..., -1, :]
                          - goal.hand[..., -1, :] + old_goal.hand[..., -1, :]) / DT
        actual_delta = mm(prior_pose.rotation.transpose(-1, -2), measured.rotation)
        goal_delta = mm(old_goal.rotation.transpose(-1, -2), goal.rotation)
        velocity = (.25 * score(squared(wrist_rate_error), .45) + .20 * score(squared(tip_rate_error), .32)
                    + .15 * score(chord2(actual_delta, goal_delta), DT * 3.))
        q = state[:, :DOF]
        distance = torch.minimum(q - self.command.lower, self.command.upper - q)
        limit = (1. - distance / self.margin).clamp(0., 2.).square().mean(-1)
        rate = (command - previous_command) / self.command.rate_scale
        smooth = .12 * rate.square().mean(-1)
        if use_request:
            requested = self.command.desired(requested_action)
        if finger_reward.still_command_weight:
            # Requests above the slew limit retain a gradient through this cost.
            request_rate = (requested - previous_command) / self.command.rate_scale
            smooth = smooth + finger_reward.still_command_weight * still_finger_command_cost(
                request_rate[:, self.finger_joints], goal_point_rate,
                pseudo_huber=finger_reward.still_command_loss == "pseudo_huber")
        reward = 1. - .38 * tracking.clamp(max=2.4) + .75 * progress + velocity / 10.6 - smooth
        if finger_reward.opposition_weight:
            reward = reward - finger_reward.opposition_weight * opposing_fingertip_cost(
                measured, prior_pose, goal, old_goal, DT)
        if finger_reward.still_landmark_weight:
            position = requested.clamp(self.command.lower, self.command.upper)
            requested_hand = self.observe.fk(position).pose.hand
            reward = reward - finger_reward.still_landmark_weight * requested_landmark_cost(
                requested_hand, goal.hand, goal_point_rate)
        if point_tracking:
            reward = reward - four_point_tracking_cost(measured, goal)
        elif goal.hand.shape[-2] == 4:
            reward = reward - proximal_hand_cost(measured, goal)
        if self.fast_hand_weight:
            reward = reward - self.fast_hand_weight * fast_hand_tracking_cost(measured, goal, goal_point_rate, delayed)
        tip_error = measured.hand[..., -1, :] - goal.hand[..., -1, :]
        has_tail_cost = refining and self.posture.motion is not None and self.posture.tip_precision
        if finger_reward.tracking_boost and not has_tail_cost:
            reward = reward - finger_reward.tracking_boost * fingertip_position_tail_cost(tip_error)
        if self.thumb_ip_reference is not None and not point_tracking:
            target_ip = self.thumb_ip_target(*phase)
            ip_error = measured.hand[:, :, 0, -2] - goal.hand[:, :, 0, -2]
            ip_position = torch.log1p(squared(ip_error) / .006**2)
            # Ramp in supervision as the reference moves into positive flexion.
            demand = (target_ip / .20).clamp(0., 1.)
            ip_angle = demand * torch.log1p(((q[:, list(self.thumb_ip_joints)] - target_ip) / .20).square())
            reward = (reward - .08 * ip_position.mean(-1) - .025 * ip_position.amax(-1)
                      - .09 * ip_angle.mean(-1) - .030 * ip_angle.amax(-1))
        if not refining:
            preferred = self.home_swivel - (self.home_swivel * ray).sum(-1, keepdim=True) * ray
            swivel_error = 1. - (normalize(swivel, dim=-1, eps=1e-5)
                                 * normalize(preferred, dim=-1, eps=1e-5)).sum(-1)
            tracking_value = reward - .035 * swivel_error.mean(-1) - .10 * limit
            return self.final_tracking_score(tracking_value, measured, goal)
        current = self.posture.features(q, geometry)
        old = self.posture.features(previous_state[:, :DOF], previous)
        target = self.posture.at(*phase, BRIDGE_STEPS)
        old_target = self.posture.at(*old_phase, BRIDGE_STEPS)
        if target_jump is not None:
            old_target = type(target)(*(torch.where(target_jump.reshape(-1, *([1] * (a.ndim - 1))), b, a)
                                       for a, b in zip(old_target, target, strict=True)))
        if getattr(self, "stability", False):
            # The target's motion, not the archived redundant IK branch, determines
            # whether an elbow should stay. Recovering wrists remain free to move.
            from .stability import stationary_wrist_gate
            gate = stationary_wrist_gate(goal, old_goal, previous.pose).detach()[..., None]
            target = target._replace(upper_arm=torch.lerp(target.upper_arm, old.upper_arm.detach(), gate))
            old_target = old_target._replace(upper_arm=torch.lerp(old_target.upper_arm, old.upper_arm.detach(), gate))
        if getattr(self, "stability_v2", False):
            from .stability import static_elbow_motion_cost
            if self.elbow_swivel_weight:
                from .stability import swivel_continuity_cost
                posture_cost = self.elbow_swivel_weight * swivel_continuity_cost(
                    geometry, previous, goal, old_goal, target_jump)
            else:
                posture_cost = static_elbow_motion_cost(current.upper_arm, old.upper_arm, ray,
                                                        goal, old_goal, prior_pose)
            if self.elbow_envelope_weight:
                from .stability import elbow_envelope_cost
                posture_cost = posture_cost + self.elbow_envelope_weight * elbow_envelope_cost(
                    geometry, previous, goal, self.posture.outward, self.elbow_bounds)
        else:
            posture_cost = self.posture.penalty(current, old, ray, swivel, target, old_target, frame=phase[0])
        if self.posture.motion is not None:
            weights = self.posture.motion
            relative_rate = (measured.relative_position - prior_pose.relative_position
                             - goal.relative_position + old_goal.relative_position) / DT
            wrist_motion = (.047 * torch.log1p(squared(wrist_rate_error) / .17**2).mean(-1)
                            + .036 * torch.log1p(squared(relative_rate) / .24**2)
                            + .034 * torch.log1p(chord2(actual_delta, goal_delta) / (DT * 1.9)**2).mean(-1))
            digit_rate_error = tip_rate_error
            if point_tracking:
                digit_rate_error = (measured.hand - prior_pose.hand) / DT - goal_point_rate
            digit_motion = .043 * torch.log1p(squared(digit_rate_error) / .13**2).flatten(1).mean(-1)
            digit_pose = .037 * (1. - score(squared(measured.hand[..., -1, :] - goal.hand[..., -1, :]), .005))
            acc = (state[:, DOF:] - previous_state[:, DOF:]).abs() / DT
            excess = (acc / self.acceleration_budget - 1.).clamp_min(0.)
            acceleration = torch.log1p(excess.square())
            # Normalize the 14 arm and 40 hand components independently.
            smoothness = (acceleration * self.acceleration_weight).sum(-1)
            reach = ((1. - score(squared(measured.wrist - goal.wrist), .0045))
                     + .65 * (1. - score(chord2(measured.rotation, goal.rotation), .040))
                     + .65 * (1. - score(squared(measured.relative_position - goal.relative_position), .008)))
            reward = (reward - weights.wrist * (wrist_motion + .025 * reach)
                      - weights.finger * (digit_motion + digit_pose)
                      - self.posture.goal_direction_cost(current, target, goal)
                      - weights.acceleration * smoothness)
            if weights.relative:
                separation = squared(measured.relative_position - goal.relative_position)
                orientation = chord2(measured.relative_rotation, goal.relative_rotation)
                relative_cost = (.034 * torch.log1p(separation / .016**2)
                                 + .032 * torch.log1p(orientation / .10**2))
                reward = reward - weights.relative * relative_cost
            if has_tail_cost:
                reward = reward - fingertip_tail_cost(tip_error, tip_rate_error,
                                                      position_scale=1. + finger_reward.tracking_boost)
            if self.posture.thumb_precision:
                thumb_error = squared(measured.hand[:, :, 0, -3:] - goal.hand[:, :, 0, -3:])
                thumb_shape = torch.log1p(thumb_error / .009**2).mean((1, 2))
                thumb_rate = torch.log1p(squared(tip_rate_error[:, :, 0]) / .11**2).mean(-1)
                shape_weight = .080 if self.posture.thumb_stability else .055
                reward = reward - shape_weight * thumb_shape - .022 * thumb_rate
            if self.posture.wrist_precision:
                reward = reward - wrist_precision_tail_cost(measured, goal)
            if point_tracking:
                reward = reward - slow_finger_motion_cost(digit_rate_error, goal_point_rate)
            elif self.posture.thumb_stability:
                target_rate = (goal.hand[..., -1, :] - old_goal.hand[..., -1, :]) / DT
                reward = reward - slow_finger_motion_cost(tip_rate_error, target_rate)
        tracking_value = (reward - .10 * limit - precision_cost(measured, goal, self.posture.precise, self.posture.right_elbow)
                          - posture_cost)
        return self.final_tracking_score(tracking_value, measured, goal)


class DirectEnv:
    def __init__(self, backend, bank, model, seed, robust, *, record_transition=False, posture_profile=None,
                 switch=None, finger_reward=None, reference_hand_fk=False, observation_version=OBSERVATION_VERSION,
                 positive_tracking_reward=False, wrist_execution_weight=0., hand_execution_weight=0.,
                 stationary_reset_fraction=0., wrist_rotation_tail_weight=0., fast_hand_weight=0.,
                 recorded_clock_fraction=0.):
        if not 0. <= stationary_reset_fraction <= 1. or (stationary_reset_fraction and bank.split != "train"):
            raise ValueError("Stationary augmentation requires a training split and a fraction in [0,1]")
        self.backend, self.bank = backend, bank
        self.device, self.num_envs = backend.device, backend.num_envs
        self.task = DirectTask(model, bank.metadata, self.num_envs, self.device, seed + 1, robust,
                               finger_reward=finger_reward, proximal=reference_hand_fk,
                               observation_version=observation_version, positive_tracking_reward=positive_tracking_reward,
                               wrist_execution_weight=wrist_execution_weight,
                               hand_execution_weight=hand_execution_weight,
                               wrist_rotation_tail_weight=wrist_rotation_tail_weight, fast_hand_weight=fast_hand_weight)
        self.task.stability = bool(getattr(bank, "stability", False))
        self.task.stability_v2 = bool(getattr(bank, "stability_v2", False))
        if self.task.stability or reference_hand_fk or self.task.observe.current_only:
            bank.set_reference_hand_targets(self.task.observe.fk)
        if "thumb_projection" in bank.metadata and posture_profile != "natural_point_tracking":
            self.task.thumb_ip_joints = thumb_ip_indices(model)
            self.task.thumb_ip_reference = bank.rsi_q[:, list(self.task.thumb_ip_joints)]
        self.refine_posture = posture_profile is not None
        if self.refine_posture:
            self.task.posture = PostureReference(model, bank, self.task, posture_profile)
        self.domain = self.task.domain
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.switch_generator = torch.Generator(device=self.device).manual_seed(seed + 137)
        if switch is None:
            switch = SwitchConfig(probability=.10 if bank.split == "train" else 0.)
        self.reference_switch = ReferenceSwitch(bank, switch, causal=self.refine_posture)
        self.domain_buffers = apply_newton_domain(backend, self.task)
        self.record_transition = record_transition
        self.goal_visits = None
        self.steps = 0
        self.set_sampling_iteration(1)
        count = self.num_envs
        self.frame = torch.zeros(count, dtype=torch.long, device=self.device)
        self.elapsed = torch.zeros_like(self.frame)
        self.recorded_clock_fraction = recorded_clock_fraction
        if recorded_clock_fraction:
            bank.prepare_recorded_clock()
            self.recorded_clock = torch.zeros(count, dtype=torch.bool, device=self.device)
            self.recorded_generator = torch.Generator(device=self.device).manual_seed(seed + 641)
        self.stationary_reset_fraction = stationary_reset_fraction
        self.hold_min_steps = 250 if self.task.stability else STATIONARY_STEPS
        self.hold_max_steps = 500 if self.task.stability else STATIONARY_STEPS
        self.episode_limit = torch.zeros_like(self.frame)
        if stationary_reset_fraction:
            self.stationary_steps = torch.zeros_like(self.frame)
            self.stationary_generator = torch.Generator(device=self.device).manual_seed(seed + 613)
        self.bridge = torch.zeros_like(self.frame)
        self.previous_frame = torch.zeros_like(self.frame)
        self.previous_bridge = torch.zeros_like(self.bridge)
        self.last_success = torch.full_like(self.frame, -1)
        self.switch_recovery = torch.zeros_like(self.frame)
        self.state = torch.zeros(count, 2 * DOF, device=self.device)
        home = self.task.home_pose
        self.goal = Pose(*(value.expand(count, *value.shape[1:]).clone() for value in home))
        self.history = History(self.state.clone(), self.task.command.home.expand(count, -1).clone(), self.goal)
        self.geometry = self.task.observe.fk(self.state[:, :DOF])
        self.actual = self.geometry.pose
        self.transition = {}
        self.policy_history = torch.zeros(count, self.task.observe.history_frames, self.task.observe.frame_features,
                                          device=self.device)
        if self.refine_posture:
            self.goal_history = self.goal
            self.history_frame = torch.zeros_like(self.frame)
            self.history_bridge = torch.zeros_like(self.bridge)

    def set_sampling_iteration(self, iteration):
        progress = min(1., (iteration - 1) / SAMPLING_RAMP_UPDATES)
        self.segment_steps = 24 + round((SEGMENT_STEPS - 24) * progress)
        self.home_fraction = HOME_FRACTION * progress

    def _goal(self, frames, bridge):
        goal = self.bank.at(frames)
        if not torch.any(bridge > 0):
            return goal
        fraction = (1. - bridge / BRIDGE_STEPS).clamp(0., 1.)
        return bridge_pose(self.task.home_pose, goal, fraction)

    def observation(self, ids=None, *, current_only=False):
        if getattr(self.task.observe, "history_frames", 1) > 1 and not current_only:
            history = self.policy_history if ids is None else self.policy_history[ids]
            return history.flatten(1)
        state = self.state + self.domain.observation_bias
        goal = self.history.previous_goal if self.refine_posture else self.goal
        history = self.history
        if self.refine_posture:
            history = History(history.previous_state, history.previous_command, self.goal_history)
        if ids is None:
            return self.task.observe(state, goal, history)
        subset_goal = Pose(*(value[ids] for value in goal))
        old_goal = Pose(*(value[ids] for value in history.previous_goal))
        return self.task.observe(state[ids], subset_goal, History(history.previous_state[ids],
                                                                 history.previous_command[ids], old_goal))

    @torch.no_grad()
    def reset(self, ids=None, *, frames=None, home=None, exact=False):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        if frames is None:
            frames = self.bank.sample(len(ids), self.generator)
        if home is None:
            home = torch.rand(len(ids), generator=self.generator, device=self.device) < self.home_fraction
        command = self.task.command
        q = torch.where(home[:, None], command.home, self.bank.rsi_q[frames])
        hold = torch.zeros_like(home)
        if self.stationary_reset_fraction:
            if not exact:
                choose_hold = torch.rand(len(ids), device=self.device, generator=self.stationary_generator)
                hold = (choose_hold < self.stationary_reset_fraction) & ~home
            length = torch.randint(self.hold_min_steps, self.hold_max_steps + 1, (len(ids),),
                                   device=self.device, generator=self.stationary_generator)
            self.stationary_steps[ids] = hold.long() * length
        self.episode_limit[ids] = self.segment_steps
        if self.task.stability and self.stationary_reset_fraction:
            self.episode_limit[ids] = torch.where(hold, self.stationary_steps[ids], self.episode_limit[ids])
        qd = torch.where((home | hold)[:, None], 0., self.bank.rsi_qd[frames])
        advance = (~(home | hold)).long()
        recorded = None
        if self.recorded_clock_fraction:
            recorded = torch.zeros_like(home)
            if not exact:
                chance = torch.rand(len(ids), device=self.device, generator=self.recorded_generator)
                recorded = (chance < self.recorded_clock_fraction) & advance.bool() & self.bank.recorded_eligible[frames]
            self.recorded_clock[ids] = recorded
        current_frame = self.bank.advance_frame(frames, advance, recorded, max_steps=1)
        if recorded is not None:
            speed = (self.bank.rsi_q[current_frame] - self.bank.rsi_q[frames]) / DT
            qd = torch.where(recorded[:, None], speed.clamp(-command.slew / DT, command.slew / DT), qd)
        if not exact:
            q, qd = self.domain.reset_noise(q, qd, ids)
        q = torch.minimum(torch.maximum(q, command.lower), command.upper)
        self.backend.reset(ids, q, qd)
        self.frame[ids] = current_frame
        self.previous_frame[ids] = frames
        self.last_success[ids] = -1
        self.switch_recovery[ids] = 0
        self.elapsed[ids] = -home.long() * (BRIDGE_STEPS - 1)
        self.bridge[ids] = home.long() * (BRIDGE_STEPS - 1)
        self.previous_bridge[ids] = home.long() * BRIDGE_STEPS
        state = torch.cat((q, qd), dim=-1)
        self.state = self.state.index_copy(0, ids, state)
        target = self._goal(current_frame, self.bridge[ids])
        previous_target = self._goal(frames, home.long() * BRIDGE_STEPS)
        actual = self.task.observe.fk(q)
        if hasattr(self, "elbow_hold_anchor"):
            from .stability import capture_stationary_elbow_anchor
            self.elbow_hold_anchor[ids] = capture_stationary_elbow_anchor(actual, target, hold)
        self.goal = Pose(*(old.index_copy(0, ids, new) for old, new in zip(self.goal, target, strict=True)))
        self.actual = Pose(*(old.index_copy(0, ids, new) for old, new in zip(self.actual, actual.pose, strict=True)))
        self.geometry = Geometry(self.actual, *(old.index_copy(0, ids, new)
                                                 for old, new in zip(self.geometry[1:], actual[1:], strict=True)))
        previous_state = self.history.previous_state.index_copy(0, ids, state + self.domain.observation_bias[ids])
        previous_command = self.history.previous_command.index_copy(0, ids, q)
        old_target = Pose(*(old.index_copy(0, ids, new) for old, new in
                            zip(self.history.previous_goal, previous_target, strict=True)))
        self.history = History(previous_state, previous_command, old_target)
        if self.refine_posture:
            self.goal_history = Pose(*(old.index_copy(0, ids, new) for old, new in
                                       zip(self.goal_history, previous_target, strict=True)))
            self.history_frame[ids] = frames
            self.history_bridge[ids] = self.previous_bridge[ids]
        first = self.observation(ids, current_only=True)
        self.policy_history[ids] = first[:, None].expand(-1, self.task.observe.history_frames, -1)

    def tracking_failure(self, measured, goal, wrist_dist, wrist_rot):
        return wrist_tracking_failure(wrist_dist, wrist_rot)

    def command_targets(self, action):
        command = self.task.command(action, self.history.previous_command)
        applied = torch.where(self.domain.delay[:, None], self.history.previous_command, command)
        return command, applied

    @torch.no_grad()
    def step(self, action):
        task = self.task
        invalid_action = ~torch.isfinite(action).all(-1)
        action = action.nan_to_num(nan=0., posinf=0., neginf=0.)
        command, applied = self.command_targets(action)
        state, torque, depth, substep, flags, deep_substeps = self.backend.interval(applied, task)
        flags = flags | (NONFINITE * invalid_action.int())
        nonfinite = (flags & NONFINITE) != 0
        overflow = (flags & OVERFLOW) != 0
        invalid = nonfinite | overflow
        state = state.nan_to_num(nan=0., posinf=0., neginf=0.)
        depth = depth.nan_to_num(nan=HARD_DEPTH)
        geometry = task.observe.fk(state[:, :DOF])
        measured = geometry.pose
        previous = self.geometry if self.refine_posture else self.actual
        scored_goal = self.goal
        target_jump = self.bank.clip_id[self.frame] != self.bank.clip_id[self.previous_frame]
        if self.refine_posture:
            reward = task.tracking_reward(state, geometry, previous, self.goal, self.history.previous_goal,
                                          command, self.history.previous_command, previous_state=self.state,
                                          phase=(self.frame, self.bridge), old_phase=(self.previous_frame, self.previous_bridge),
                                          target_jump=target_jump, requested_action=action, delayed=self.domain.delay)
        else:
            reward = task.tracking_reward(state, geometry, previous, self.goal, self.history.previous_goal,
                                          command, self.history.previous_command, phase=(self.frame, self.bridge),
                                          target_jump=target_jump, requested_action=action, delayed=self.domain.delay)
        if hasattr(self, "elbow_hold_anchor"):
            from .stability import stationary_elbow_anchor_cost
            hold_cost, hold_displacement, hold_gate = stationary_elbow_anchor_cost(
                geometry, self.geometry, self.goal, self.history.previous_goal,
                self.elbow_hold_anchor, self.stationary_steps > 0,
            )
            reward -= task.elbow_hold_drift_weight * hold_cost
        reward -= .20 * ((depth - AUDIT_DEPTH) / (HARD_DEPTH - AUDIT_DEPTH)).clamp(0., 1.)
        reward -= (flags != 0).float()
        reward = torch.where(invalid, -1.5, reward)
        wrist_dist = squared(measured.wrist - self.goal.wrist)
        wrist_rot = chord2(measured.rotation, self.goal.rotation)
        tracking_lost = self.tracking_failure(measured, scored_goal, wrist_dist, wrist_rot)
        tracking_lost &= (flags == 0) & (self.switch_recovery == 0)
        physical_failure = flags != 0
        flags |= TRACKING * tracking_lost.int()
        reward -= .75 * tracking_lost.float()
        advancing = self.bridge == 0
        if self.goal_visits is not None:
            self.goal_visits.scatter_add_(0, self.frame, (advancing & (flags == 0)).to(torch.int32))
        clip_done = advancing & (self.frame == self.bank.end[self.frame])
        self.elapsed += 1
        horizon = self.elapsed >= getattr(self, "episode_limit", self.segment_steps)
        done = clip_done | horizon | (flags != 0)
        timeout = horizon & ~clip_done & (flags == 0)
        stationary = getattr(self, "stationary_steps", None)
        hold = None if stationary is None else stationary > 0
        recorded = getattr(self, "recorded_clock", None)
        reference = self.reference_switch.advance(
            measured, self.frame, self.bridge, self.last_success, self.switch_recovery,
            self.switch_generator, eligible=(~done if not getattr(self.task, "stability", False) or hold is None else ~done & ~hold),
            hold=hold, recorded=recorded)
        if stationary is not None:
            self.stationary_steps = torch.where(reference.switched, 0, (stationary - 1).clamp_min(0))
        before_push = state
        self.steps += 1
        if self.domain.robust and self.steps % 25 == 0:
            pushing = (self.domain.uniform((self.num_envs,)) < 25 * IMPULSE_PROBABILITY) & self.domain.randomized & ~done
            ids = pushing.nonzero().flatten()
            if len(ids):
                speed = state[ids, DOF:] + self.domain.velocity_impulse((len(ids), DOF))
                with self.backend.simulation_stream():
                    self.backend.robot.write_joint_velocity_to_sim_index(velocity=speed, env_ids=ids)
                    self.backend.scene.write_data_to_sim()
                    self.backend.sim.forward()
                    self.backend.scene.update(0.)
                state = torch.cat((state[:, :DOF], state[:, DOF:].index_copy(0, ids, speed)), dim=-1)
        if self.refine_posture:
            self.goal_history = self.history.previous_goal
            self.history_frame, self.history_bridge = self.previous_frame, self.previous_bridge
        self.history = History(self.state + self.domain.observation_bias, command, self.goal)
        self.state, self.actual = state, measured
        self.geometry = geometry
        self.previous_frame, self.previous_bridge = reference.previous_frame, reference.previous_bridge
        self.frame, self.bridge = reference.frame, reference.bridge
        self.last_success, self.switch_recovery = reference.last_success, reference.recovery
        self.goal = self._goal(self.frame, self.bridge)
        if self.refine_posture:
            received = self._goal(self.previous_frame, self.previous_bridge)
            self.history = History(self.history.previous_state, self.history.previous_command, received)
        if hasattr(self, "policy_history"):
            current_obs = self.observation(current_only=True)
            self.policy_history = torch.cat((self.policy_history[:, 1:], current_obs[:, None]), dim=1)
        errors = pose_errors(measured, scored_goal)
        rollout = {key: value.reshape(self.num_envs, -1).mean(-1) for key, value in errors.items()}
        events = dict(hard_contact=(flags & COLLISION) != 0, failures=flags != 0, invalid=invalid,
                      nonfinite=nonfinite, solver_overflow=overflow, tracking_lost=tracking_lost, timeout=timeout,
                      clip_done=clip_done & (flags == 0),
                      target_success=reference.success, switch_attempt=reference.attempted,
                      trajectory_switch=reference.switched, switch_unavailable=reference.unavailable,
                      switch_recovery=self.switch_recovery > 0)
        if task.finger_reward.opposition_weight:
            target_vectors = bilateral_tip_vectors(scored_goal)
            close = (target_vectors.norm(dim=-1) <= BILATERAL_NEAR_M) & ~invalid[:, None]
            separation_error = (bilateral_tip_vectors(measured) - target_vectors).norm(dim=-1)
            rollout["opposition_vector_error_mm_weighted"] = (
                1000. * separation_error.masked_fill(~close, 0.).sum(-1) / close.sum(-1).clamp_min(1))
            events["opposition_goal_share"] = close.any(-1)
        if hasattr(self, "elbow_hold_anchor"):
            valid_arms = (hold_gate > 1e-3) & ~invalid[:, None]
            drift_mm = 1e3 * hold_displacement * valid_arms
            for side, name in enumerate(("left", "right")):
                rollout[f"elbow_hold_drift_{name}_mm_weighted"] = drift_mm[:, side]
                events[f"elbow_hold_anchor_{name}_share"] = valid_arms[:, side]
        if hold is not None:
            events["stationary_goal"] = hold
        if recorded is not None:
            events["recorded_goal_clock"] = recorded
            events["recorded_frame_advance"] = (reference.frame > reference.previous_frame + 1) & ~reference.switched
        if self.record_transition:
            full_step = ((substep == 0) | (substep == SUBSTEPS)) & ~invalid
            self.transition = dict(x1=before_push, c=command, u=applied, torque=torque, depth=depth,
                                   contact=torch.stack((depth > AUDIT_DEPTH, depth > HARD_DEPTH), -1).float(),
                                   nan=invalid, drop=flags != 0, full_step=full_step, safety_failure=physical_failure,
                                   rollout_stats=rollout, rollout_event_flags=events)
        stats = {key: value.float().mean() for key, value in {**rollout, **events, "reward": reward}.items()}
        return reward, done, timeout, stats
