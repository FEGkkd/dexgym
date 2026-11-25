"""Training-only shape and elbow priors derived from the archived reference."""

import math
from typing import NamedTuple

import mujoco
import numpy as np
import torch
from torch.nn.functional import normalize

from ..wholebody_s0.robot import FIRMWARE_FINGERS
from . import DT
from .kinematics import Pose

PRIOR_VERSIONS = {
    "balanced": "distal_elbow_precision_v1",
    "precision": "distal_elbow_precision_v2",
    "preserve_arm": "distal_elbow_precision_v3",
    "right_elbow": "right_elbow_knuckle_precision_v4",
    "mixed_wrist": "mixed_egopro_physical_wrist_v1",
    "natural_motion": "natural_motion_v1",
    "natural_fingers": "natural_motion_finger_priority_v1",
    "natural_reach": "natural_motion_wrist_priority_v2",
    "natural_elbows": "natural_motion_elbow_priority_v2",
    "natural_tip_precision": "natural_motion_tip_precision_v1",
    "natural_thumb": "natural_thumb_tracking_v1",
    "natural_thumb_accuracy": "natural_thumb_goal_precision_v2",
    "natural_thumb_stable": "natural_thumb_stable_precision_v3",
    "natural_goal_stable": "natural_goal_stable_precision_v4",
    "natural_point_tracking": "natural_four_point_tracking_v1",
}
HYPEREXTENSION_RAD = math.radians(4.)
ELBOW_SMOOTHING_WEIGHTS = (1., 2., 3., 2., 1.)


def distal_vector(pose: Pose):
    return pose.hand[..., -1, :] - pose.hand[..., -2, :]


def distal_indices(model):
    return [int(model.jnt_qposadr[int(model.actuator(f"{side}_{finger}{number}").trnid[0])])
            for side in "lr" for finger in FIRMWARE_FINGERS for number in (2, 3)]


def extension_excess(actual, reference):
    allowed_minimum = reference[..., 1:, :].clamp(max=0.) - HYPEREXTENSION_RAD
    return (allowed_minimum - actual[..., 1:, :]).clamp_min(0.)


def swivel_tangent(ray, reference_upper_arm):
    return normalize(torch.linalg.cross(ray, reference_upper_arm), dim=-1, eps=1e-5)


def rightward_shoulder_axis(model, home):
    data = mujoco.MjData(model)
    data.qpos[:] = home
    mujoco.mj_kinematics(model, data)
    rightward = data.body("Link2_R").xpos - data.body("Link2_L").xpos
    return rightward / np.linalg.norm(rightward)


class Posture(NamedTuple):
    distal: torch.Tensor
    flexion: torch.Tensor
    upper_arm: torch.Tensor


class MotionWeights(NamedTuple):
    finger: float
    wrist: float
    elbow: float
    acceleration: float
    relative: float = 0.


MOTION_WEIGHTS = {
    "natural_motion": MotionWeights(1., 1., 1., 1.),
    "natural_fingers": MotionWeights(1.35, 1., 1., 1.),
    "natural_reach": MotionWeights(1., 1.35, .95, .65, 1.25),
    "natural_elbows": MotionWeights(1., 1., 1.15, 1., 1.),
    "natural_tip_precision": MotionWeights(1., 1., 1.15, 1., 1.),
    "natural_thumb": MotionWeights(1., 1.12, 1.15, 1.1, 1.10),
    "natural_thumb_accuracy": MotionWeights(1., 1.12, 1.15, 1.1, 1.10),
    "natural_thumb_stable": MotionWeights(1., 1.12, 1.15, 1.1, 1.10),
    "natural_goal_stable": MotionWeights(1., 1.12, 1.15, 1.1, 1.10),
}
MOTION_WEIGHTS["natural_point_tracking"] = MOTION_WEIGHTS["natural_goal_stable"]
ARM_TEACHER_PROFILES = ("preserve_arm", "right_elbow", "mixed_wrist", *MOTION_WEIGHTS)


def finger_reference_confidence(reference_hand, goal_hand):
    """Training-only per-digit confidence; independent of policy action/state."""
    error = torch.linalg.vector_norm(reference_hand - goal_hand, dim=-1).amax(-1)
    return .15 + .85 / (1. + (error / .012).pow(4))


class PostureReference:
    def __init__(self, model, bank, task, profile):
        device = bank.rsi_q.device
        self.indices = torch.tensor(distal_indices(model), device=device).reshape(2, 5, 2)
        self.angle_slack = torch.tensor(((.28, .24), (.20, .13), (.20, .13), (.20, .13), (.20, .13)),
                                        device=device)[None, None]
        self.precise = profile != "balanced"
        self.motion = MOTION_WEIGHTS.get(profile)
        self.point_tracking = profile == "natural_point_tracking"
        if self.point_tracking and bank.pose.hand.shape[-2] != 4:
            raise ValueError("Four-point tracking requires four FK landmarks per finger")
        self.finger_goal_priority = self.point_tracking or profile == "natural_goal_stable"
        self.thumb_stability = profile in ("natural_thumb_stable", "natural_goal_stable")
        self.thumb_accuracy = profile in ("natural_thumb_accuracy", "natural_thumb_stable", "natural_goal_stable")
        self.thumb_precision = profile in ("natural_thumb", "natural_thumb_accuracy", "natural_thumb_stable", "natural_goal_stable")
        self.tip_precision = self.point_tracking or profile in ("natural_tip_precision", "natural_thumb", "natural_thumb_accuracy", "natural_thumb_stable", "natural_goal_stable")
        inward_profiles = ("natural_elbows", "natural_tip_precision", "natural_thumb", "natural_thumb_accuracy",
                           "natural_thumb_stable", "natural_goal_stable", "natural_point_tracking")
        self.teleop_inward = ~bank.egopro if profile in inward_profiles else None
        self.wrist_precision = self.thumb_accuracy or self.point_tracking
        self.right_elbow = profile in ("right_elbow", "mixed_wrist") or self.motion is not None
        self.mixed_wrist = profile == "mixed_wrist" or self.motion is not None
        if self.right_elbow:
            axis = rightward_shoulder_axis(model, task.command.home.cpu().numpy())
            self.rightward = task.command.home.new_tensor(axis)
        if self.motion is not None:
            self.outward = torch.stack((-self.rightward, self.rightward))
        self.distal_weights = torch.tensor((.55, 1., 1., 1., 1.), device=device)
        if self.thumb_precision:
            self.distal_weights[0] = 1.
        if self.thumb_accuracy:
            # The legacy teleop thumb posture can disagree with the mapped
            # skeleton. Track the supplied thumb geometry, with the anatomical
            # penalty below, instead of opposing it with a joint-reference prior.
            self.distal_weights[0] = 0.
        self.home = self.features(task.command.home[None], task.home_geometry)
        with torch.no_grad():
            reference, elbow_confidence, finger_confidence, offset = [], [], [], 0
            for q in bank.rsi_q.split(1024):
                geometry = task.observe.fk(q)
                reference.append(self.features(q, geometry))
                if self.wrist_precision:
                    error = torch.linalg.vector_norm(
                        geometry.pose.wrist - bank.pose.wrist[offset:offset + len(q)], dim=-1)
                    # An elbow prior derived from a wrist-mismatched IK pose must
                    # not outweigh the supplied task-space wrist target.
                    rotation_error = (.5 * (geometry.pose.rotation
                        - bank.pose.rotation[offset:offset + len(q)]).square().sum(dim=(-1, -2))).sqrt()
                    elbow_confidence.append(.15 + .85 / (1. + (error / .012).pow(4)
                        + (rotation_error / math.radians(5.)).pow(4)))
                if self.finger_goal_priority and not self.point_tracking:
                    finger_confidence.append(finger_reference_confidence(
                        geometry.pose.hand, bank.pose.hand[offset:offset + len(q)]))
                offset += len(q)
            self.elbow_confidence = torch.cat(elbow_confidence) if self.wrist_precision else None
            self.finger_confidence = torch.cat(finger_confidence) if finger_confidence else None
            distal, flexion, upper_arm = (torch.cat(batch) for batch in zip(*reference, strict=True))
            frame = torch.arange(len(upper_arm), device=device)
            starts = torch.tensor([entry["start"] for entry in bank.clips], device=device)
            start = starts[torch.searchsorted(starts, frame, right=True) - 1]
            offsets = torch.arange(-2, 3, device=device)
            neighborhood = torch.minimum(torch.maximum(frame[:, None] + offsets, start[:, None]), bank.end[:, None])
            weights = upper_arm.new_tensor(ELBOW_SMOOTHING_WEIGHTS) / sum(ELBOW_SMOOTHING_WEIGHTS)
            upper_arm = (upper_arm[neighborhood] * weights[None, :, None, None]).sum(1)
            self.reference = Posture(distal, flexion, upper_arm)

    def features(self, q, geometry):
        return Posture(distal_vector(geometry.pose), q[:, self.indices], geometry.upper_arm)

    def at(self, frame, bridge, bridge_steps):
        t = 1. - bridge.to(dtype=self.home.flexion.dtype) / bridge_steps
        mix = t.square() * (3. - 2. * t)
        return Posture(*(torch.lerp(initial, archived[frame], mix.reshape((-1,) + (1,) * (archived.ndim - 1)))
                         for initial, archived in zip(self.home, self.reference, strict=True)))

    def penalty(self, actual, previous, ray, swivel, target, old_target, *, frame):
        if not self.point_tracking:
            direction = normalize(actual.distal, dim=-1, eps=1e-6)
            ref_direction = normalize(target.distal, dim=-1, eps=1e-6)
            direction_error = (direction - ref_direction).square().sum(-1)
            # A broad tolerance preserves freedom to improve MANUS landmark alignment.
            direction_excess = (direction_error - .12**2).clamp_min(0.)
            distal_values = torch.log1p(direction_excess / .23**2)
            distal_cost = distal_values.mean((1, 2))
            angle_excess = ((actual.flexion - target.flexion).abs() - self.angle_slack).clamp_min(0.)
            angle_values = torch.log1p((angle_excess / .25).square())
            angle_cost = angle_values.mean((1, 2, 3))
            if self.precise:
                distal_cost = (distal_values * self.distal_weights).mean((1, 2))
                angle_cost = (angle_values * self.distal_weights[:, None]).mean((1, 2, 3))
            if self.finger_goal_priority:
                confidence = self.finger_confidence[frame]
                # Preserve compatible priors; conflicting archived joint poses are
                # secondary to the current wrist-local geometry supplied to policy.
                distal_cost = (distal_values * self.distal_weights * confidence).mean((1, 2))
                angle_cost = (angle_values * self.distal_weights[:, None] * confidence[..., None]).mean((1, 2, 3))
            extension = extension_excess(actual.flexion, target.flexion)
            extension_cost = torch.log1p((extension / .13).square()).mean((1, 2, 3))

        ref_swivel = target.upper_arm - (target.upper_arm * ray).sum(-1, keepdim=True) * ray
        arm_delta = actual.upper_arm - previous.upper_arm - target.upper_arm + old_target.upper_arm
        tangent = swivel_tangent(ray, target.upper_arm)
        lateral_motion = (arm_delta * tangent).sum(-1) / DT
        swivel_error = (swivel - ref_swivel).square().sum(-1)
        confidence = self.elbow_confidence[frame] if self.elbow_confidence is not None else 1.
        elbow_cost = (confidence * torch.log1p(swivel_error / .040**2)).mean(-1)
        motion_cost = (confidence * torch.log1p((lateral_motion / .09).square())).mean(-1)

        if self.point_tracking:
            cost = .095 * elbow_cost + .030 * motion_cost
        else:
            distal_weight, angle_weight = (.050, .032) if self.precise else (.075, .060)
            cost = (distal_weight * distal_cost + angle_weight * angle_cost + .055 * extension_cost
                    + .095 * elbow_cost + .030 * motion_cost)
        if self.motion is not None:
            lateral_error = ((actual.upper_arm - target.upper_arm) * self.outward).sum(-1)
            lateral_speed = (arm_delta * self.outward).sum(-1) / DT
            # Both inward and outward deviations can accompany a wrist reached with the wrong arm posture.
            excess = (lateral_error.abs() - .010).clamp_min(0.)
            position = confidence * torch.log1p((excess / .026).square())
            motion = confidence * torch.log1p((lateral_speed / .055).square())
            cost = cost + self.motion.elbow * (.172 * position.mean(-1) + .035 * position.amax(-1)
                    + .100 * motion.mean(-1) + .030 * (confidence * torch.log1p((lateral_motion / .065).square())).mean(-1))
            if self.teleop_inward is not None:
                # Large inward errors in the teleop clips need a sustained gradient.
                # The two-sided terms above still apply to both data sources.
                inward = (-lateral_error - .020).clamp_min(0.) / .035
                tail = confidence * (torch.sqrt(1. + inward.square()) - 1.)
                cost = cost + self.teleop_inward[frame] * (.055 * tail.mean(-1) + .025 * tail.amax(-1))
        elif self.right_elbow:
            lateral_error = ((actual.upper_arm[:, 1] - target.upper_arm[:, 1]) * self.rightward).sum(-1)
            outer_excess = (lateral_error - .010).clamp_min(0.)
            lateral_speed_error = (arm_delta[:, 1] * self.rightward).sum(-1) / DT
            # The two-sided term keeps penalizing an elbow that moves too far inward.
            cost = (cost + .130 * torch.log1p((outer_excess / .025).square())
                    + .045 * torch.log1p((lateral_error / .040).square())
                    + .075 * torch.log1p((lateral_speed_error / .050).square())
                    + .033 * torch.log1p((lateral_motion[:, 1] / .063).square()))
        if self.mixed_wrist and not self.point_tracking:
            extension = (-actual.flexion[..., 1:, :] - HYPEREXTENSION_RAD).clamp_min(0.)
            cost = cost + .045 * torch.log1p((extension / .085).square()).mean((1, 2, 3))
            if self.motion is not None:
                tail = torch.log1p((extension / .065).square()).amax(dim=(-1, -2)).mean(-1)
                cost = cost + self.motion.finger * .038 * tail
        if self.thumb_precision:
            # Positive MCP/IP flexion curls both mirrored thumbs toward the palm.
            # Use a soft anatomical penalty so the policy learns the correction.
            thumb = actual.flexion[..., 0, :]
            extension = (-thumb - HYPEREXTENSION_RAD).clamp_min(0.)
            scaled = extension / (.070 if self.thumb_stability else .12)
            extension_loss = torch.sqrt(1. + scaled.square()) - 1.
            # A single reversed IP cannot be hidden by averaging the other digits.
            mean_weight, worst_weight = (.160, .100) if self.thumb_stability else (.130, .080)
            cost = cost + mean_weight * extension_loss.mean((1, 2)) + worst_weight * extension_loss.amax(dim=(1, 2))
            if not self.thumb_accuracy:
                direction = distal_values[..., 0]
                cost = cost + .045 * direction.mean(-1)
        return cost

    def goal_direction_cost(self, actual, target, goal):
        desired = normalize(distal_vector(goal), dim=-1, eps=1e-6)
        reachable = normalize(target.distal, dim=-1, eps=1e-6)
        current = normalize(actual.distal, dim=-1, eps=1e-6)
        lower, upper = math.cos(math.radians(45.)), math.cos(math.radians(20.))
        trust = (((desired * reachable).sum(-1) - lower) / (upper - lower)).clamp(0., 1.)
        if self.finger_goal_priority:
            # The target's distal direction follows its two supplied landmarks.
            # An incompatible archived pose must not disable this goal term.
            trust = torch.ones_like(trust)
        chord = (current - desired).square().sum(-1)
        if self.thumb_accuracy:
            trust = trust.clone()
            trust[..., 0] = 1.
        cost = trust * torch.log1p((chord - .105**2).clamp_min(0.) / .26**2)
        penalty = .052 * cost.mean((1, 2)) + .025 * cost.amax(-1).mean(-1)
        if self.thumb_accuracy:
            weight = .025 if self.thumb_stability else .045
            penalty = penalty + weight * cost[..., 0].mean(-1)
        return self.motion.finger * penalty
