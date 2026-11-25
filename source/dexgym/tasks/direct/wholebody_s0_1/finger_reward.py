"""Fingertip tracking reward configuration."""

from dataclasses import dataclass
from math import isfinite

import torch

REWARD_VARIANT = "fingertip_tracking_v1"
STILL_COMMAND_VERSION = "still_finger_request_v2"
STILL_HUBER_VERSION = "still_finger_request_pseudo_huber_v3"
STILL_LANDMARK_VERSION = "still_finger_requested_landmarks_v1"
THUMB_IP_REWARD_VERSION = "projected_thumb_ip_tracking_v1"
OPPOSITION_REWARD_VERSION = "opposing_fingertip_vectors_v1"
BILATERAL_NEAR_M = .035
BILATERAL_FAR_M = .055


def thumb_ip_indices(model):
    return tuple(int(model.joint(f"{side}_thumb_ip").qposadr[0]) for side in "lr")


def bilateral_tip_vectors(pose):
    tips = pose.hand[..., -1, :]
    world = pose.wrist[:, :, None, :] + (pose.rotation[:, :, None] @ tips[..., None]).squeeze(-1)
    return world[:, 1] - world[:, 0]


def thumb_tip_vectors(pose):
    tips = pose.hand[..., -1, :]
    return (tips[:, :, 1:] - tips[:, :, :1]).flatten(1, 2)


def bilateral_opposition_gate(goal):
    closest = bilateral_tip_vectors(goal).norm(dim=-1).amin(-1)
    gate = ((BILATERAL_FAR_M - closest) / (BILATERAL_FAR_M - BILATERAL_NEAR_M)).clamp(0., 1.)
    return (gate.square() * (3. - 2. * gate)).detach()


def opposing_fingertip_cost(measured, previous, goal, old_goal, dt):
    """Follow relative tip positions and their motion as the target fingers approach."""
    costs = []
    for vectors, near, far in ((bilateral_tip_vectors, BILATERAL_NEAR_M, BILATERAL_FAR_M),
                               (thumb_tip_vectors, .021, .035)):
        target = vectors(goal)
        delta = vectors(measured) - target
        prior_delta = vectors(previous) - vectors(old_goal)
        weight = ((far - target.norm(dim=-1)) / (far - near)).clamp(0., 1.).detach()
        position = torch.log1p(delta.square().sum(-1) / .012**2)
        velocity = torch.log1p((delta - prior_delta).square().sum(-1) / (dt * .12)**2)
        costs.append((weight * (position + .25 * velocity)).sum(-1) / weight.sum(-1).clamp_min(1.))
    bilateral, thumb = costs
    return bilateral + .6 * thumb


@dataclass(frozen=True)
class FingerRewardConfig:
    tracking_boost: float = .15
    still_command_weight: float = 0.
    still_command_loss: str = "log"
    still_landmark_weight: float = 0.
    opposition_weight: float = 0.
    opposition_wrist_position_relaxation: float = 0.
    opposition_wrist_rotation_relaxation: float = 0.

    def __post_init__(self):
        if not isfinite(self.tracking_boost) or self.tracking_boost < 0.:
            raise ValueError("Finger tracking boost must be finite and nonnegative")
        if not isfinite(self.still_command_weight) or self.still_command_weight < 0.:
            raise ValueError("Still finger command weight must be finite and nonnegative")
        if self.still_command_loss not in ("log", "pseudo_huber"):
            raise ValueError("Unknown still-finger command loss")
        if not isfinite(self.still_landmark_weight) or self.still_landmark_weight < 0.:
            raise ValueError("Still finger landmark weight must be finite and nonnegative")
        if not 0. <= self.opposition_weight <= 1.:
            raise ValueError("Opposition weight must be in [0,1]")
        relaxations = (self.opposition_wrist_position_relaxation, self.opposition_wrist_rotation_relaxation)
        if not all(0. <= value <= 1. for value in relaxations):
            raise ValueError("Opposition wrist relaxation weights must be in [0,1]")

    def to_dict(self):
        result = {
            "version": REWARD_VARIANT,
            "tracking_boost": self.tracking_boost,
        }
        if self.still_command_weight:
            result["still_command_weight"] = self.still_command_weight
            result["still_command_loss"] = self.still_command_loss
        if self.still_landmark_weight:
            result["still_landmark_weight"] = self.still_landmark_weight
        if self.opposition_weight:
            result["opposition_weight"] = self.opposition_weight
        if self.opposition_wrist_position_relaxation:
            result["opposition_wrist_position_relaxation"] = self.opposition_wrist_position_relaxation
        if self.opposition_wrist_rotation_relaxation:
            result["opposition_wrist_rotation_relaxation"] = self.opposition_wrist_rotation_relaxation
        return result


def add_finger_reward_arguments(parser):
    defaults = FingerRewardConfig()
    parser.add_argument("--finger-tracking-boost", type=float, default=defaults.tracking_boost,
                        help="Add this fraction of the wrist-local fingertip position precision penalty")
    parser.add_argument("--still-finger-command-weight", type=float, default=defaults.still_command_weight,
                        help="Penalize large requested joint position changes when the corresponding hand targets are nearly still")
    parser.add_argument("--still-finger-command-loss", choices=("log", "pseudo_huber"), default=defaults.still_command_loss,
                        help="Position-request loss for slow-moving finger targets")
    parser.add_argument("--still-finger-landmark-weight", type=float, default=defaults.still_landmark_weight,
                        help="Track the wrist-local target landmarks of the requested pose during slow motion")
    parser.add_argument("--opposition-weight", type=float, default=defaults.opposition_weight,
                        help="Track relative fingertip positions and motion near opposition poses")
    parser.add_argument("--opposition-wrist-position-relaxation", type=float,
                        default=defaults.opposition_wrist_position_relaxation,
                        help="Soften absolute and relative wrist position errors near a bilateral fingertip target")
    parser.add_argument("--opposition-wrist-rotation-relaxation", type=float,
                        default=defaults.opposition_wrist_rotation_relaxation,
                        help="Soften absolute and relative wrist rotation errors near a bilateral fingertip target")


def finger_reward_config(args):
    return FingerRewardConfig(args.finger_tracking_boost, getattr(args, "still_finger_command_weight", 0.),
                              getattr(args, "still_finger_command_loss", "log"),
                              getattr(args, "still_finger_landmark_weight", 0.),
                              getattr(args, "opposition_weight", 0.),
                              getattr(args, "opposition_wrist_position_relaxation", 0.),
                              getattr(args, "opposition_wrist_rotation_relaxation", 0.))
