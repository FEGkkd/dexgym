"""Additional Wuji landmark targets in the recorded world wrist frame."""
import os

import torch

BASE_FRAME_FEATURES = 4 * 54 + 3 * 30
EXTRA_FRAME_FEATURES = 3 * 90
GOAL_POINTS = int(os.environ.get("S04_GOAL_POINTS", "5"))
if GOAL_POINTS not in (5, 20):
    raise ValueError("S04_GOAL_POINTS must be 5 or 20")


def pad_legacy_observation(value, history=5):
    if value.shape[-1] != history * BASE_FRAME_FEATURES:
        raise ValueError("Expected the original S04 observation")
    rows = value.reshape(*value.shape[:-1], history, BASE_FRAME_FEATURES)
    return torch.nn.functional.pad(rows, (0, EXTRA_FRAME_FEATURES)).flatten(-2)


def non_tip_local(pose, proximal):
    anchor = proximal.unsqueeze(0).expand(pose.hand.shape[0], -1, -1, -1)
    return torch.cat((anchor.unsqueeze(-2), pose.hand[..., :-1, :]), dim=-2).flatten(2,3)


def transform(pose, points):
    return pose.wrist[:, :, None] + points @ pose.rotation.transpose(-1,-2)


def articulated_points(pose, proximal):
    return transform(pose, non_tip_local(pose, proximal))


def arm_landmark_cost(measured, goal, proximal):
    """Use the same reference shape under both wrists to isolate the 14 arm axes."""
    local = non_tip_local(goal, proximal)
    error_sq = (transform(measured,local) - transform(goal,local)).square().sum(-1)
    robust = torch.sqrt(1. + error_sq/.025**2) - 1.
    return .7*robust.mean((1,2)) + .3*robust.amax((1,2))


def worst_rotation_deg(measured, goal):
    relative = measured.rotation @ goal.rotation.transpose(-1,-2)
    w = torch.stack((relative[...,2,1]-relative[...,1,2],
                     relative[...,0,2]-relative[...,2,0],
                     relative[...,1,0]-relative[...,0,1]), -1)
    sine = .5*w.norm(dim=-1)
    cosine = .5*(relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1.)
    return torch.atan2(sine,cosine).rad2deg().amax(dim=-1)
