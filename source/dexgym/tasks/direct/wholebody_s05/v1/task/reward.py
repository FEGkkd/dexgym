"""Frozen S0-5 reward with an exact physical and smooth imagination path."""

from typing import NamedTuple

import torch
from torch import Tensor

from ....hand2_cube_supermimic.rotations import mm, so3_log
from .geometry import Geometry, GeometryVelocity
from .observation import GeometryMask, TargetFrame, VelocityMask


class Reward(NamedTuple):
    value: Tensor
    terms: dict[str, Tensor]
    penalties: dict[str, Tensor]


def distance(error, soft=False):
    if not soft:
        return torch.linalg.vector_norm(error, dim=-1)
    return ((error * error).sum(-1) + 1e-12).sqrt() - 1e-6


def similarity(e, scales):
    values = e[..., None] / e.new_tensor(scales)
    return (e.new_tensor((.25, .35, .40)) / (1 + values.square())).sum(-1)


def masked_mean(value, mask):
    axes = tuple(range(1, value.ndim))
    numerator = torch.where(mask, value, 0.).sum(dim=axes)
    return numerator / mask.sum(dim=axes).clamp_min(1)


def _worst(value, valid, soft):
    value = value.flatten(1)
    valid = valid.flatten(1)
    active = valid.any(-1)
    if soft:
        terms = torch.where(valid, -value / .05, -torch.inf)
        terms = torch.where(active[:, None], terms, 0.)
        return torch.where(active, -.05 * (torch.logsumexp(terms, dim=-1) - valid.sum(-1).clamp_min(1).log()), 0.)
    return torch.where(active, torch.where(valid, value, 1.).amin(-1), 0.)


class GeometryReward:
    def __init__(self, *, clearance, margin_band=.0045):
        self.clearance = clearance
        self.margin_band = margin_band

    def __call__(self, evaluated: Geometry, velocity: GeometryVelocity, actual: GeometryMask, actual_v: VelocityMask,
                 target: TargetFrame, residual: Tensor, previous: Tensor, earlier: Tensor, scale: Tensor,
                 previous_valid: Tensor, earlier_valid: Tensor, projected: Tensor, clearance: Tensor,
                 valid: Tensor, *, soft=False) -> Reward:
        target_g, target_v = target.geometry, target.velocity
        g_mask, v_mask = target.geometry_mask, target.velocity_mask
        def pos(diff):
            return similarity(1000 * distance(diff, soft), (20., 5., 2.))
        def angle(a, b):
            radians = distance(so3_log(mm(a.transpose(-1, -2), b)), soft)
            return similarity(torch.rad2deg(radians), (20., 5., 2.))
        def vel(a, b, angular=False):
            diff = distance(a - b, soft)
            if angular:
                return similarity(torch.rad2deg(diff), (60., 20., 5.))
            return similarity(1000 * diff, (150., 50., 15.))
        def sized(diff):
            return (diff / scale).square().clamp(max=1).mean(-1)

        tips = pos(evaluated.tips - target_g.tips)
        valid_tips = actual.tips & g_mask.tips
        worst = _worst(tips, valid_tips, soft)
        global_score = .65 * masked_mean(tips, valid_tips) + .35 * worst
        local = masked_mean(pos(evaluated.local - target_g.local), actual.local & g_mask.local)
        pinch_gap = (distance(evaluated.pinch, soft) - distance(target_g.pinch, soft)).abs()
        pinch_score = .5 * (pos(evaluated.pinch - target_g.pinch) + similarity(1000 * pinch_gap, (20., 5., 2.)))
        pinch = masked_mean(pinch_score, actual.pinch & g_mask.pinch & target.pinch_active)
        cross = masked_mean(pos(evaluated.cross - target_g.cross), actual.cross & g_mask.cross)
        ap, ar = evaluated.relative
        gp, gr = target_g.relative
        relative = .5 * (pos(ap - gp) + angle(ar, gr))
        bimanual = .5 * (cross + torch.where(actual.relative & g_mask.relative, relative, 0.))
        wrist = masked_mean(.5 * (pos(evaluated.wrist - target_g.wrist) + angle(evaluated.rotation, target_g.rotation)),
                            actual.wrist & g_mask.wrist)
        tip_velocity = masked_mean(vel(velocity.tips, target_v.tips), actual_v.tips & v_mask.tips)
        wrist_velocity = .5 * (vel(velocity.wrist_linear, target_v.wrist_linear)
                               + vel(velocity.wrist_angular, target_v.wrist_angular, angular=True))
        velocity_score = .5 * (tip_velocity + masked_mean(wrist_velocity, actual_v.wrist & v_mask.wrist))

        terms = dict(global_tip=global_score, local=local, pinch=pinch, bimanual=bimanual, wrist=wrist,
                     velocity=velocity_score)
        near_contact = (1 - (clearance - self.clearance) / self.margin_band).clamp(0, 1)
        penalties = dict(tail=1 - worst, residual=sized(residual),
                         smoothness=torch.where(previous_valid, sized(residual - previous), 0.),
                         acceleration=torch.where(previous_valid & earlier_valid,
                                                  sized(residual - 2 * previous + earlier), 0.),
                         safety=.5 * (projected.clamp(0, 1) + near_contact))
        # Independent safety determines eligibility before this scalar can reach learning.
        positive = 4 * global_score + 2 * (local + pinch + bimanual) + wrist + velocity_score
        negative = (.3 * penalties["tail"] + .1 * (penalties["residual"] + penalties["smoothness"])
                    + .03 * penalties["acceleration"] + .2 * penalties["safety"])
        value = ((positive - negative + .73) / 12.73).clamp(0, 1)
        return Reward(torch.where(valid & torch.isfinite(value), value, 0.), terms, penalties)
