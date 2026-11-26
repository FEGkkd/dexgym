"""Shared causal S0-5 observation ABI with explicit per-field masks."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from . import DT, HISTORY, JOINTS, OBSERVATION_FRAME
from .geometry import Geometry, GeometryVelocity, error6, pose9


class GeometryMask(NamedTuple):
    wrist: Tensor  # B x 2
    tips: Tensor  # B x 2 x 5
    local: Tensor  # B x 2 x 5 x 4
    pinch: Tensor  # B x 2 x 4
    cross: Tensor  # B x 5 x 5
    relative: Tensor  # B


class VelocityMask(NamedTuple):
    wrist: Tensor  # B x 2
    tips: Tensor  # B x 2 x 5


class Feedback(NamedTuple):
    q: Tensor  # B x 54
    qd: Tensor
    geometry: Geometry
    velocity: GeometryVelocity
    geometry_mask: GeometryMask
    velocity_mask: VelocityMask
    valid: Tensor  # B x 2, q then qd
    age_seconds: Tensor  # B
    age_known: Tensor  # B


class TargetFrame(NamedTuple):
    nominal: Tensor
    nominal_velocity: Tensor
    nominal_valid: Tensor  # B x 2, all current joints then all velocities
    geometry: Geometry
    velocity: GeometryVelocity
    geometry_mask: GeometryMask
    velocity_mask: VelocityMask
    stream_valid: Tensor  # B x 4, MANUS L/R then Tracker L/R
    age_seconds: Tensor  # B x 4
    age_known: Tensor  # B x 4, validity and clock knowledge are independent
    permitted: Tensor  # B, causal safety and all required target geometry
    pinch_active: Tensor  # B x 2 x 4, frozen independent target mask
    generation: Tensor  # B, strictly control/history metadata
    tick_ns: Tensor  # B, strictly control/history metadata


class CommandMemory(NamedTuple):
    requested: Tensor
    issued: Tensor
    residual: Tensor
    servo: Tensor
    previous_valid: Tensor
    servo_valid: Tensor


def geometry_mask(valid: Tensor) -> GeometryMask:
    n = valid.shape[0]
    return GeometryMask(valid[:, None].expand(n, 2), valid[:, None, None].expand(n, 2, 5),
                        valid[:, None, None, None].expand(n, 2, 5, 4), valid[:, None, None].expand(n, 2, 4),
                        valid[:, None, None].expand(n, 5, 5), valid)


def velocity_mask(valid: Tensor) -> VelocityMask:
    n = valid.shape[0]
    return VelocityMask(valid[:, None].expand(n, 2), valid[:, None, None].expand(n, 2, 5))


def _masked(values, valid):
    for _ in range(values.ndim - valid.ndim):
        valid = valid.unsqueeze(-1)
    return torch.where(valid & torch.isfinite(values), values, 0)


def _join(parts):
    return torch.cat(tuple(part.reshape(part.shape[0], -1) for part in parts), dim=-1)


def _triple(actual, target, actual_mask, target_mask, scale):
    return tuple(_masked(value / scale, mask) for value, mask in (
        (actual, actual_mask), (target, target_mask), (target - actual, actual_mask & target_mask)
    ))


def _pose_triplet(actual_p, actual_r, goal_p, goal_r, a_mask, g_mask):
    a = pose9(actual_p / .1, actual_r)
    g = pose9(goal_p / .1, goal_r)
    delta = error6(actual_p / .1, actual_r, goal_p / .1, goal_r)
    return _masked(a, a_mask), _masked(g, g_mask), _masked(delta, a_mask & g_mask)


def _velocity_triple(actual, goal, a_mask, g_mask):
    def spatial(v):
        return torch.cat((v.wrist_linear / .5, v.wrist_angular / 5), dim=-1)
    wrist = _triple(spatial(actual), spatial(goal), a_mask.wrist, g_mask.wrist, 1)
    tips = _triple(actual.tips, goal.tips, a_mask.tips, g_mask.tips, .5)
    return wrist, tips


class ObservationBuilder(nn.Module):
    """No episode indices, future features, or source labels enter this builder."""

    def __init__(self, *, compact=False):
        super().__init__()
        columns = torch.ones(OBSERVATION_FRAME, dtype=torch.bool)
        if compact:
            columns[1236:1461] = False
            # Quality: age 10, input 4, q/nominal 4, wrist 8, tips 40, local 80, pinch 16.
            cross_start = 1485 + 10 + 4 + 4 + 8 + 40 + 80 + 16
            columns[cross_start:cross_start + 50] = False
        self.register_buffer("columns", columns)
        self.compact = compact
        self.frame_dim = 1428 if compact else OBSERVATION_FRAME

    def forward(self, feedback: Feedback, target: TargetFrame, memory: CommandMemory) -> Tensor:
        q_mask, qd_mask = feedback.valid.unbind(-1)
        n_mask, nd_mask = target.nominal_valid.unbind(-1)
        state = (
            _masked(feedback.q, q_mask), _masked(feedback.qd / 5, qd_mask),
            _masked(target.nominal, n_mask), _masked(target.nominal_velocity / 5, nd_mask),
            _masked(target.nominal - feedback.q, n_mask & q_mask),
            _masked((target.nominal_velocity - feedback.qd) / 5, nd_mask & qd_mask),
        )
        prev = tuple(_masked(value, memory.previous_valid) for value in memory[:3]) + (
            _masked(memory.servo, memory.servo_valid),
        )
        actual, goal = feedback.geometry, target.geometry
        a_valid, g_valid = feedback.geometry_mask, target.geometry_mask
        wrist = _pose_triplet(actual.wrist, actual.rotation, goal.wrist, goal.rotation, a_valid.wrist, g_valid.wrist)
        wrist_velocity, tip_velocity = _velocity_triple(feedback.velocity, target.velocity,
                                                       feedback.velocity_mask, target.velocity_mask)
        tips = _triple(actual.tips, goal.tips, a_valid.tips, g_valid.tips, .1)
        local = _triple(actual.local, goal.local, a_valid.local, g_valid.local, .1)
        pinch = _triple(actual.pinch, goal.pinch, a_valid.pinch, g_valid.pinch, .1)
        cross = _triple(actual.cross, goal.cross, a_valid.cross, g_valid.cross, .1)
        ap, ar = actual.relative
        gp, gr = goal.relative
        relative = _pose_triplet(ap, ar, gp, gr, a_valid.relative, g_valid.relative)
        age = torch.cat((target.age_seconds, feedback.age_seconds[:, None]), dim=-1)
        known = torch.cat((target.age_known, feedback.age_known[:, None]), dim=-1)
        quality = (
            _masked((age / DT).clamp(0, 10), known), known, target.stream_valid,
            feedback.valid, target.nominal_valid, a_valid.wrist, g_valid.wrist,
            feedback.velocity_mask.wrist, target.velocity_mask.wrist,
            a_valid.tips, g_valid.tips, feedback.velocity_mask.tips, target.velocity_mask.tips,
            a_valid.local, g_valid.local, a_valid.pinch, g_valid.pinch, a_valid.cross, g_valid.cross,
            a_valid.relative, g_valid.relative, memory.previous_valid, memory.servo_valid,
            target.permitted, torch.ones_like(target.permitted),
        )
        frame = _join((*state, *prev, *wrist, *wrist_velocity, *tips, *tip_velocity, *local, *pinch,
                       *cross, *relative, *quality))
        if frame.shape[-1] != OBSERVATION_FRAME:
            raise RuntimeError(f"Actor frame ABI expected {OBSERVATION_FRAME}, got {frame.shape[-1]}")
        return frame[:, self.columns] if self.compact else frame


class CausalHistory:
    """Five frames oldest-first. A gap, rejoin or reset starts a fresh sequence."""

    def __init__(self, environments: int, frame_dim: int, device):
        self.frames = torch.zeros((environments, HISTORY, frame_dim), device=device)
        # Missing frames, including their final present bit, are all zero.
        self.generation = torch.zeros(environments, dtype=torch.int64, device=device)
        self.tick = torch.zeros_like(self.generation)
        self.valid = torch.zeros_like(self.tick, dtype=torch.bool)

    def reset(self, ids):
        self.frames[ids] = 0
        self.valid[ids] = False

    def append(self, frame, generation, tick, current_valid):
        same = self.valid & current_valid & (self.generation == generation) & (tick - self.tick == 20_000_000)
        pad = torch.zeros_like(self.frames[:, :-1])
        previous = torch.where(same[:, None, None], self.frames[:, 1:], pad)
        self.frames = torch.cat((previous, frame[:, None]), dim=1)
        self.generation = generation.clone()
        self.tick = tick.clone()
        self.valid = current_valid.clone()
        return self.frames.flatten(1)
