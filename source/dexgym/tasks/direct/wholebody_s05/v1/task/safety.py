"""Simulation command projection and conservative swept-pose proof."""

from enum import IntFlag
from typing import NamedTuple

import numpy as np
import torch

from ...p02.model import COUPLING_LIMIT, COUPLING_WEIGHTS
from . import DT, JOINTS
from .native_sweep import GAP_REPORT_CAP_M, NativeSweep


class Stop(IntFlag):
    INVALID_INPUT = 1
    ILLEGAL_NOMINAL = 2
    INVALID_ACTION = 4
    PATH_REJECTED = 8
    HOLD_INFEASIBLE = 16
    BAD_FEEDBACK = 32
    PHYSICAL_CONSTRAINT = 64
    OVERFLOW = 128
    TIMING = 256


class Decision(NamedTuple):
    action: torch.Tensor
    requested: torch.Tensor
    issued: torch.Tensor
    residual: torch.Tensor
    event: torch.Tensor
    hard_stop: torch.Tensor
    fallback: torch.Tensor
    projection_fraction: torch.Tensor


class SafetyProjection:
    """Independent simulation guard. All commands and fallbacks share the same proof."""

    def __init__(self, robot, *, device, max_step, reserve_rad=0., sweep=None):
        self.robot = robot
        self.bounds = torch.as_tensor(robot.bounds, dtype=torch.float32, device=device)
        self.max_step = torch.as_tensor(max_step, dtype=torch.float32, device=device)
        self.scale = torch.deg2rad(torch.tensor([8.] * 14 + [5.] * 40, device=device))
        self.coupling = self.bounds.new_tensor(COUPLING_WEIGHTS)
        self.coupling_limit = float(COUPLING_LIMIT - reserve_rad)
        self.sweep = NativeSweep(robot) if sweep is None else sweep
        self.cache_anchor = None
        self.cache_lower = None
        self.influence = self.bounds.new_tensor(self.sweep.weights) if isinstance(self.sweep, NativeSweep) else None
        if self.max_step.shape != (JOINTS,) or not torch.isfinite(self.max_step).all() or (self.max_step <= 0).any():
            raise ValueError("A finite positive command-step limit is required for each axis")
        if not 0 <= reserve_rad < COUPLING_LIMIT:
            raise ValueError("Invalid J6/J7 guard margin")

    def positions_valid(self, value, *, tolerance=2e-6):
        low, high = self.bounds.unbind(-1)
        pair = value[:, :14].reshape(-1, 2, 7)[..., 5:]
        bounded = ((value >= low - tolerance) & (value <= high + tolerance)).all(-1)
        coupled = ((pair.abs() * self.coupling).sum(-1) <= self.coupling_limit + tolerance).all(-1)
        return torch.isfinite(value).all(-1) & bounded & coupled

    def velocities_valid(self, value, *, tolerance=2e-5):
        return torch.isfinite(value).all(-1) & (value.abs() <= self.max_step / DT + tolerance).all(-1)

    def minimum_gaps(self, pose, candidate):
        gaps = pose.new_zeros(len(pose))
        if self.influence is not None and self.cache_anchor is not None and self.cache_anchor.shape == pose.shape:
            # A proved lower bound above the report cap has the same result as
            # a full capped narrow-phase query.
            clear = candidate & (self._cached_clearance(pose, pose) >= GAP_REPORT_CAP_M).all(-1)
            gaps[clear] = GAP_REPORT_CAP_M
            candidate = candidate & ~clear
        queried = candidate.nonzero().flatten()
        if queried.numel() > 0:
            rows = pose[queried].detach().cpu().numpy()
            unique, inverse = np.unique(rows, axis=0, return_inverse=True)
            exact = np.asarray([self.sweep.pose_gap(row) for row in unique])
            gaps[queried] = torch.as_tensor(exact[inverse], device=pose.device, dtype=pose.dtype)
        return gaps

    def project(self, request, previous):
        low = torch.maximum(self.bounds[:, 0], previous - self.max_step)
        high = torch.minimum(self.bounds[:, 1], previous + self.max_step)
        projected = torch.clamp(request, min=low, max=high)
        pair = projected[:, :14].reshape(-1, 2, 7)[..., 5:]
        def paired(bounds):
            return bounds[:, :14].reshape(-1, 2, 7)[..., 5:]
        anchor = torch.clamp(torch.zeros_like(pair), min=paired(low), max=paired(high))
        floor = (anchor.abs() * self.coupling).sum(-1, keepdim=True)
        excess = (pair.abs() * self.coupling).sum(-1, keepdim=True) - floor
        ratio = ((self.coupling_limit - floor) / excess.clamp_min(1e-12)).clamp(0, 1)
        move = torch.lerp(anchor, pair, ratio)
        arms = torch.cat((projected[:, :14].reshape(-1, 2, 7)[..., :5], move), dim=-1).reshape(-1, 14)
        return torch.cat((arms, projected[:, 14:]), dim=-1)

    def _literal_paths(self, start, end, candidate):
        candidates = candidate.nonzero().flatten()
        if candidates.numel() == 0:
            return candidate
        starts = start[candidates].detach().cpu().numpy()
        ends = end[candidates].detach().cpu().numpy()
        found = [self.sweep.permits(a, b) for a, b in zip(starts, ends, strict=True)]
        result = torch.zeros_like(candidate)
        result[candidates] = torch.as_tensor(found, device=start.device)
        return result

    def _refresh_cache(self, pose, ids):
        rows = pose[ids].detach().cpu().numpy()
        unique, labels = np.unique(rows, axis=0, return_inverse=True)
        q_cache = self.cache_anchor
        d_cache = self.cache_lower
        q_cache[ids] = pose[ids]
        results = np.stack([self.sweep.pair_distances(row) for row in unique])
        d_cache[ids] = torch.as_tensor(results[labels], device=q_cache.device, dtype=q_cache.dtype) - 2e-6

    def _cached_clearance(self, start, end):
        delta = torch.maximum((start - self.cache_anchor).abs(), (end - self.cache_anchor).abs())
        # The extra relative allowance covers fp32/TF32 rounding in this positive matmul.
        # Overflow or any absent pair produces no proof and takes the exact path.
        with torch.autocast(device_type=start.device.type, enabled=False):
            variation = 1.005 * (delta.float() @ self.influence.T) + 2e-6
        return self.cache_lower - variation

    def permitted_paths(self, start, end, candidate):
        if self.influence is None or start.device.type != "cuda":
            return self._literal_paths(start, end, candidate)
        if self.cache_anchor is None or self.cache_anchor.shape != start.shape:
            self.cache_anchor = torch.zeros_like(start)
            self.cache_lower = start.new_full((len(start), self.influence.shape[0]), -torch.inf)
        is_proven = candidate & (self._cached_clearance(start, end) >= self.sweep.clearance).all(-1)
        unresolved = (candidate & ~is_proven).nonzero().flatten()
        if unresolved.numel() == 0:
            return is_proven
        midpoint = .5 * (start + end)
        self._refresh_cache(midpoint, unresolved)
        is_proven = candidate & (self._cached_clearance(start, end) >= self.sweep.clearance).all(-1)
        return is_proven | self._literal_paths(start, end, candidate & ~is_proven)

    def decide(self, action, nominal, previous, measured, servo, permitted, nominal_legal, open_axes):
        if action.shape != nominal.shape or action.shape[-1] != JOINTS:
            raise ValueError("Expected action and nominal command shaped B x 54")
        action = torch.where(open_axes, action, 0.)
        actor_valid = (torch.isfinite(action) & (action.abs() <= 1)).all(-1)
        state_valid = self.positions_valid(previous) & self.positions_valid(measured) & self.positions_valid(servo)
        nominal_legal = nominal_legal & self.positions_valid(nominal)
        requested = nominal + action * self.scale
        raw_valid = actor_valid & state_valid & permitted & nominal_legal
        proposal = self.project(torch.where(raw_valid[:, None], requested, previous), previous)
        candidate = raw_valid & self.positions_valid(proposal)
        safe_path = self.permitted_paths(servo, proposal, candidate)
        safe_path = self.permitted_paths(measured, proposal, safe_path)
        held = ~safe_path
        safe_hold = self.permitted_paths(servo, previous, held & state_valid)
        safe_hold = self.permitted_paths(measured, previous, safe_hold)
        hard_stop = held & ~safe_hold
        issued = torch.where(safe_path[:, None], proposal, previous)
        residual = torch.where((safe_path & nominal_legal)[:, None], issued - nominal, 0)
        event = ((~permitted).int() * Stop.INVALID_INPUT
                 | (~nominal_legal).int() * Stop.ILLEGAL_NOMINAL
                 | (~actor_valid).int() * Stop.INVALID_ACTION
                 | (~state_valid).int() * Stop.BAD_FEEDBACK
                 | (candidate & held).int() * Stop.PATH_REJECTED
                 | hard_stop.int() * Stop.HOLD_INFEASIBLE)
        fraction = torch.where(
            safe_path, ((proposal - torch.where(raw_valid[:, None], requested, previous)).abs() / self.scale)
            .amax(-1).clamp(0, 1), 1.)
        return Decision(action, requested, issued, residual, event, hard_stop, held, fraction)
