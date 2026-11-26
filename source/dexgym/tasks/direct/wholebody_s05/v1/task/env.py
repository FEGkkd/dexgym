"""Causal, simulation-only 54-axis residual task. The caller owns reference scheduling."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from . import DT, JOINTS, SUBSTEPS
from .geometry import Geometry, GeometryVelocity, ResidualKinematics, difference
from .observation import (CausalHistory, CommandMemory, Feedback, ObservationBuilder, TargetFrame,
                          geometry_mask, velocity_mask)
from .physics import context
from .reward import GeometryReward


AXES = {"B0": 0, "B14": 14, "B54": 54}


@dataclass(frozen=True)
class Transition:
    x: Tensor
    x1_pre_reset: Tensor
    s: Tensor
    s_next: Tensor
    action: Tensor
    requested: Tensor
    c: Tensor
    applied: Tensor
    ctrl: Tensor
    torque: Tensor
    u: Tensor
    sim_context: Tensor
    goal: TargetFrame
    obs: Tensor
    reward: Tensor
    reward_terms: dict[str, Tensor]
    costs: dict[str, Tensor]
    contact: Tensor
    contact_depth_m: Tensor
    minimum_gap_m: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor
    full_step: Tensor
    safety_failure: Tensor
    substeps: Tensor
    events: Tensor
    fallback: Tensor

    def engine_capture(self):
        """Capture the exact pre-reset states expected by SuperMimicEngine."""
        index = (self.substeps - 1).clamp_min(0).long()[:, None, None].expand(-1, 1, JOINTS)
        final_torque = self.torque.gather(1, index).squeeze(1)
        nan = ~torch.isfinite(self.x1_pre_reset).all(-1)
        return dict(x1=self.x1_pre_reset, c=self.c, u=self.u, contact=self.contact, torque=final_torque,
                    valid=self.valid, nan=nan, drop=self.terminated, done=self.terminated | self.truncated, terminated=self.terminated,
                    truncated=self.truncated, full_step=self.full_step, depth=self.contact_depth_m,
                    safety_failure=self.safety_failure)


class Pending(NamedTuple):
    state: Tensor
    feedback: Feedback
    target: TargetFrame
    observation: Tensor
    continuous: Tensor


class ResidualTask:
    def __init__(self, robot, backend, guard, *, mode="B54", compact=False):
        if mode not in AXES:
            raise ValueError(f"Unknown residual comparison {mode!r}")
        self.num_envs = backend.num_envs
        self.device = backend.device
        self.physics = backend
        self.guard = guard
        self.geometry = ResidualKinematics(robot).to(self.device)
        self.builder = ObservationBuilder(compact=compact).to(self.device)
        self.history = CausalHistory(self.num_envs, self.builder.frame_dim, self.device)
        self.reward = GeometryReward(clearance=guard.sweep.clearance)
        self.sim_context = context(robot, backend.profile, self.device)[None].expand(self.num_envs, -1)
        self.open_axes = torch.arange(JOINTS, device=self.device) < AXES[mode]
        self.previous_issued = backend.state()[:, :JOINTS].clone()
        self.previous_requested = self.previous_issued.clone()
        self.last_residual = torch.zeros_like(self.previous_issued)
        self.earlier_residual = torch.zeros_like(self.last_residual)
        self.previous_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.earlier_valid = self.previous_valid.clone()
        self.needs_reset = torch.ones_like(self.previous_valid)
        self.prior_geometry = self.geometry(self.previous_issued)
        self.pending = None

    def reset(self, ids: Tensor, q: Tensor, qd: Tensor):
        if self.pending is not None:
            raise RuntimeError("Consume or abort the pending command before resetting")
        valid = self.guard.positions_valid(q) & torch.isfinite(qd).all(-1)
        if not self.guard.permitted_paths(q, q, valid).all():
            raise ValueError("Simulation reset failed the named joint/contact envelope")
        if not self.guard.velocities_valid(qd).all():
            raise ValueError("Simulation reset velocity exceeded the command profile")
        self.physics.reset(ids, q, qd)
        self.history.reset(ids)
        self.previous_issued[ids] = q
        self.previous_requested[ids] = q
        self.last_residual[ids] = 0
        self.earlier_residual[ids] = 0
        self.previous_valid[ids] = False
        self.earlier_valid[ids] = False
        self.needs_reset[ids] = False
        return self.physics.state()

    def observe(self, target: TargetFrame) -> Tensor:
        if self.pending is not None:
            raise RuntimeError("A reference has already been admitted for the current control tick")
        state = self.physics.state()
        q, qd = state.split(JOINTS, dim=-1)
        q_legal = self.guard.positions_valid(q)
        qd_valid = self.guard.velocities_valid(qd)
        available = q_legal & qd_valid & ~self.needs_reset
        safe_q = torch.where(q_legal[:, None], q, self.previous_issued)
        geom = self.geometry(safe_q)
        continuous = (self.history.valid & available & target.permitted & (target.generation == self.history.generation)
                      & (target.tick_ns - self.history.tick == 20_000_000))
        velocities = difference(geom, self.prior_geometry, DT)
        feedback = Feedback(q, qd, geom, velocities, geometry_mask(q_legal & ~self.needs_reset),
                            velocity_mask(continuous), torch.stack((q_legal, qd_valid), dim=-1),
                            q.new_zeros(self.num_envs), available)
        servo = self.physics.servo()
        memory = CommandMemory(self.previous_requested, self.previous_issued, self.last_residual, servo,
                               self.previous_valid & continuous, torch.isfinite(servo).all(-1))
        frame = self.builder(feedback, target, memory)
        obs = self.history.append(frame, target.generation, target.tick_ns, available & target.permitted)
        self.prior_geometry = geom
        self.pending = Pending(state, feedback, target, obs, continuous)
        return obs

    def abort_observation(self):
        if self.pending is not None:
            self.history.reset(torch.arange(self.num_envs, device=self.device))
            self.pending = None

    @torch.no_grad()
    def step(self, action: Tensor, *, truncated: Tensor | None = None) -> Transition:
        if self.pending is None:
            raise RuntimeError("Admit the current causal reference before stepping")
        state, feedback, target, obs, continuous = self.pending
        servo = self.physics.servo()
        was_inactive = self.needs_reset.clone()
        feedback_valid = feedback.valid.all(-1)
        permitted = target.permitted & feedback_valid & ~was_inactive
        decision = self.guard.decide(action, target.nominal, self.previous_issued, feedback.q,
                                     servo, permitted, target.nominal_valid[:, 0], self.open_axes)
        interval = self.physics.interval(decision.issued, ~decision.hard_stop & ~was_inactive, self.guard)
        events = decision.event | interval.event
        hard = decision.hard_stop | (interval.event != 0)
        physical_finite = torch.isfinite(interval.state).all(-1)
        terminated = hard | ~physical_finite | was_inactive
        full_step = interval.substeps == SUBSTEPS
        truncated = (torch.zeros_like(hard) if truncated is None else truncated) & full_step & ~terminated
        normal = permitted & target.nominal_valid[:, 0] & ~decision.fallback & ~terminated & full_step
        q_fin = torch.where(physical_finite[:, None], interval.state[:, :JOINTS], feedback.q)
        geometry = self.geometry(torch.nan_to_num(q_fin))
        velocity = difference(geometry, feedback.geometry, DT)
        masks = geometry_mask(physical_finite)
        result = self.reward(geometry, velocity, masks, velocity_mask(full_step & physical_finite), target,
                             decision.residual, self.last_residual, self.earlier_residual, self.guard.scale,
                             self.previous_valid & continuous, self.earlier_valid & continuous, decision.projection_fraction,
                             interval.minimum_gap, normal)
        masked_applied = interval.applied.sum(1)
        u = masked_applied / interval.substeps.to(masked_applied.dtype).clamp_min(1)[:, None]
        output = Transition(
            state, interval.state, servo, interval.servo, decision.action, decision.requested, decision.issued,
            interval.applied, interval.ctrl, interval.torque, u, self.sim_context, target, obs, result.value,
            result.terms, result.penalties, interval.contact, interval.depth, interval.minimum_gap, terminated,
            truncated, normal, full_step, hard, interval.substeps, events, decision.fallback,
        )
        advanced = interval.substeps > 0
        self.previous_issued = torch.where(advanced[:, None], decision.issued, self.previous_issued)
        self.previous_requested = torch.where(normal[:, None], decision.requested, 0.)
        self.earlier_residual = self.last_residual
        self.last_residual = decision.residual
        self.earlier_valid = self.previous_valid & normal & continuous
        self.previous_valid = normal
        self.needs_reset = terminated | truncated
        self.history.valid &= normal
        self.pending = None
        return output
