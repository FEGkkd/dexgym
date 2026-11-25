"""Success-triggered, same-local-frame reference switching (no robot reset)."""

from dataclasses import asdict, dataclass
from math import cos, isfinite, radians
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class SwitchConfig:
    probability: float = .10
    wrist_m: float = .013
    rotation_deg: float = 6.5
    hand_m: float = .0195
    fingertip_m: float = .013
    recovery_steps: int = 50
    min_future_steps: int = 16

    def __post_init__(self):
        if not isfinite(self.probability) or not 0. <= self.probability <= 1.:
            raise ValueError("Switch probability must be in [0, 1]")
        if any(not isfinite(x) or x <= 0 for x in
               (self.wrist_m, self.rotation_deg, self.hand_m, self.fingertip_m)):
            raise ValueError("Success tolerances must be finite and positive")
        if self.rotation_deg > 180 or self.recovery_steps < 0 or self.min_future_steps < 1:
            raise ValueError("Invalid rotation, recovery, or future-frame setting")

    def to_dict(self):
        return {"version": "success_same_local_frame_v1", **asdict(self),
                "sampling": "uniform_other_train_clip_with_sufficient_length",
                "clock": "continuous_50hz_first_switched_target_held_one_interval",
                "success": "both_wrists_and_all_hand_landmarks_and_all_tips"}


class ReferenceAdvance(NamedTuple):
    frame: torch.Tensor
    bridge: torch.Tensor
    previous_frame: torch.Tensor
    previous_bridge: torch.Tensor
    last_success: torch.Tensor
    recovery: torch.Tensor
    success: torch.Tensor
    attempted: torch.Tensor
    switched: torch.Tensor
    unavailable: torch.Tensor


class ReferenceSwitch:
    def __init__(self, bank, config=None, *, causal=False):
        self.bank = bank
        self.config = config or SwitchConfig()
        self.causal = causal
        if self.config.probability and bank.split != "train":
            raise ValueError("Reference switching must use the training split")
        self.sorted_lengths, self.order = torch.sort(bank.clip_lengths)
        self.rank = torch.empty_like(self.order)
        self.rank[self.order] = torch.arange(len(self.order), device=self.order.device)

    def reached(self, measured, goal):
        c = self.config
        wrist = (measured.wrist - goal.wrist).square().sum(-1)
        rotation = .5 * (measured.rotation - goal.rotation).square().sum((-1, -2))
        hand = (measured.hand - goal.hand).square().sum(-1)
        return ((wrist <= c.wrist_m**2).all(-1)
                & (rotation <= 2. * (1. - cos(radians(c.rotation_deg)))).all(-1)
                & (hand <= c.hand_m**2).all(dim=(-1, -2, -3))
                & (hand[..., -1] <= c.fingertip_m**2).all(dim=(-1, -2)))

    def sample_other(self, frame, generator):
        """Uniform among length-eligible other clips, without a dense env x clip matrix."""
        bank = self.bank
        current = bank.clip_id[frame]
        local = frame - bank.clip_starts[current]
        first = torch.searchsorted(self.sorted_lengths, local + self.config.min_future_steps + 1)
        old_rank = self.rank[current]
        exclude = old_rank >= first
        count = len(self.order) - first - exclude.long()
        offset = (torch.rand(frame.shape, device=frame.device, generator=generator)
                  * count.clamp_min(1)).long()
        rank = first + offset
        rank = rank + (exclude & (rank >= old_rank)).long()
        other = self.order[rank.clamp(max=len(self.order) - 1)]
        candidate = bank.clip_starts[other] + local
        available = count > 0
        candidate = torch.where(available, candidate, frame)
        return candidate, available

    @torch.no_grad()
    def advance(self, measured, frame, bridge, last_success, recovery, generator, *, eligible=None, hold=None,
                recorded=None):
        bank, c = self.bank, self.config
        advancing = bridge == 0
        advancing_clock = advancing if hold is None else advancing & ~hold
        next_frame = torch.minimum(frame + advancing_clock.long(), bank.end[frame])
        if recorded is not None:
            next_frame = torch.where(advancing_clock & recorded, bank.recorded_next[frame], next_frame)
        next_bridge = (bridge - 1).clamp_min(0)
        zeros = torch.zeros_like(frame, dtype=torch.bool)
        success = attempted = switched = unavailable = zeros
        if c.probability:
            success = self.reached(measured, bank.at(frame)) & advancing & (frame != last_success)
            success &= frame < bank.end[frame]
            if eligible is not None:
                success &= eligible
            # One Bernoulli trial per newly reached target. No cooldown changes its probability.
            attempted = success & (torch.rand(frame.shape, device=frame.device, generator=generator) < c.probability)
            ids = attempted.nonzero().flatten()
            destination = frame.clone()
            if len(ids):
                candidate, available = self.sample_other(frame[ids], generator)
                destination[ids] = candidate
                switched = zeros.clone()
                switched[ids] = available
                unavailable = attempted & ~switched
            next_frame = torch.where(switched, destination, next_frame)
            last_success = torch.where(success, frame, last_success)
        # A fresh goal is visible before its first scored interval. With the causal
        # input convention this also sets the received target to B[k], not A[k].
        received = torch.where(switched, next_frame, frame) if self.causal else frame
        previous_bridge = torch.where(switched, 0, bridge) if self.causal else bridge
        recovery = torch.where(success, 0, (recovery - 1).clamp_min(0))
        recovery = torch.where(switched, c.recovery_steps, recovery)
        return ReferenceAdvance(next_frame, next_bridge, received, previous_bridge,
                                last_success, recovery, success, attempted, switched, unavailable)


def add_switch_arguments(parser):
    defaults = SwitchConfig()
    parser.add_argument("--switch-probability", type=float, default=.10)
    parser.add_argument("--switch-wrist-m", type=float, default=defaults.wrist_m)
    parser.add_argument("--switch-rotation-deg", type=float, default=defaults.rotation_deg)
    parser.add_argument("--switch-hand-m", type=float, default=defaults.hand_m)
    parser.add_argument("--switch-fingertip-m", type=float, default=defaults.fingertip_m)
    parser.add_argument("--switch-recovery-steps", type=int, default=50)
    parser.add_argument("--switch-min-future-steps", type=int, default=16)


def switch_config(args):
    return SwitchConfig(**{name: getattr(args, "switch_" + name) for name in SwitchConfig.__dataclass_fields__})
