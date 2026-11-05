"""One policy step maps (s_t, frame_t, a_t) to (s_{t+1}, frame_t+1)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReferenceTimeIndex:
    """Frame indices with an exclusive clip length and no implicit end hold."""

    frame: torch.Tensor
    clip_length: torch.Tensor

    def __post_init__(self):
        if self.frame.dtype != torch.long or self.clip_length.dtype != torch.long:
            raise TypeError("Reference frame and clip length must be int64 tensors")
        if self.frame.shape != self.clip_length.shape:
            raise ValueError("Frame and clip length must have matching shapes")
        if bool(((self.frame < 0) | (self.frame >= self.clip_length)).any()):
            raise ValueError("Reference frame is outside its motion")

    def next_frame(self) -> torch.Tensor:
        result = self.frame + 1
        if bool((result >= self.clip_length).any()):
            raise ValueError("No next reference frame: motion ended")
        return result

    def future_frames(self, count: int) -> torch.Tensor:
        if count < 1:
            raise ValueError("Future goal count must be positive")
        return self.frame[..., None] + torch.arange(1, count + 1, device=self.frame.device)

    def full_horizon_valid(self, horizon: int) -> torch.Tensor:
        if horizon < 1:
            raise ValueError("Imagination horizon must be positive")
        return self.frame + horizon < self.clip_length


def arrival_reference_features(reference_provider, frame: torch.Tensor) -> torch.Tensor:
    """Reference for reward/termination of the state reached by an action at frame."""
    length = torch.full_like(frame, reference_provider.length)
    next_frame = ReferenceTimeIndex(frame, length).next_frame()
    return reference_provider.reference_features_at(next_frame)


@dataclass(frozen=True)
class TransitionBatch:
    """Physical next_state is captured before any automatic reset."""

    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    frame: torch.Tensor
    next_frame: torch.Tensor
    episode: torch.Tensor
    motion_id: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    mask: torch.Tensor

    def validate(self) -> None:
        leading = self.action.shape[:-1]
        for name in ("state", "next_state"):
            if getattr(self, name).shape[:-1] != leading:
                raise ValueError(f"{name} batch shape differs from action")
        for name in ("frame", "next_frame", "episode", "motion_id", "terminated", "truncated", "mask"):
            if getattr(self, name).shape != leading:
                raise ValueError(f"{name} batch shape differs from action")
        valid = self.mask.bool()
        if bool((self.next_frame[valid] != self.frame[valid] + 1).any()):
            raise ValueError("A valid transition must advance reference time exactly once")
        if not bool(torch.isfinite(self.state[valid]).all() and torch.isfinite(self.next_state[valid]).all()):
            raise ValueError("Physical transition contains nonfinite state")
