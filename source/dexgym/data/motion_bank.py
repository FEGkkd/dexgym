"""Flattened right-hand motion library for vectorized GPU goal sampling."""

import json
from pathlib import Path

import numpy as np
import torch


class MotionBank:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        device: str | torch.device,
        min_frames: int | None = None,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if min_frames is not None and min_frames < 1:
            raise ValueError("min_frames must be positive")
        manifest_path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        if not np.isclose(float(manifest["fps"]), 30.0):
            raise ValueError("Hand tracking task requires a 30-Hz motion library")
        records = [
            entry
            for entry in manifest["motions"]
            if entry["split"] == split and (min_frames is None or entry["frames"] >= min_frames)
        ]
        if not records:
            raise ValueError(f"No {split} motions in {manifest_path}")
        q_parts, tip_parts, lengths = [], [], []
        for entry in records:
            with np.load(manifest_path.parent / entry["file"], allow_pickle=False) as motion:
                q = np.asarray(motion["q_ref"], dtype=np.float32)
                tips = np.asarray(motion["tip_ref"], dtype=np.float32)
            if q.ndim != 2 or q.shape[1] != 20 or tips.shape != (len(q), 5, 3) or len(q) != entry["frames"]:
                raise ValueError(f"Invalid motion shape: {entry['file']}")
            if not np.isfinite(q).all() or not np.isfinite(tips).all():
                raise ValueError(f"Nonfinite motion: {entry['file']}")
            q_parts.append(q)
            tip_parts.append(tips)
            lengths.append(len(q))
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.offsets = torch.tensor([0, *np.cumsum(lengths[:-1])], dtype=torch.long, device=device)
        self.q = torch.as_tensor(np.concatenate(q_parts), dtype=torch.float32, device=device)
        self.tip = torch.as_tensor(np.concatenate(tip_parts), dtype=torch.float32, device=device)
        self.motion_count = len(records)
        self.records = records

    def sample_starts(self, count: int, *, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
        if stride < 1 or torch.any(self.lengths <= stride + 1):
            raise ValueError("Every motion must have enough frames for the subgoal stride")
        ids = torch.randint(self.motion_count, (count,), device=self.lengths.device)
        max_start_plus_one = self.lengths[ids] - stride
        frames = torch.floor(torch.rand(count, device=self.lengths.device) * max_start_plus_one).long()
        return ids, frames

    def joints(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        return self.q[self.offsets[motion_ids] + frames]

    def fingertips(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        return self.tip[self.offsets[motion_ids] + frames]


def advance_goal_frames(
    current: torch.Tensor, lengths: torch.Tensor, reached: torch.Tensor, *, stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance reached goals by a fixed stride; report reaching the last frame."""
    if stride < 1:
        raise ValueError("stride must be positive")
    next_frame = torch.where(reached, torch.minimum(current + stride, lengths - 1), current)
    completed = reached & (current >= lengths - 1)
    return next_frame, completed
