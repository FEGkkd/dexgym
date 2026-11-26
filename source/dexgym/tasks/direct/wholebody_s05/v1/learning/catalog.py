"""Packed causal geometry and frozen source roles for the SuperMimic adapter."""
from dataclasses import dataclass
from functools import reduce
from operator import and_

import numpy as np
import torch

from ..task.observation import TargetFrame
from ..task.smoke import target_frame


def indexed(item, indices):
    if isinstance(item, torch.Tensor):
        return item[indices]
    return type(item)(*(indexed(sub, indices) for sub in item))


def concatenate(items):
    first = items[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(items)
    return type(first)(*(concatenate([item[i] for item in items]) for i in range(len(first))))


@dataclass
class FrameCatalog:
    """Current/past inputs for actual rollout, with integer scheduling metadata separate."""

    frames: TargetFrame
    source: torch.Tensor
    partition: torch.Tensor
    remaining: torch.Tensor
    fit_ids: tuple[int, ...]
    holdout_ids: tuple[int, ...]

    def __post_init__(self):
        eligible = self.remaining >= 10
        fit = torch.isin(self.source, self.source.new_tensor(self.fit_ids))
        held = torch.isin(self.source, self.source.new_tensor(self.holdout_ids))
        self.fit_pool = (fit & eligible).nonzero().flatten()
        self.holdout_pool = (held & eligible).nonzero().flatten()
        if not self.fit_pool.numel() or (self.holdout_ids and not self.holdout_pool.numel()):
            raise ValueError("no contiguous source-eligible starts")

    def target(self, cursor):
        return indexed(self.frames, cursor)

    @staticmethod
    def _lengths(allowed, generation, time):
        # Count contiguous legal source frames without treating a future frame as
        # an Actor feature. The count is only for offline sampling/termination.
        ends = np.r_[~allowed[:-1] | ~allowed[1:] | (generation[1:] != generation[:-1]) |
                     (np.diff(time) != 20_000_000), True]
        ids = np.flatnonzero(ends)
        positions = np.arange(len(allowed))
        return np.where(allowed, ids[np.searchsorted(ids, positions)] - positions + 1, 0)

    @classmethod
    def from_rmb(cls, provider, *, fit_roots, holdout_roots):
        fit_roots, holdout_roots = set(fit_roots), set(holdout_roots)
        eligible_roots = fit_roots | holdout_roots
        if not fit_roots or fit_roots & holdout_roots:
            raise ValueError("fitting and independent roots must be disjoint")
        if eligible_roots != {row.root for row in provider.partitions.values()}:
            raise ValueError("predeclared WM roles must cover the supplied signed roots")
        ids = {name: i for i, name in enumerate(sorted(eligible_roots))}
        stages = []
        remaining, source, partition = [], [], []
        for block, row in enumerate(provider.partitions.values()):
            data = row.arrays
            n = len(data["tick_ns"])
            frame = provider.current([(row.key, i) for i in range(n)])
            stages.append(frame)
            allowed = reduce(and_, (data[name] for name in
                                    ("domain_nominal_legal", "domain_mapped_valid", "inputs_safety_permission")))
            remaining.append(cls._lengths(allowed, data["audit_generation"], data["tick_ns"]))
            source.append(np.full(n, ids[row.root], dtype=np.int64))
            partition.append(np.full(n, block, dtype=np.int64))
        def convert(values):
            return torch.as_tensor(np.concatenate(values), device=provider.device)
        return cls(concatenate(stages), convert(source), convert(partition), convert(remaining),
                   tuple(ids[x] for x in sorted(fit_roots)), tuple(ids[x] for x in sorted(holdout_roots)))

    @classmethod
    def smoke(cls, task, frames=96):
        if frames < 11:
            raise ValueError("synthetic reference too short")
        q = task.physics.state()[:1, :54].expand(2 * frames, -1).clone()
        target = target_frame(task, q, 0)
        ticks = torch.arange(frames, device=q.device).repeat(2) * 20_000_000
        s = torch.arange(2, device=q.device).repeat_interleave(frames)
        target = target._replace(generation=s, tick_ns=ticks)
        remaining = torch.arange(frames, 0, -1, device=q.device).repeat(2)
        return cls(target, s, s, remaining, (0,), (1,))
