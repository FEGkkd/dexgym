"""Per-environment accounting for active Newton MJWarp contacts."""

import torch
from dexgym.algorithms.supermimic.termination import TerminationThresholds


def contact_counts(world_ids: torch.Tensor, active_count: int, num_envs: int) -> torch.Tensor:
    """Count only active MJWarp contacts by their `Contact.worldid`."""
    if world_ids.ndim != 1 or num_envs < 1:
        raise ValueError("Contact world IDs must be one-dimensional with positive num_envs")
    if active_count < 0 or active_count > world_ids.numel():
        raise ValueError("Active contact count exceeds the contact buffer")
    if active_count == 0:
        return torch.zeros(num_envs, dtype=torch.int32, device=world_ids.device)
    active_worlds = world_ids[:active_count].to(dtype=torch.long)
    if ((active_worlds < 0) | (active_worlds >= num_envs)).any():
        raise ValueError("Active contact has an invalid world ID")
    return torch.bincount(active_worlds, minlength=num_envs).to(torch.int32)


def unsafe_arm_states(
    q: torch.Tensor,
    qd: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    tcp_error: torch.Tensor,
    contacts: torch.Tensor,
    thresholds: TerminationThresholds = TerminationThresholds(),
) -> torch.Tensor:
    """Return one unsafe flag per world using the existing Marvin safety thresholds."""
    if q.ndim != 2 or q.shape[1] != 14 or qd.shape != q.shape:
        raise ValueError("Arm states must have shape [N,14]")
    if tcp_error.shape != (len(q), 2) or contacts.shape != (len(q),):
        raise ValueError("TCP errors and contact counts must match the environment count")
    bad = ~torch.isfinite(q).all(-1) | ~torch.isfinite(qd).all(-1)
    bad |= (q < lower - thresholds.joint_limit_margin_rad).any(-1) | (
        q > upper + thresholds.joint_limit_margin_rad
    ).any(-1)
    bad |= (tcp_error > thresholds.tcp_error_m).any(-1)
    bad |= (qd.abs() > thresholds.speed_rad_s).any(-1)
    return bad | contacts.gt(0)
