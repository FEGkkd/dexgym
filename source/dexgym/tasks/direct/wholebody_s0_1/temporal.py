"""Causal replay windows and function-preserving single-frame initialization."""
import torch


def replay_history(data, rows, chronological, env, frames):
    offsets = torch.arange(1 - frames, 1, device=rows.device)
    positions = (chronological[:, None] + offsets).clamp_min(0)
    sampled = rows[positions]
    previous = rows[(positions - 1).clamp_min(0)]
    # A row after a done/invalid transition belongs to a new episode. Also pad
    # at the oldest available replay row. No history can cross either boundary.
    boundary = (data['done'][previous, env[:, None]] | ~data['valid'][previous, env[:, None]])
    boundary |= positions == 0
    start = torch.where(boundary, positions, 0).cummax(dim=1).values[:, -1:]
    positions = torch.maximum(positions, start)
    return data['obs'][rows[positions], env[:, None]].float()


def expand_actor_weights(source, destination, history_frames=8):
    result = dict(source)
    key = 'net.encoder.0.weight'
    old, new = source[key], destination[key]
    if old.shape == new.shape:
        return result
    if new.shape != (old.shape[0], history_frames * old.shape[1]):
        raise ValueError(f'History warm start requires a single source frame and {history_frames} destination frames')
    expanded = torch.zeros_like(new)
    expanded[:, -old.shape[1]:] = old
    result[key] = expanded
    return result
