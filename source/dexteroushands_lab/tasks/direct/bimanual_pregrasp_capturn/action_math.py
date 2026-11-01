"""Action helpers shared by deterministic replay and counterfactuals."""

import torch


def mutable_inference_action(policy, observation: torch.Tensor) -> torch.Tensor:
    """Return a normal Tensor so replay can mask selected actor dimensions."""
    with torch.inference_mode():
        output = policy(observation)
    return output.clone()


def base_evaluation_action(policy_name: str, policy, observation: torch.Tensor) -> torch.Tensor:
    """Select a replay action; baseline labels never call a loaded Actor."""
    shape = (observation.shape[0], 52)
    if policy_name == "random":
        return torch.randn(shape, device=observation.device, dtype=observation.dtype)
    if policy_name in ("zero", "no_hands") or (policy_name == "reverse" and policy is None):
        return torch.zeros(shape, device=observation.device, dtype=observation.dtype)
    if policy is None:
        raise ValueError(f"{policy_name} requires a loaded Actor")
    return mutable_inference_action(policy, observation)


def policy_wrist_speed(start_pos: torch.Tensor, end_pos: torch.Tensor,
                       start_quat_xyzw: torch.Tensor, end_quat_xyzw: torch.Tensor,
                       policy_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure actual translation and rotation over one full policy action."""
    if policy_dt <= 0:
        raise ValueError("policy_dt must be positive")
    linear = torch.linalg.vector_norm(end_pos - start_pos, dim=-1) / policy_dt
    half_cos = (start_quat_xyzw * end_quat_xyzw).sum(dim=-1).abs().clamp(0., 1.)
    half_sin = (1. - half_cos.square()).clamp_min(0.).sqrt()
    angular = 2. * torch.atan2(half_sin, half_cos) / policy_dt
    return linear, angular
