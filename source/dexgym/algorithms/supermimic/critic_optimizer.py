"""Fixed-target TD(lambda) critic fitting with configurable minibatch epochs."""

import torch
from .optimization import safe_gradient_step


def optimize_critic(critic, optimizer, features: torch.Tensor, targets: torch.Tensor,
                    *, epochs: int = 1, minibatches: int = 1, grad_norm_clip: float = 1.0,
                    grad_value_clip: float | None = None) -> dict:
    if features.ndim != 2 or targets.shape != (len(features),) or len(features) < 1:
        raise ValueError("Critic features and fixed targets must have aligned nonempty batches")
    if epochs < 1 or minibatches < 1 or minibatches > len(features):
        raise ValueError("Critic epochs/minibatches must be positive and fit the batch")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Critic input features contain nonfinite data")
    parameters = tuple(critic.parameters())
    if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise FloatingPointError("Critic parameters already contain nonfinite values")
    updates, skipped, losses, td_errors, value_means, gradient_norms = 0, 0, [], [], [], []
    for _ in range(epochs):
        for index in torch.randperm(len(features), device=features.device).tensor_split(minibatches):
            prediction = critic.forward_features(features[index])
            target = targets[index]
            if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
                skipped += 1
                continue
            error = prediction - target
            loss = error.square().mean()
            if not bool(torch.isfinite(loss)):
                skipped += 1
                continue
            result = safe_gradient_step(loss, parameters, optimizer,
                                        grad_norm_clip=grad_norm_clip,
                                        grad_value_clip=grad_value_clip)
            if not result["updated"]:
                skipped += 1
                continue
            updates += 1
            losses.append(float(loss.detach()))
            td_errors.append(float(error.detach().abs().mean()))
            value_means.append(float(prediction.detach().mean()))
            gradient_norms.append(result["grad_norm"])
    return {
        "updates": updates, "skipped_nonfinite": skipped,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "td_error_abs_mean": sum(td_errors) / len(td_errors) if td_errors else None,
        "predicted_value_mean": sum(value_means) / len(value_means) if value_means else None,
        "bootstrap_value_mean": None,
        "target_abs_max": float(targets[torch.isfinite(targets)].abs().max())
        if bool(torch.isfinite(targets).any()) else None,
        "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms) if gradient_norms else None,
    }
