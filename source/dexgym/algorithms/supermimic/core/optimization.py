"""Finite-gradient optimizer protection and parameter freezing for BPTT."""

from contextlib import contextmanager
import math

import torch


@contextmanager
def frozen_parameters(*modules):
    parameters = tuple(parameter for module in modules if module is not None
                       for parameter in module.parameters())
    original = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original):
            parameter.requires_grad_(requires_grad)


def safe_gradient_step(loss, parameters, optimizer, *, grad_norm_clip=1.0,
                       grad_value_clip=None):
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss).all()):
        return {"updated": False, "reason": "nonfinite_loss", "grad_norm": math.nan}
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        return {"updated": False, "reason": "nonfinite_parameter_before_step", "grad_norm": math.nan}
    loss.backward()
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
           for parameter in parameters):
        optimizer.zero_grad(set_to_none=True)
        return {"updated": False, "reason": "nonfinite_gradient", "grad_norm": math.nan}
    if grad_value_clip is not None:
        torch.nn.utils.clip_grad_value_(parameters, grad_value_clip)
    norm = torch.nn.utils.clip_grad_norm_(parameters, grad_norm_clip, error_if_nonfinite=False)
    if not bool(torch.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        return {"updated": False, "reason": "nonfinite_norm", "grad_norm": math.nan}
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        with torch.no_grad():
            for parameter, saved in zip(parameters, before):
                parameter.copy_(saved)
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        return {"updated": False, "reason": "nonfinite_parameter", "grad_norm": float(norm)}
    return {"updated": True, "reason": None, "grad_norm": float(norm)}


def optimize_critic(critic, optimizer, features, targets, *, epochs=2,
                    minibatches=4, grad_norm_clip=1.0):
    if features.ndim != 2 or targets.shape != (len(features),) or len(features) < 1:
        raise ValueError("Critic batch features and fixed targets differ")
    if epochs < 1 or not 1 <= minibatches <= len(features):
        raise ValueError("Critic update schedule is invalid")
    if not bool(torch.isfinite(features).all() and torch.isfinite(targets).all()):
        raise ValueError("Critic batch contains nonfinite data")
    updates, skipped, losses, td_errors = 0, 0, [], []
    for _ in range(epochs):
        for index in torch.randperm(len(features), device=features.device).tensor_split(minibatches):
            prediction = critic(features[index])
            error = prediction - targets[index]
            loss = error.square().mean()
            result = safe_gradient_step(loss, critic.parameters(), optimizer,
                                        grad_norm_clip=grad_norm_clip)
            if result["updated"]:
                updates += 1
                losses.append(float(loss.detach()))
                td_errors.append(float(error.detach().abs().mean()))
            else:
                skipped += 1
    return {"updates": updates, "skipped_nonfinite": skipped,
            "loss_mean": sum(losses) / len(losses) if losses else None,
            "td_error_abs_mean": sum(td_errors) / len(td_errors) if td_errors else None,
            "target_abs_max": float(targets.abs().max())}

