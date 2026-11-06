"""Guard every learned update against nonfinite losses, gradients and parameters."""

import math

import torch


def safe_gradient_step(loss, parameters, optimizer, *, grad_norm_clip=1.0,
                       grad_value_clip=None):
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss).all()):
        return {"updated": False, "reason": "nonfinite_loss", "grad_norm": math.nan}
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
        optimizer.zero_grad(set_to_none=True)
        return {"updated": False, "reason": "nonfinite_parameter", "grad_norm": float(norm)}
    return {"updated": True, "reason": None, "grad_norm": float(norm)}
