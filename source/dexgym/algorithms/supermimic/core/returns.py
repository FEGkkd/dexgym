"""SuperMimic survival-weighted Actor objective and TD(lambda) Critic targets."""

import torch


def survival_weights(termination_probabilities: torch.Tensor) -> torch.Tensor:
    if termination_probabilities.ndim != 2 or termination_probabilities.shape[0] < 1:
        raise ValueError("Termination probabilities must have shape [H,B]")
    first = torch.ones_like(termination_probabilities[:1])
    return torch.cumprod(torch.cat((first, 1 - termination_probabilities), dim=0), dim=0)


def discounted_imagined_return(rewards: torch.Tensor, termination_probabilities: torch.Tensor,
                               terminal_value: torch.Tensor, *, gamma: float = 0.99) -> torch.Tensor:
    if rewards.ndim != 2 or termination_probabilities.shape != rewards.shape or terminal_value.shape != rewards.shape[1:]:
        raise ValueError("Imagined reward, termination and bootstrap shapes differ")
    if not 0 <= gamma < 1:
        raise ValueError("Gamma must be within [0,1)")
    horizon = len(rewards)
    survival = survival_weights(termination_probabilities)
    discount = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    return ((discount[:, None] * survival[:-1] * rewards).sum(0)
            + gamma**horizon * survival[-1] * terminal_value)


def lambda_returns(rewards: torch.Tensor, values: torch.Tensor,
                   termination_probabilities: torch.Tensor, *,
                   gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    if rewards.ndim != 2 or values.shape != (len(rewards) + 1, rewards.shape[1]):
        raise ValueError("TD(lambda) values must have shape [H+1,B]")
    if termination_probabilities.shape != rewards.shape:
        raise ValueError("TD(lambda) termination shape differs")
    if not 0 <= gamma < 1 or not 0 <= lam <= 1:
        raise ValueError("Gamma and lambda are outside their valid ranges")
    running = values[-1]
    out = []
    for step in range(len(rewards) - 1, -1, -1):
        running = (rewards[step] + gamma * (1 - termination_probabilities[step])
                   * ((1 - lam) * values[step + 1] + lam * running))
        out.append(running)
    return torch.stack(out[::-1])

