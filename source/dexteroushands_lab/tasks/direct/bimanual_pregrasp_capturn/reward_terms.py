"""Unified pregrasp-to-cap-turn numerical reward."""

import torch

from .contract import bounded_turn_progress

SUCCESS_BONUS = 25.0


def accumulate_reward_terms(sums: dict[str, torch.Tensor],
                            terms: dict[str, torch.Tensor]) -> None:
    """Accumulate all profile terms, including terms added after base setup."""
    for name, value in terms.items():
        if name not in sums:
            sums[name] = torch.zeros_like(value)
        sums[name] += value


def unified_reward_terms(
    left_score: torch.Tensor, right_score: torch.Tensor,
    left_proximity: torch.Tensor, right_proximity: torch.Tensor,
    previous_left_proximity: torch.Tensor, previous_right_proximity: torch.Tensor,
    opening: torch.Tensor, previous_opening: torch.Tensor,
    stable_bilateral_support: torch.Tensor, bottle_stable: torch.Tensor,
    pitch_error_m: torch.Tensor, clipped_action: torch.Tensor,
    *, dt: float, goal_rad: float, pitch_tolerance_m: float,
    profile: str = "baseline",
) -> dict[str, torch.Tensor]:
    """Bounded dense aids plus real, supported, pitch-consistent turn progress.

    At 50 Hz for five seconds, maximal non-turn dense return is 7; the
    telescoping approach potential contributes at most 1. True success adds
    25 once. Contact pressure saturates through the supplied [0,1] scores.
    """
    n = left_score.numel()
    scalars = (right_score, left_proximity, right_proximity,
               previous_left_proximity, previous_right_proximity,
               opening, previous_opening, pitch_error_m)
    if (dt <= 0 or goal_rad <= 0 or pitch_tolerance_m <= 0
            or profile not in ("baseline", "contact_coverage")
            or any(x.shape != (n,) for x in scalars)
            or stable_bilateral_support.shape != (n,)
            or bottle_stable.shape != (n,)
            or clipped_action.shape != (n, 52)):
        raise ValueError("Unified reward input shape or scale invalid")
    left = left_score.clamp(0, 1)
    right = right_score.clamp(0, 1)
    contact = dt * .2 * (left + right)
    holding = dt * .5 * stable_bilateral_support.float()
    stability = dt * .5 * torch.sqrt(left * right) * bottle_stable.float()
    # A signed potential difference cannot be farmed by back-and-forth motion.
    approach = .5 * ((left_proximity - previous_left_proximity)
                     + (right_proximity - previous_right_proximity))
    turn_support = stable_bilateral_support & (pitch_error_m.abs() < pitch_tolerance_m)
    progress = bounded_turn_progress(opening, previous_opening, turn_support,
                                     goal_rad=goal_rad)
    action = -.01 * clipped_action.square().mean(dim=-1)
    constraint = -.1 * (pitch_error_m.abs() / pitch_tolerance_m).clamp(0, 1)
    terms = {"contact": contact, "holding": holding, "stability": stability,
             "approach": approach, "progress": progress,
             "action": action, "constraint": constraint}
    if profile == "contact_coverage":
        # One controlled reward ablation: maintain bottle stability before
        # bilateral contact, then encourage the right tripod to close. At
        # 50 Hz for 5 s these extra terms sum to at most 14, keeping the
        # no-turn dense maximum (7 + 1 + 14) below the success bonus (25).
        terms["left_stability"] = dt * left * bottle_stable.float()
        terms["right_proximity_hold"] = dt * .8 * right_proximity.clamp(0, 1)
        terms["right_contact_dense"] = dt * right
    return terms
