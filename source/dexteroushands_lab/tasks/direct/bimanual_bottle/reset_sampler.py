"""Fixed bank split by sample index for reproducible evaluation.

The four stable samples share one relative grasp family. Their held-out row is
therefore same-family evaluation, not evidence of cross-family generalization.
"""


def split_bank_indices(stable_count: int, pregrasp_count: int,
                       *, evaluation: bool) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if stable_count < 2 or pregrasp_count < 2:
        raise ValueError("Both reset banks need at least two samples for a held-out split")
    stable_cut = max(1, stable_count * 3 // 4)
    if evaluation:
        return tuple(range(stable_cut, stable_count)), split_pregrasp_indices(
            pregrasp_count, evaluation=True)
    return tuple(range(stable_cut)), split_pregrasp_indices(
        pregrasp_count, evaluation=False)


def split_pregrasp_indices(pregrasp_count: int, *, evaluation: bool) -> tuple[int, ...]:
    """Keep the pregrasp split when an external stable evaluation bank has one row."""
    if pregrasp_count < 2:
        raise ValueError("Pregrasp reset bank needs at least two samples")
    cut = max(1, pregrasp_count * 3 // 4)
    return tuple(range(cut, pregrasp_count)) if evaluation else tuple(range(cut))
