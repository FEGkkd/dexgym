"""Frozen local-shape source split for Phase 1B evaluation."""


def fixed_training_pregrasp_index(rows, seed: int, manifest: dict) -> int:
    """Select one near-contact row only if the frozen split assigned it to training."""
    if manifest.get("split_key") != "local_shape_seed":
        raise ValueError("Fixed training Reset requires a local-shape split manifest")
    matches = [i for i, row in enumerate(rows) if row.seed == seed]
    if len(matches) != 1:
        raise ValueError("Fixed pregrasp seed is absent or duplicated in the Reset bank")
    if seed in manifest.get("evaluation_group_values_global", []):
        raise ValueError("Fixed pregrasp seed is reserved for evaluation")
    if seed not in manifest.get("train_local_shape_seeds", []):
        raise ValueError("Fixed pregrasp seed is not frozen for training")
    return matches[0]


def split_group_indices(rows, *, key: str, evaluation: bool,
                        evaluation_groups: set | frozenset | None = None) -> tuple[int, ...]:
    """Split by source preshape before selecting local perturbation rows."""
    if not key or len(rows) < 2:
        raise ValueError("A populated group key and at least two rows are required")
    groups = [row.metadata.get(key) for row in rows]
    if any(value is None for value in groups):
        raise ValueError(f"Missing local-shape group key: {key}")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("At least two independent local-shape sources are required")
    if evaluation_groups is None:
        cut = max(1, len(unique) * 3 // 4)
        chosen = set(unique[cut:] if evaluation else unique[:cut])
    else:
        chosen = set(evaluation_groups) if evaluation else set(unique) - set(evaluation_groups)
    indices = tuple(i for i, group in enumerate(groups) if group in chosen)
    if not indices:
        raise ValueError("Frozen split leaves the selected level empty")
    return indices
