"""Select the bimanual hand asset without relabeling old Reset certificates."""


def configure_bimanual_hand_variant(cfg, variant: str):
    if variant not in ("beta1", "beta2"):
        raise ValueError("hand variant must be beta1 or beta2")
    cfg.hand_variant = variant
    if cfg.task_kind == "grasp_hold":
        cfg.stable_reset_probability = 0.65 if variant == "beta1" else 0.0
    elif cfg.task_kind == "cap_turn_30":
        cfg.stable_reset_probability = 1.0 if variant == "beta1" else 0.0
    elif cfg.task_kind == "pregrasp_cap_turn_30":
        cfg.stable_reset_probability = 0.0
        if variant == "beta1":
            cfg.pregrasp_bank_path = "docs/assets/phase1b_pregrasp_bank.json"
            cfg.pregrasp_split_key = "right_natural_source_seed"
            cfg.pregrasp_split_manifest_path = "docs/assets/phase1b_pregrasp_split.json"
        else:
            cfg.pregrasp_bank_path = None
            cfg.pregrasp_split_key = "local_shape_seed"
            cfg.pregrasp_split_manifest_path = "docs/assets/beta2_phase1b_pregrasp_split.json"
    return cfg
