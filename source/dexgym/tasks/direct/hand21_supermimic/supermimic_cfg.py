"""Registry-backed configuration for the standalone Hand21 SuperMimic task."""

import json
import warnings
from importlib.resources import files
from pathlib import Path

from dexgym.tasks import HAND21_SUPERMIMIC_ID


def default_experiment() -> dict:
    """Return an independent configuration, including from an installed wheel."""
    resource = files("dexgym.tasks.direct.hand21_supermimic.agents").joinpath("hand_s0.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_hand_configuration(task_id: str = HAND21_SUPERMIMIC_ID, override: Path | None = None):
    """Resolve both environment and learner through the task's Gym registration."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if task_id != HAND21_SUPERMIMIC_ID:
        raise ValueError(f"The hand SuperMimic runner does not support {task_id}")
    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    experiment = (
        json.loads(Path(override).read_text(encoding="utf-8"))
        if override is not None
        else load_cfg_from_registry(task_id, "supermimic_cfg_entry_point")
    )
    if experiment["version"] == "hand21_s0_v1":
        experiment["version"] = "hand21_s0_v2"
        for key in ("trust", "tip_mean_error_limits_m", "train_env_fraction"):
            experiment.pop(key, None)
        experiment["trainer"]["actor_update_schedule"] = "every_iteration"
        experiment["trainer"]["world_warmup_updates"] = 0
        for key in ("trust_interval", "validation_batch"):
            experiment["trainer"].pop(key, None)
        warnings.warn(
            "Migrated Hand S0 v1: removed Actor trust gate and model warmup; all environments train.",
            UserWarning,
            stacklevel=2,
        )
    if experiment["version"] != "hand21_s0_v2":
        raise ValueError("Unknown hand experiment version")
    trainer = experiment["trainer"]
    if (
        trainer.get("actor_update_schedule") != "every_iteration"
        or trainer.get("world_warmup_updates") != 0
        or trainer.get("max_imagination_horizon") != 32
    ):
        raise ValueError("Hand S0 requires every-iteration BPTT32 with zero world-model warmup")
    if any(key in experiment for key in ("trust", "tip_mean_error_limits_m", "train_env_fraction")) or any(
        key in trainer for key in ("trust_interval", "validation_batch")
    ):
        raise ValueError("Hand S0 no longer accepts trust-gating configuration")
    if experiment["num_envs"] < 1:
        raise ValueError("num_envs must be positive")
    if experiment["iterations"] < 1 or experiment["checkpoint_interval"] < 1:
        raise ValueError("Iterations and checkpoint interval must be positive")
    return cfg, experiment
