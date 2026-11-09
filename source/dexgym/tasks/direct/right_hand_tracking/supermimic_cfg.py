"""Registry-backed configuration for the standalone Hand21 SuperMimic task."""

import json
from importlib.resources import files
from pathlib import Path

from dexgym.tasks import HAND21_SUPERMIMIC_ID


def default_experiment() -> dict:
    """Return an independent configuration, including from an installed wheel."""
    resource = files("dexgym.tasks.direct.right_hand_tracking.agents").joinpath("hand_s0.json")
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
    if experiment["version"] != "hand21_s0_v1":
        raise ValueError("Unknown hand experiment version")
    num_envs = experiment["num_envs"]
    train_count = int(num_envs * experiment["train_env_fraction"])
    if not 0 < train_count < num_envs:
        raise ValueError("Training and model-validation lanes must both be nonempty")
    if experiment["iterations"] < 1 or experiment["checkpoint_interval"] < 1:
        raise ValueError("Iterations and checkpoint interval must be positive")
    return cfg, experiment
