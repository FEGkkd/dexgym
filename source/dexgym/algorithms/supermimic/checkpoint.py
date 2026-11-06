"""Strict versioned trainer checkpoints and explicit M4.1 actor compatibility."""

import os
from pathlib import Path

import torch


class CheckpointManager:
    schema_version = "m42_v1"
    required_contract = ("action_schema", "state_schema", "observation_schema", "goal_schema",
                         "reference_hashes", "robot_asset_hashes")

    def __init__(self, metadata: dict):
        missing = set(self.required_contract) - set(metadata)
        if missing:
            raise ValueError(f"Checkpoint metadata missing {sorted(missing)}")
        self.metadata = metadata.copy()

    def save(self, path: str | Path, *, models: dict, optimizers: dict, training: dict,
             config: dict, seed: int):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": self.schema_version, "metadata": self.metadata.copy(),
            "models": {name: model.state_dict() for name, model in models.items()},
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "training": training.copy(), "config": config.copy(), "seed": seed,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "replay_saved": False, "resume_requires_recollection": True,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(record, temporary)
        os.replace(temporary, path)

    def load(self, path: str | Path, *, models: dict, optimizers: dict,
             restore_rng: bool = True, map_location=None) -> dict:
        record = torch.load(path, map_location=map_location, weights_only=False)
        if record.get("schema_version") != self.schema_version:
            raise ValueError("Checkpoint schema version differs")
        for key in self.required_contract:
            if record["metadata"].get(key) != self.metadata[key]:
                raise ValueError(f"Checkpoint {key} differs")
        if set(record["models"]) != set(models) or set(record["optimizers"]) != set(optimizers):
            raise ValueError("Checkpoint model or optimizer set differs")
        for name, model in models.items():
            model.load_state_dict(record["models"][name], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(record["optimizers"][name])
        if restore_rng:
            torch.set_rng_state(record["rng_cpu"])
            if record.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(record["rng_cuda"])
        return record


def load_legacy_m41_actor(path: str | Path, *, device: str | torch.device = "cpu"):
    """Load the historical Wave actor without introducing new normalization layers."""
    from .networks import GoalActor

    record = torch.load(path, map_location=device, weights_only=False)
    if record.get("reference_sha256") != "b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f":
        raise ValueError("Legacy actor uses a different Wave reference")
    actor = GoalActor(goal_latent_tanh=False, normalization_mode="legacy").to(device)
    actor.load_state_dict(record["actor"], strict=True)
    return actor.eval()
