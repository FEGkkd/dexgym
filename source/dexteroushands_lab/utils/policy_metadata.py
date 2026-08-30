"""Write machine-readable policy and data-contract metadata for runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_contract import TaskDataContract


def get_task_contract(task_id: str) -> TaskDataContract | None:
    """Return the contract owned by a migrated task, if one exists."""

    if task_id == "DexterousHands-Wuji-Hand-Reorientation-Direct-v0":
        from ..tasks.direct.wuji_hand_reorientation.task_contract import (
            WUJI_REORIENTATION_CONTRACT,
        )

        return WUJI_REORIENTATION_CONTRACT
    if task_id == "DexterousHands-Shadow-Hand-Reorientation-Direct-v0":
        from ..tasks.direct.shadow_hand_reorientation.task_contract import (
            REORIENTATION_CONTRACT,
        )

        return REORIENTATION_CONTRACT
    return None


def build_policy_artifacts(
    contract: TaskDataContract,
    *,
    runtime: dict[str, Any] | None = None,
    checkpoint: str | None = None,
    exported_files: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the on-disk schema documents required by the contract."""

    runtime = dict(runtime or {})
    exported_files = list(exported_files or [])
    policy_metadata = {
        "metadata_version": "dexteroushands-policy-v1",
        "task_id": contract.task_id,
        "contract_version": contract.version,
        "asset_variant": contract.asset_variant,
        "simulator": "isaac-sim",
        "physics_backend": "physx",
        "training_framework": "isaac-lab",
        "rl_library": "rsl-rl",
        "checkpoint": checkpoint,
        "exported_files": exported_files,
        "action_dim": contract.action_dim,
        "observation_dim": contract.observation_dim,
        "critic_observation_dim": contract.critic_observation_dim,
        "frame_observation_dim": contract.frame_observation_dim,
        "observation_history_length": contract.observation_history_length,
        "quaternion_convention": contract.quaternion_convention,
        "runtime": runtime,
    }

    observation_schema = {
        "metadata_version": "dexteroushands-observation-v1",
        "task_id": contract.task_id,
        "contract_version": contract.version,
        "policy_input": {
            "dim": contract.observation_dim,
            "frame_dim": contract.frame_observation_dim,
            "history_length": contract.observation_history_length,
            "layout": "term-major, oldest-to-newest frames",
            "clip": [-5.0, 5.0],
        },
        "fields": [
            {
                "name": field.name,
                "size": field.size,
                "frame": field.frame,
                "description": field.description,
            }
            for field in contract.observation_fields
        ],
        "quaternion_convention": contract.quaternion_convention,
    }
    if contract.critic_observation_dim is not None:
        observation_schema["critic_input"] = {
            "dim": contract.critic_observation_dim,
            "actor_deployment": False,
            "purpose": "privileged asymmetric critic input used during training only",
            "layout": "207 clean shared history + 206 privileged state terms",
            "placeholder_fields": [
                "perturbation_direction",
                "perturbation_velocity",
                "domain_randomization_ratios",
            ],
        }

    action_schema = {
        "metadata_version": "dexteroushands-action-v1",
        "task_id": contract.task_id,
        "contract_version": contract.version,
        "dim": contract.action_dim,
        "policy_range": [-1.0, 1.0],
        "joint_order": list(contract.joint_order),
        "mode": contract.action_mode,
        "runtime_mapping": {
            "clip_policy_action": [-1.0, 1.0],
            "target_scale": runtime.get("action_scale"),
            "ema_alpha": runtime.get("ema_alpha"),
            "warmup_time_s": runtime.get("warmup_time_s"),
        },
    }

    normalization = {
        "metadata_version": "dexteroushands-normalization-v1",
        "task_id": contract.task_id,
        "contract_version": contract.version,
        "joint_position": {
            "formula": "(q - 0.5*(lower+upper)) / (0.5*(upper-lower))",
            "clip": [-1.0, 1.0],
        },
        "qpos_error": {
            "formula": "normalized_joint_position - normalized_processed_target",
            "clip": [-2.0, 2.0],
        },
        "policy_observation_clip": [-5.0, 5.0],
        "policy_normalizer": "RSL-RL empirical observation normalizer embedded in export",
        "quaternion_convention": contract.quaternion_convention,
    }

    return {
        "policy_metadata.yaml": policy_metadata,
        "observation_schema.yaml": observation_schema,
        "action_schema.yaml": action_schema,
        "normalization.yaml": normalization,
    }


def runtime_from_env_cfg(env_cfg: Any) -> dict[str, Any]:
    """Extract stable timing/controller values without serializing the config."""

    sim = getattr(env_cfg, "sim", None)
    dt = getattr(sim, "dt", None)
    decimation = getattr(env_cfg, "decimation", None)
    return {
        "physics_dt": dt,
        "decimation": decimation,
        "control_dt": dt * decimation if dt is not None and decimation is not None else None,
        "episode_length_s": getattr(env_cfg, "episode_length_s", None),
        "action_scale": getattr(env_cfg, "action_scale", None),
        "ema_alpha": getattr(env_cfg, "ema_alpha", None),
        "warmup_time_s": getattr(env_cfg, "warmup_time_s", None),
    }


def write_policy_artifacts(
    log_dir: str | Path,
    task_id: str,
    *,
    env_cfg: Any | None = None,
    checkpoint: str | None = None,
    exported_files: list[str] | None = None,
) -> list[Path]:
    """Write contract artifacts for a known migrated task and return paths."""

    contract = get_task_contract(task_id)
    if contract is None:
        return []

    from isaaclab.utils.io import dump_yaml

    runtime = runtime_from_env_cfg(env_cfg) if env_cfg is not None else {}
    artifacts = build_policy_artifacts(
        contract,
        runtime=runtime,
        checkpoint=checkpoint,
        exported_files=exported_files,
    )
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, data in artifacts.items():
        path = output_dir / filename
        dump_yaml(str(path), data)
        paths.append(path)
    return paths
