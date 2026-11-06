"""Executable dynamics signature required before reusing learned model weights."""

from dataclasses import asdict
import hashlib
import json


def world_model_dynamics_signature(robot_adapter, network_spec) -> tuple[dict, str]:
    contract = getattr(robot_adapter, "checkpoint_contract", None)
    if not callable(contract):
        raise ValueError("Shared World Model needs Robot Adapter checkpoint_contract")
    robot_contract = contract()
    if not isinstance(robot_contract, dict):
        raise ValueError("Robot dynamics contract must be a mapping")
    spec = robot_adapter.spec
    if spec.context_dim and (not robot_contract.get("context_schema")
                             or not robot_contract.get("context_model_range")):
        raise ValueError("Shared World Model needs context schema and declared model range")
    manifest = {
        "signature_version": "world_model_dynamics_v1",
        "robot_spec": asdict(spec),
        "physical_controller_and_integration": robot_contract,
        "world_model": {
            "hidden": network_spec.world_hidden,
            "normalization": network_spec.normalize,
            "normalizer_version": network_spec.normalizer_version,
            "normalizer_min_std": network_spec.normalizer_min_std,
            "normalizer_max_count": network_spec.normalizer_max_count,
            "delta_integration": "robot_adapter_delta_v1",
            "weight_contract": "same_model_instance_v1",
        },
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return manifest, hashlib.sha256(encoded.encode()).hexdigest()


def first_dynamics_difference(saved, current, prefix="dynamics"):
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(saved.keys() | current.keys()):
            if key not in saved or key not in current:
                return f"{prefix}.{key}: missing field"
            found = first_dynamics_difference(saved[key], current[key], f"{prefix}.{key}")
            if found:
                return found
        return None
    if saved != current:
        return f"{prefix}: saved={saved!r}, current={current!r}"
    return None
