"""Run a full, hash-checked Newton playback of M4.1 or M4.2 actors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dexgym.algorithms.supermimic.checkpoint import load_legacy_m41_actor, load_legacy_m42_actor
from dexgym.tasks.marvin_supermimic.robot_adapter import DEFAULT_MJCF
from dexgym.tasks.marvin_supermimic.legacy_evaluator import evaluate_full


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=140116)
    parser.add_argument("--initial-perturb-rad", type=float, default=0.0)
    args = parser.parse_args()
    record = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if record.get("schema_version") == "m42_v1":
        metadata = record["metadata"]
        if metadata["reference_hashes"].get(str(args.reference)) != _hash(args.reference):
            raise ValueError("M4.2 checkpoint reference hash differs")
        if metadata["robot_asset_hashes"]["mjcf"] != _hash(DEFAULT_MJCF):
            raise ValueError("M4.2 checkpoint robot asset hash differs")
        config = record["config"]
        actor, _ = load_legacy_m42_actor(
            args.checkpoint, reference_path=args.reference,
            robot_asset_path=DEFAULT_MJCF, device="cuda")
        action_schema = metadata["action_schema"]
        reward_profile = config["reward_profile"]
    else:
        if _hash(args.reference) != "b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f":
            raise ValueError("M4.1 actor requires its pinned Wave reference")
        actor = load_legacy_m41_actor(args.checkpoint, device="cuda")
        action_schema = "legacy_wave_v1"
        reward_profile = "marvin_tcp_priority"
    result = evaluate_full(actor, num_envs=1, seed=args.seed, output_dir=args.output,
                           initial_perturb_rad=args.initial_perturb_rad,
                           reward_profile=reward_profile,
                           reference_path=args.reference, action_schema=action_schema,
                           initialize_hands_from_pd_replay=action_schema == "legacy_wave_v1")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
