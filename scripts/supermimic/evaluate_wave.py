"""Reproduce a checkpoint in full 1357-frame Newton physical closed loop."""

import argparse
import json
from pathlib import Path

import torch
from dexgym.algorithms.supermimic.networks import GoalActor
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES
from preflight import REFERENCE_SHA256, verify
from train_wave import evaluate_full


def run(
    checkpoint: Path,
    output: Path,
    *,
    warmup_only=False,
    initial_perturb_rad=0.0,
    physics_contact_audit=True,
    reward_profile="marvin_tcp_priority",
):
    if not 0 <= initial_perturb_rad <= 0.1:
        raise ValueError("Evaluation joint perturbation must lie in [0, 0.1] rad")
    verify()
    record = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if warmup_only:
        actor = GoalActor().cuda().eval()
        actor.load_state_dict(record)
    else:
        if record["reference_sha256"] != REFERENCE_SHA256:
            raise ValueError("Checkpoint reference hash mismatch")
        actor = (
            GoalActor(
                goal_latent_tanh=record.get("training", {}).get("configuration", {}).get("goal_latent_tanh", False)
            )
            .cuda()
            .eval()
        )
        actor.load_state_dict(record["actor"])
        reward_profile = record["training"]["configuration"].get("reward_profile", reward_profile)
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_full(
        actor,
        output_dir=output,
        initial_perturb_rad=initial_perturb_rad,
        physics_contact_audit=physics_contact_audit,
        reward_profile=reward_profile,
    )
    result["checkpoint"] = str(checkpoint)
    result["reference_sha256"] = REFERENCE_SHA256
    (output / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument("--initial-perturb-rad", type=float, default=0.0)
    parser.add_argument("--no-physics-contact-audit", action="store_true")
    parser.add_argument("--reward-profile", choices=tuple(REWARD_PROFILES), default="marvin_tcp_priority")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.checkpoint,
                args.output,
                warmup_only=args.warmup_only,
                initial_perturb_rad=args.initial_perturb_rad,
                physics_contact_audit=not args.no_physics_contact_audit,
                reward_profile=args.reward_profile,
            ),
            indent=2,
        )
    )
