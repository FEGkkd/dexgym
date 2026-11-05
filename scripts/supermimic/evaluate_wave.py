"""Reproduce a checkpoint in full 1357-frame Newton physical closed loop."""

import argparse
import json
from pathlib import Path

import torch
from dexgym.algorithms.supermimic.networks import GoalActor
from preflight import REFERENCE_SHA256, verify
from train_wave import evaluate_full


def run(checkpoint: Path, output: Path, *, warmup_only=False):
    verify()
    actor = GoalActor().cuda().eval()
    record = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if warmup_only:
        actor.load_state_dict(record)
    else:
        if record["reference_sha256"] != REFERENCE_SHA256:
            raise ValueError("Checkpoint reference hash mismatch")
        actor.load_state_dict(record["actor"])
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_full(actor, output_dir=output)
    result["checkpoint"] = str(checkpoint)
    result["reference_sha256"] = REFERENCE_SHA256
    (output / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.output, warmup_only=args.warmup_only), indent=2))
