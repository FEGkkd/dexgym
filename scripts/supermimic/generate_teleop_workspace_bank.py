"""Generate reproducible, FK-reachable 50 Hz dual TCP target streams."""

import argparse
import json
from pathlib import Path

from dexgym.motion.teleop_workspace import generate_teleop_workspace_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--train-clips", type=int, default=256)
    parser.add_argument("--validation-clips", type=int, default=32)
    parser.add_argument("--test-clips", type=int, default=32)
    args = parser.parse_args()
    report = generate_teleop_workspace_bank(
        args.output, seed=args.seed,
        split_counts=(args.train_clips, args.validation_clips, args.test_clips))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
