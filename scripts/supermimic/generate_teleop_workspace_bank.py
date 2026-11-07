"""Generate reproducible, FK-reachable 50 Hz dual TCP target streams."""

import argparse
import json
from pathlib import Path

from dexgym.motion.teleop_workspace import (
    generate_teleop_reach_curriculum_bank, generate_teleop_workspace_bank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--train-clips", type=int, default=256)
    parser.add_argument("--validation-clips", type=int, default=32)
    parser.add_argument("--test-clips", type=int, default=32)
    parser.add_argument("--profile", choices=("local_v1", "broad_v2", "reach_curriculum_v4"),
                        default="local_v1")
    parser.add_argument("--duration-s", type=int, choices=(4, 8, 12, 16, 20), default=4)
    args = parser.parse_args()
    if args.profile == "reach_curriculum_v4" and args.duration_s < 8:
        parser.error("Measured reach curriculum requires an episode of at least 8 seconds")
    generator = (generate_teleop_reach_curriculum_bank
                 if args.profile == "reach_curriculum_v4" else generate_teleop_workspace_bank)
    kwargs = {} if args.profile == "reach_curriculum_v4" else {"profile": args.profile}
    report = generator(
        args.output, seed=args.seed,
        split_counts=(args.train_clips, args.validation_clips, args.test_clips),
        frames=args.duration_s * 50 + 1, **kwargs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
