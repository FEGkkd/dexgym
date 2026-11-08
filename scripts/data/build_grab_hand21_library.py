"""Retarget GRAB s1-s10 with official wuji-sdk to the Wuji Hand 2.1 right hand."""

import argparse
from pathlib import Path

from dexgym.assets.wuji_hand21 import OFFICIAL_MJCF
from dexgym.data.library import build_library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hand21_right_sdk"))
    parser.add_argument("--subject", action="append", choices=[f"s{i}" for i in range(1, 11)])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    grab_root = data_root / "grab"
    subjects = args.subject or [f"s{i}" for i in range(1, 11)]
    selected = sorted(path for subject in subjects for path in (grab_root / subject).glob("*.npz"))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        selected = selected[: args.limit]
    result = build_library(
        grab_root=grab_root,
        data_root=data_root,
        mano_path=data_root / "models/mano_v1_2/MANO_RIGHT.pkl",
        sdk_path=None,
        right_xml=OFFICIAL_MJCF,
        output_dir=args.output,
        selected=selected,
        retarget_backend="official_python",
        validation_subjects=("s8", "s9", "s10"),
    )
    print(f"accepted={result['accepted']} rejected={result['rejected']} frames={result['total_frames']}")


if __name__ == "__main__":
    main()
