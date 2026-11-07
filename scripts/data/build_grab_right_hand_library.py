"""Convert local GRAB recordings to right-hand Wuji fingertip tracking motions."""

import argparse
from pathlib import Path

from dexgym.data.library import build_library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hand_only_right"))
    parser.add_argument("--sdk", type=Path, default=Path("~/teleop_ws/third_party/wuji_c_api/lib/libwuji_sdk_c.so"))
    parser.add_argument(
        "--right-xml",
        type=Path,
        default=Path("~/teleop_ws/src/whole_body_mujoco_driver/model/source/wujihand2_beta2/mjcf/right.xml"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N motions for a pilot")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    grab_root = data_root / "grab"
    selected = sorted(grab_root.glob("s*/*.npz"))[: args.limit] if args.limit is not None else None
    result = build_library(
        grab_root=grab_root,
        data_root=data_root,
        mano_path=data_root / "models/mano_v1_2/MANO_RIGHT.pkl",
        sdk_path=args.sdk,
        right_xml=args.right_xml,
        output_dir=args.output,
        selected=selected,
    )
    print(f"accepted={result['accepted']} rejected={result['rejected']} frames={result['total_frames']}")
    print(f"manifest={args.output.expanduser().resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
