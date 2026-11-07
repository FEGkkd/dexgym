"""Create an unseen reverse-time target bank for causal Newton evaluation only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def reverse_test_bank(source: Path, output: Path) -> dict:
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError("Reverse evaluation bank must be a new artifact")
    with np.load(source, allow_pickle=False) as data:
        if str(data["version"]) != "marvin_teleop_workspace_bank_v1" or int(data["fps"]) != 50:
            raise ValueError("Expected a 50 Hz workspace target bank")
        ids = np.nonzero(data["split"] == 2)[0]
        if len(ids) < 1:
            raise ValueError("Source bank has no independent test clips")
        names = np.array([str(data["names"][i]) + "_reverse" for i in ids])
        families = np.array([str(data["family"][i]) + "_reverse" for i in ids])
        difficulty = data["difficulty"][ids].copy()
        joints = data["joint_names_arm"].copy()
        witness = np.flip(data["q_reset_witness"][ids], axis=1).copy()
        position = np.flip(data["tcp_pos"][ids], axis=1).copy()
        quaternion = np.flip(data["tcp_quat_xyzw"][ids], axis=1).copy()
        seed = int(data["seed"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, version="marvin_teleop_workspace_bank_v1",
                        transform="test_split_reverse_time_v1", seed=seed,
                        fps=50, joint_names_arm=joints, names=names,
                        split=np.full(len(ids), 2, dtype=np.int8),
                        family=families, difficulty=difficulty,
                        q_reset_witness=witness, tcp_pos=position,
                        tcp_quat_xyzw=quaternion)
    return {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "clips": len(ids), "transform": "test_split_reverse_time_v1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reverse_test_bank(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
