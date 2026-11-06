"""Export a Replay-free Core checkpoint for inference and fresh-physics resume."""

import argparse
import hashlib
import json
from pathlib import Path

from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = CoreCheckpoint.export_fresh_physics(args.checkpoint, args.output)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    print(json.dumps({"checkpoint": str(output), "sha256": digest.hexdigest(),
                      "bytes": output.stat().st_size,
                      "resume_mode": "fresh_physics"}), flush=True)


if __name__ == "__main__":
    main()
