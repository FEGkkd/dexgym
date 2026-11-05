"""Fetch five cross-Subject BVHs from the pinned commit, verifying Git blob bytes."""

import argparse
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path("/home/ubuntu/dexgym_data")
IDS = ("18_01", "19_01", "79_06", "111_37", "113_27")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    repo = args.root / "cmu/cmu-mocap"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    out = args.root / "cmu/spot_checks"
    out.mkdir(parents=True, exist_ok=True)
    for ident in IDS:
        relative = f"data/{ident.split('_')[0].zfill(3)}/{ident}.bvh"
        expected = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"HEAD:{relative}"], text=True).strip()
        target = out / f"{ident}.bvh"
        if (
            not target.exists()
            or subprocess.check_output(["git", "hash-object", str(target)], text=True).strip() != expected
        ):
            url = f"https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/{commit}/{relative}"
            for attempt in range(5):
                try:
                    with urlopen(url, timeout=120) as response:
                        content = response.read()
                    target.write_bytes(content)
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(attempt + 1)
        actual = subprocess.check_output(["git", "hash-object", str(target)], text=True).strip()
        if actual != expected:
            raise ValueError(f"Git blob mismatch for {ident}: {actual} != {expected}")
        print(f"{ident}: {len(target.read_bytes())} bytes, Git blob {actual}")


if __name__ == "__main__":
    main()
