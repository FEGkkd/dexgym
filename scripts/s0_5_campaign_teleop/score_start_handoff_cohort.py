"""Run bounded independent wrist scoring for all roots in a completed native receipt."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


SCRIPTS = Path(__file__).resolve().parent


def score(args, root):
    root_id = root["root"]
    identifier = root_id.split(":")[-1]
    attempts = root["attempts"]
    latest = attempts[-1] if attempts else {}
    command = [sys.executable, "-B", str(SCRIPTS / "compare_start_handoff_layers.py"), "--root", root_id,
               "--out", str(args.layers / identifier)]
    for key in ("cohort", "frames", "direct_bank", "dcr_comparison", "model", "docs"):
        command.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    if latest.get("trace") is not None:
        command.extend(("--trace", latest["trace"]))
    if latest.get("anchor_sha256") is not None:
        command.extend(("--anchor", str(Path(latest["directory"]) / "anchor.json")))
    with (args.layers / f"{identifier}.log").open("x", encoding="utf-8") as output:
        completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    print(root_id, root["result"], completed.returncode, flush=True)
    return root_id, completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cohort", "run", "frames", "direct-bank", "dcr-comparison", "model", "docs", "layers", "home-proof", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    args = parser.parse_args()
    rows = json.loads(args.run.read_bytes())["roots"]
    if len(rows) != 32 or len({row["root"] for row in rows}) != 32 or not all(row["selected"] for row in rows):
        raise ValueError("Expected all 32 planned roots in the native run receipt")
    args.layers.mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        status = dict(pool.map(lambda row: score(args, row), rows))
    with (args.layers / "scoring_status.json").open("x", encoding="utf-8") as output:
        json.dump(status, output, indent=2)
        output.write("\n")
    if any(status.values()):
        raise SystemExit(1)
    command = [sys.executable, "-B", str(SCRIPTS / "seal_start_handoff_layers.py")]
    for key in ("cohort", "run", "layers", "home_proof", "out"):
        command.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
