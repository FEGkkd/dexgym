"""Inventory the pinned CMU mirror; distinguish Git tree entries from local BVH bytes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/ubuntu/dexgym_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT / "cmu/cmu-mocap")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bvh_inventory.json")
    args = parser.parse_args()
    repo = args.repo.resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.check_output(["git", "-C", str(repo), "remote", "get-url", "origin"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "HEAD"], text=True)
    rows = []
    for line in tree.splitlines():
        mode, kind, blob, name = line.split(None, 3)
        if not name.lower().endswith(".bvh"):
            continue
        file = repo / name
        record = {
            "path": name,
            "subject": Path(name).parent.name,
            "motion_id": Path(name).stem,
            "git_blob_sha1": blob,
            "materialized": file.is_file(),
            "sha256": None,
            "frame_count": None,
            "frame_time_s": None,
        }
        if file.is_file():
            data = file.read_bytes()
            record["sha256"] = hashlib.sha256(data).hexdigest()
            for raw in data.splitlines():
                line_text = raw.strip()
                if line_text.startswith(b"Frames:"):
                    record["frame_count"] = int(line_text.split(b":", 1)[1])
                elif line_text.startswith(b"Frame Time:"):
                    record["frame_time_s"] = float(line_text.split(b":", 1)[1])
                    break
        rows.append(record)
    report = {
        "source": remote,
        "commit": commit,
        "bvh_total_in_git_tree": len(rows),
        "subject_total_in_git_tree": len({x["subject"] for x in rows}),
        "materialized_bvh_count": sum(x["materialized"] for x in rows),
        "complete_byte_inventory": all(x["materialized"] for x in rows),
        "motions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "motions"}, indent=2))


if __name__ == "__main__":
    main()
