"""Screen every motion by direct 30 Hz replay in stable full-contact Newton."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from dexgym.data.quality_filter import replay_contract_sha256, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Bounded smoke test; omit for the complete split")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    manifest_path = args.library_path.expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    records = [entry for entry in manifest["motions"] if entry["split"] == args.split]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        parser.error("No motions matched the selected split")
    ordered = sorted(records, key=lambda entry: (entry["frames"], entry["source"]))
    frames_by_source = {entry["source"]: entry["frames"] for entry in records}
    npz_hashes = {entry["source"]: sha256_file(manifest_path.parent / entry["file"]) for entry in records}
    batches = [ordered[i : i + args.batch_size] for i in range(0, len(ordered), args.batch_size)]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    controller_sha256 = replay_contract_sha256(repo_root)
    replay_script = repo_root / "scripts/probe_right_hand_sequence_replay.py"
    by_source = {}
    for batch_index, batch in enumerate(batches):
        batch_file = output_dir / f"batch_{batch_index:03d}.json"
        expected = {entry["source"] for entry in batch}
        cached = json.loads(batch_file.read_text()) if batch_file.is_file() else None
        if (
            cached is None
            or cached["library_path"] != str(manifest_path)
            or cached.get("manifest_sha256") != manifest_sha256
            or cached.get("controller_sha256") != controller_sha256
            or {row["source"]: row.get("input_npz_sha256") for row in cached["motions"]}
            != {source: npz_hashes[source] for source in expected}
        ):
            command = [
                sys.executable,
                str(replay_script),
                "--library-path",
                str(manifest_path),
                "--split",
                args.split,
                "--output",
                str(batch_file),
            ]
            for entry in batch:
                command.extend(("--motion", entry["source"]))
            log_file = output_dir / f"batch_{batch_index:03d}.log"
            with log_file.open("w") as stream:
                subprocess.run(command, cwd=repo_root, stdout=stream, stderr=subprocess.STDOUT, check=True)
            cached = json.loads(batch_file.read_text())
        if cached.get("controller_sha256") != controller_sha256:
            raise ValueError(f"Replay controller fingerprint changed during batch {batch_index}")
        for row in cached["motions"]:
            if row["sampled_frames"] != frames_by_source[row["source"]]:
                raise ValueError(f"Incomplete replay for {row['source']}")
            by_source[row["source"]] = row
        print(f"batch {batch_index + 1}/{len(batches)}: screened {len(by_source)}/{len(records)}", flush=True)
        report = {
            "library_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "controller_sha256": controller_sha256,
            "split": args.split,
            "physics_dt_s": 1 / 500,
            "tip_threshold_mm": 10.0,
            "complete": len(by_source) == len(records),
            "motions": [by_source[entry["source"]] for entry in records if entry["source"] in by_source],
        }
        report["passed_motions"] = sum(
            row["all_five_within_10mm_frames"] == row["sampled_frames"] for row in report["motions"]
        )
        report["failed_motions"] = len(report["motions"]) - report["passed_motions"]
        temporary = output_dir / "screen.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output_dir / "screen.json")
    print(f"pass={report['passed_motions']} fail={report['failed_motions']} total={len(records)}")
    print(f"screen={output_dir / 'screen.json'}")


if __name__ == "__main__":
    main()
