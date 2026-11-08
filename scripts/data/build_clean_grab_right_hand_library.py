"""Copy only fully replayable right-hand motions into a clean local library."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from dexgym.data.quality_filter import (
    replay_contract_sha256,
    select_clean_records,
    sha256_file,
    verify_replay_file_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Full collision-aware library")
    parser.add_argument("--train-screen", type=Path, required=True)
    parser.add_argument("--validation-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; must not already exist")
    args = parser.parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if output_dir == input_dir or output_dir.exists():
        parser.error("--output must be a new directory distinct from --input")
    manifest_path = input_dir / "manifest.json"
    source_bytes = manifest_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    controller_sha256 = replay_contract_sha256(Path(__file__).resolve().parents[2])
    source = json.loads(source_bytes)
    replay = {}
    screen_files = {}
    for split, path in (("train", args.train_screen), ("validation", args.validation_screen)):
        file = path.expanduser().resolve()
        raw = file.read_bytes()
        report = json.loads(raw)
        if (
            report["split"] != split
            or not report["complete"]
            or Path(report["library_path"]).resolve() != manifest_path
            or report.get("manifest_sha256") != source_sha256
            or report.get("controller_sha256") != controller_sha256
            or abs(float(report["physics_dt_s"]) - 0.002) > 1e-9
            or float(report["tip_threshold_mm"]) != 10.0
        ):
            parser.error(f"Invalid or incomplete {split} replay evidence: {file}")
        entries = [entry for entry in source["motions"] if entry["split"] == split]
        rows = {row["source"]: row for row in report["motions"]}
        if len(rows) != len(entries) or set(rows) != {entry["source"] for entry in entries}:
            parser.error(f"{split} replay does not cover exactly the input motions")
        replay.update(rows)
        screen_files[split] = {"path": str(file), "sha256": hashlib.sha256(raw).hexdigest()}
    verify_replay_file_hashes(source["motions"], replay, input_dir)
    kept, excluded = select_clean_records(source["motions"], replay)
    if (
        not kept
        or not any(entry["split"] == "train" for entry in kept)
        or not any(entry["split"] == "validation" for entry in kept)
    ):
        parser.error("Screening left an empty library split")
    output_dir.mkdir(parents=True)
    for entry in kept:
        shutil.copy2(input_dir / entry["file"], output_dir / entry["file"])
        if sha256_file(output_dir / entry["file"]) != replay[entry["source"]]["input_npz_sha256"]:
            raise RuntimeError(f"Copied motion hash mismatch: {entry['source']}")
    manifest = dict(source)
    manifest.update(
        {
            "format_version": 3,
            "description": "Collision-aware Wuji right-hand motions passing every-frame full-contact Newton replay",
            "screening": {
                "criterion": (
                    "all five fingertips <10 mm on every 30 Hz frame in 1/500 s "
                    "full-contact hand_x2 Newton replay; no SDK fallback"
                ),
                "source_manifest_sha256": source_sha256,
                "controller_sha256": controller_sha256,
                "screen_files": screen_files,
            },
            "motions": kept,
            "accepted": len(kept),
            "rejected": len(excluded),
            "total_frames": sum(entry["frames"] for entry in kept),
            "errors": [{"source": key, **value} for key, value in excluded.items()],
        }
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "excluded.json").write_text(json.dumps(excluded, indent=2, sort_keys=True) + "\n")
    print(f"kept={len(kept)} excluded={len(excluded)} frames={manifest['total_frames']}")
    print(f"manifest={output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
