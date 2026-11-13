"""Assemble and verify a multi-day wholebody_s0 motion library."""

import argparse
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from .bag import digest
from .library import MotionLibrary, SPLITS
from .native_replay import checked_clip
from .pipeline import FILTER, read_json, read_npz, velocity, verify, viewer
from .prepare import write_json
from .sync import PERIOD_NS


def write_library(root, template, entries, **metadata):
    if len(entries) != len({entry["id"] for entry in entries}):
        raise ValueError("Duplicate clip ID")
    assignment = {}
    for entry in entries:
        episode, split = entry["source_episode"], entry["split"]
        if split not in SPLITS or assignment.setdefault(episode, split) != split:
            raise ValueError(f"Source recording appears in multiple splits: {episode}")
    manifest = {**template, **metadata, "clips": entries, "splits": {}}
    for split in SPLITS:
        rows = [entry for entry in entries if entry["split"] == split]
        manifest["splits"][split] = {
            "path": f"{split}.json", "episodes": len({entry["source_episode"] for entry in rows}),
            "clips": len(rows), "frames": sum(entry["frames"] for entry in rows),
        }
        write_json(root / f"{split}.json", [entry["path"] for entry in rows])
    write_json(root / "manifest.json", manifest)
    return manifest


def copy_clips(source, destination, entries):
    for entry in entries:
        relative = Path(entry["path"])
        if relative != Path("clips") / f"{entry['id']}.npz" or relative.name != f"{entry['id']}.npz":
            raise ValueError(f"Unexpected clip path: {relative}")
        target = destination / relative
        if target.exists():
            raise ValueError(f"Duplicate destination: {target}")
        shutil.copyfile(source.root / relative, target)
        if digest(target) != entry["sha256"]:
            raise ValueError(f"Invalid clip hash: {relative}")


def merge(args):
    output = args.output
    if output.exists():
        raise ValueError(f"Choose a new batch directory: {output}")
    days = sorted(set(args.dates))
    if len(days) < 2:
        raise ValueError("Provide at least two recording dates")
    folders = [args.processed_root / str(day) for day in days]
    datasets = [read_json(folder / "dataset.json") for folder in folders]
    library_paths = [folder / dataset["motion_library"] for folder, dataset in zip(folders, datasets, strict=True)]
    libraries = [MotionLibrary.open(path) for path in library_paths]
    first = libraries[0].manifest
    varying = {"clips", "splits", "tracker_frame"}
    for folder, dataset, library, day in zip(folders, datasets, libraries, days, strict=True):
        proof = read_json(folder / "verification.json")
        if (not proof["verified"] or proof["clips"] != len(library.manifest["clips"])
                or dataset["recording_date"] != str(day) or dataset["timezone"] != datasets[0]["timezone"]):
            raise ValueError(f"Daily verification or date mismatch: {folder}")
        if {k: v for k, v in library.manifest.items() if k not in varying} != {
            k: v for k, v in first.items() if k not in varying
        }:
            raise ValueError(f"Daily retargeter or library contract differs: {folder}")
    episodes = [row for dataset in datasets for row in dataset["episodes"]]
    if len(episodes) != len({row["id"] for row in episodes}):
        raise ValueError("Duplicate source episode across dates")
    daily_manifests = {str(day): {"path": str(path), "sha256": digest(path)}
                       for day, path in zip(days, library_paths, strict=True)}
    output.mkdir(parents=True)
    root = output / "motion_library"
    (root / "clips").mkdir(parents=True)
    (output / "episodes").mkdir()
    entries = []
    for day, dataset, folder, library in zip(days, datasets, folders, libraries, strict=True):
        for episode in dataset["episodes"]:
            source = (folder / "episodes" / episode["id"]).resolve(strict=True)
            (output / "episodes" / episode["id"]).symlink_to(source, target_is_directory=True)
        day_entries = [{**entry, "recording_date": str(day), "timing": "original"} for entry in library.manifest["clips"]]
        copy_clips(library, root, day_entries)
        entries.extend(day_entries)
    metadata = {"recording_dates": list(daily_manifests), "source_daily_manifests": daily_manifests,
                "tracker_frame": sorted({row["tracker_frame"] for row in episodes})}
    manifest = write_library(root, first, entries, **metadata)
    write_json(output / "dataset.json", {
        **datasets[0], **metadata, "recording_date": " / ".join(daily_manifests),
        "episodes": episodes, "splits": manifest["splits"],
    })
    print(f"Merged {len(episodes)} source recordings and {len(entries)} original-speed candidates", flush=True)


def verified_replay(folder):
    path = folder / "motion_library/manifest.json"
    library = MotionLibrary.open(path)
    report = read_json(folder / "replay_report.json")
    selection = report["summary"]["filtered_library"]
    if digest(path) != selection["sha256"]:
        raise ValueError(f"Filtered library hash differs from replay report: {folder}")
    provenance = report["summary"]["provenance"]
    if provenance != library.manifest["native_replay"] or digest(provenance["source_manifest"]) != provenance["source_manifest_sha256"]:
        raise ValueError(f"Replay input or provenance changed: {folder}")
    rows = {row["id"]: row for row in report["clips"] if row["accepted"]}
    entries = library.manifest["clips"]
    if len(rows) != len(entries) or rows.keys() != {entry["id"] for entry in entries}:
        raise ValueError(f"Filtered clips disagree with replay decisions: {folder}")
    for entry in entries:
        row = rows[entry["id"]]
        if (not row["complete"] or row["frames"] != entry["frames"] or row["split"] != entry["split"]
                or row["rejection_reasons"] or row["measured_control_intervals"] != entry["frames"] - 1):
            raise ValueError(f"Filtered clip violates native replay criteria: {entry['id']}")
    return library, {"manifest": str(path), "sha256": selection["sha256"], "report": str(folder / "replay_report.json"),
                     "overall": report["summary"]["overall"], "rejections": report["summary"]["rejections_by_reason"]}


def verify_capture(batch, library):
    """Check that each variant is traceable to an exclusive, valid source interval."""
    dataset = read_json(batch / "dataset.json")
    episodes = {row["id"]: row for row in dataset["episodes"]}
    by_source = defaultdict(list)
    for entry in library.manifest["clips"]:
        by_source[entry["source_episode"]].append(entry)
    for eid, entries in by_source.items():
        episode = episodes[eid]
        aligned_path = batch / "episodes" / eid / "aligned.npz"
        if digest(aligned_path) != episode["reference"]["aligned_sha256"]:
            raise ValueError(f"Source alignment changed: {eid}")
        aligned = read_npz(aligned_path)
        time = aligned["time_ns"]
        occupied = np.zeros(len(time), dtype=bool)
        for entry in sorted(entries, key=lambda row: row["source_start_frame"]):
            a, b = entry["source_start_frame"], entry["source_stop_frame_exclusive"]
            if (not 0 <= a < b <= len(time) or occupied[a:b].any() or not aligned["human_valid"][a:b].all()
                    or episode["split"] != entry["split"] or episode["start_local"][:10] != entry["recording_date"]
                    or episode["source_manifest_sha256"] != entry["source_manifest_sha256"]):
                raise ValueError(f"Invalid or overlapping source range: {entry['id']}")
            occupied[a:b] = True
            clip = checked_clip(library, entry)
            np.testing.assert_array_equal(clip["qd_ref"], velocity(clip["q_ref"]))
            if np.max(clip["static_penetration_m"]) > FILTER["max_static_penetration_m"]:
                raise ValueError(f"Static penetration exceeds the library filter: {entry['id']}")
            if entry["timing"] == "original":
                np.testing.assert_array_equal(clip["time_ns"], time[a:b])
                np.testing.assert_array_equal(clip["source_frame_index"], np.arange(a, b))
                continue
            source_time = clip["source_time_ns"]
            factor = entry["time_dilation_factor"]
            source_delta = np.diff(source_time)
            window_start, window_stop = entry["source_window"]
            source_index = np.searchsorted(time, source_time, side="right") - 1
            if (entry["timing"] != "retimed" or entry["split"] != "train" or factor <= 1
                    or not window_start <= a < b <= window_stop or source_time.shape != clip["time_ns"].shape
                    or not np.all(np.abs(source_delta * factor - PERIOD_NS) < factor)
                    or source_time[0] != entry["source_start_time_ns"] or source_time[-1] != entry["source_end_time_ns"]
                    or source_index[0] != a or source_index[-1] + 1 != b
                    or source_time[-1] > time[-1]):
                raise ValueError(f"Invalid original/retimed clock relationship: {entry['id']}")
            np.testing.assert_array_equal(clip["source_frame_index"], source_index)


def finalize(args):
    batch = args.output
    final = batch / "native_replay/motion_library"
    if final.parent.exists():
        raise ValueError(f"Final library already exists: {final}")
    original, original_proof = verified_replay(batch / "native_replay_original")
    slowed, slowed_proof = verified_replay(batch / "native_replay_retimed")
    base = original.manifest
    compare = ("provenance", "quality_filter", "rate_hz", "joint_names", "joint_limits_rad", "recording_dates")
    if any(base[key] != slowed.manifest[key] for key in compare):
        raise ValueError("Original and retimed clips use different contracts")
    for key in ("robot_model", "mujoco_version", "controller", "criteria"):
        if base["native_replay"][key] != slowed.manifest["native_replay"][key]:
            raise ValueError(f"Original and retimed replays differ in {key}")
    entries = original.manifest["clips"] + slowed.manifest["clips"]
    with TemporaryDirectory(prefix=".wholebody-wave-", dir=batch) as temporary:
        stage = Path(temporary) / "native_replay"
        root = stage / "motion_library"
        (root / "clips").mkdir(parents=True)
        copy_clips(original, root, original.manifest["clips"])
        copy_clips(slowed, root, slowed.manifest["clips"])
        replay = {key: base["native_replay"][key] for key in ("robot_model", "mujoco_version", "controller", "criteria")}
        replay["runs"] = {"original": original_proof, "retimed": slowed_proof}
        manifest = write_library(root, base, entries, native_replay=replay, retiming=slowed.manifest["retiming"])
        verify_capture(batch, MotionLibrary.open(root / "manifest.json"))
        if any(manifest["splits"][split]["clips"] < 1 for split in SPLITS):
            raise ValueError("A final split has no clips")
        for split in ("validation", "test"):
            if [entry for entry in entries if entry["split"] == split] != original.entries(split):
                raise ValueError(f"Held-out {split} inventory changed after retiming")
        summary = {
            "verified": True, "recording_dates": base["recording_dates"], "manifest": str(final / "manifest.json"),
            "manifest_sha256": digest(root / "manifest.json"), "clips": len(entries),
            "frames": sum(entry["frames"] for entry in entries), "splits": manifest["splits"],
            "by_date_and_timing": {}, "by_motion_type": {}, "replay": replay["runs"],
        }
        for key, group, bucket in (("recording_date", "timing", "by_date_and_timing"), ("task", "split", "by_motion_type")):
            counts = defaultdict(Counter)
            for entry in entries:
                counts[entry[key]][entry[group]] += 1
            summary[bucket] = {name: dict(counter) for name, counter in counts.items()}
        write_json(stage / "summary.json", summary)
        stage.rename(final.parent)
    print(f"Verified {summary['clips']} replay-approved clips: {final / 'manifest.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("merge", "verify", "viewer", "finalize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--dates", type=date.fromisoformat, nargs="+")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.stage == "merge":
        if not args.processed_root or not args.dates:
            parser.error("merge requires --processed-root and --dates")
        args.processed_root = args.processed_root.resolve(strict=True)
        merge(args)
    else:
        {"verify": verify, "viewer": viewer, "finalize": finalize}[args.stage](args)


if __name__ == "__main__":
    main()
