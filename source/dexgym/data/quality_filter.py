"""Conservative selection of fully replayable, contact-refined hand motions."""

import hashlib
import importlib.metadata
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(paths: Sequence[str | Path], *, root: str | Path | None = None) -> str:
    """Hash file names and contents so an asset change invalidates old evidence."""
    base = Path(root).resolve() if root is not None else None
    digest = hashlib.sha256()
    for item in sorted((Path(path).resolve() for path in paths), key=str):
        label = item.relative_to(base).as_posix() if base is not None else item.name
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def replay_contract_sha256(repo_root: str | Path) -> str:
    """Bind replay evidence to the controller, robot assets, and solver build."""
    root = Path(repo_root).resolve()
    local = [
        root / relative
        for relative in (
            "scripts/probe_right_hand_sequence_replay.py",
            "scripts/data/screen_grab_right_hand_library.py",
            "scripts/data/build_clean_grab_right_hand_library.py",
            "source/dexgym/data/quality_filter.py",
            "source/dexgym/data/motion_bank.py",
            "source/dexgym/tasks/direct/right_hand_tracking/drive_experiment_cfg.py",
            "source/dexgym/tasks/direct/right_hand_tracking/env_cfg.py",
            "source/dexgym/tasks/direct/right_hand_tracking/env.py",
            "source/dexgym/assets/marvin_wuji/robot_cfg.py",
        )
    ]
    asset_root = root / "source/dexgym/assets/marvin_wuji"
    local.extend(path for path in asset_root.rglob("*") if path.is_file() and path.suffix != ".pyc")
    solver_spec = importlib.util.find_spec("isaaclab_newton")
    solver_root = (
        Path(next(iter(solver_spec.submodule_search_locations))).resolve()
        if solver_spec is not None and solver_spec.submodule_search_locations
        else None
    )
    solver_files = list(solver_root.rglob("*.py")) if solver_root is not None else []
    versions = {}
    for package in ("isaaclab", "newton", "mujoco", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    contract = {
        "name": "right-hand-full-contact-hand_x2-500hz-30hz-1p5s-prehold-v1",
        "repo_files_sha256": fingerprint_files(local, root=root),
        "solver_files_sha256": fingerprint_files(solver_files, root=solver_root) if solver_root is not None else None,
        "package_versions": versions,
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def verify_replay_file_hashes(records: Sequence[dict], replay: Mapping[str, dict], directory: str | Path) -> None:
    """Reject replay evidence for any motion whose saved NPZ bytes changed."""
    directory = Path(directory)
    for entry in records:
        source = entry["source"]
        if source not in replay or replay[source].get("input_npz_sha256") != sha256_file(directory / entry["file"]):
            raise ValueError(f"Replay motion hash mismatch or missing hash for {source}")


def select_clean_records(records: Sequence[dict], replay: Mapping[str, dict]) -> tuple[list[dict], dict[str, dict]]:
    """Keep only clips whose every replayed frame meets the five-tip 10 mm gate."""
    kept: list[dict] = []
    excluded: dict[str, dict] = {}
    for entry in records:
        source = entry["source"]
        if source not in replay:
            raise ValueError(f"Missing replay evidence for {source}")
        row = replay[source]
        if int(row["sampled_frames"]) != int(entry["frames"]):
            raise ValueError(f"Incomplete replay evidence for {source}")
        passing = int(row["all_five_within_10mm_frames"])
        if not 0 <= passing <= int(entry["frames"]):
            raise ValueError(f"Invalid replay pass count for {source}")
        reasons = []
        if passing < int(entry["frames"]):
            reasons.append("replay_tip_error_at_least_10mm")
        reasons.extend(f"sdk_fallback:{reason}" for reason in entry["postoptimization"]["fallback_reasons"])
        if reasons:
            excluded[source] = {
                "reasons": reasons,
                "passing_frames": passing,
                "frames": int(entry["frames"]),
                "max_replay_tip_error_mm": row.get("max_five_finger_max_error_mm"),
            }
        else:
            kept.append(entry)
    return kept, excluded
