"""Capture exact Phase 1 source and bank bytes for reproducible PPO runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def snapshot_files(log_dir: Path, repo_root: Path, relative_paths: list[Path]) -> dict[str, str]:
    """Copy small relevant inputs to ignored logs and return their SHA-256s."""
    hashes = {}
    for relative in sorted(set(relative_paths)):
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Training input missing: {relative}")
        content = source.read_bytes()
        destination = log_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    return hashes


def _git_head(path: Path) -> str | None:
    result = subprocess.run(("git", "-C", str(path), "rev-parse", "HEAD"),
                            check=False, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def write_bimanual_training_manifest(log_dir: Path, repo_root: Path,
                                     task_id: str, asset_root: Path | None) -> Path:
    task_dir = Path("source/dexteroushands_lab/tasks/direct/bimanual_bottle")
    paths = [p.relative_to(repo_root) for p in (repo_root / task_dir).rglob("*.py")]
    paths += [Path(name) for name in (
        "source/dexteroushands_lab/assets/batched_wrist_pose.py",
        "source/dexteroushands_lab/assets/bottle/batched_thread.py",
        "source/dexteroushands_lab/assets/bottle/config.py",
        "source/dexteroushands_lab/assets/wuji_hand2_beta.py",
        "source/dexteroushands_lab/scenes/batched_contacts.py",
        "source/dexteroushands_lab/scenes/bimanual_bottle.py",
        "docs/assets/phase0e_bimanual_grasp_init_bank.json",
        "docs/assets/phase0_pregrasp_reset_bank.json",
        "rsl_rl/base/train.py",
    )]
    hashes = snapshot_files(log_dir, repo_root, paths)
    manifest = {
        "schema": "dexgym-phase1-training-manifest-v1",
        "task_id": task_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dexgym_git_head": _git_head(repo_root),
        "wuji_description_git_head": _git_head(asset_root) if asset_root else None,
        "source_sha256": hashes,
    }
    path = log_dir / "phase1_training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
