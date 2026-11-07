"""Build a 30-Hz library of reachable Wuji right-hand fingertip motions."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .grab_mano import load_mano_model, load_subject_template, mano_keypoints
from .wuji_retarget import (
    FIRMWARE_JOINT_NAMES,
    SDK_VERSION,
    TIP_SITE_NAMES,
    RightHandFK,
    WujiRetargetSession,
    retarget_sequence,
)


def sample_indices(frame_count: int, source_fps: float, output_fps: float = 30.0) -> np.ndarray:
    """Choose exact source frames when the source rate is an integer multiple."""
    ratio = source_fps / output_fps
    stride = round(ratio)
    if frame_count < 1 or stride < 1 or not np.isclose(ratio, stride, rtol=0.0, atol=1e-6):
        raise ValueError(f"Expected a positive integer source/output frame-rate ratio: {source_fps}/{output_fps}")
    return np.arange(0, frame_count, stride, dtype=np.int32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_library(
    *,
    grab_root: str | Path,
    data_root: str | Path,
    mano_path: str | Path,
    sdk_path: str | Path,
    right_xml: str | Path,
    output_dir: str | Path,
    selected: Iterable[str | Path] | None = None,
    output_fps: float = 30.0,
) -> dict:
    """Convert selected GRAB files, retaining errors in a reproducible manifest."""
    grab_root = Path(grab_root).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    mano_path = Path(mano_path).expanduser().resolve()
    sdk_path = Path(sdk_path).expanduser().resolve()
    right_xml = Path(right_xml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected is not None:
        sources = sorted(Path(path).resolve() for path in selected)
    else:
        sources = sorted(grab_root.glob("s*/*.npz"))
    if not sources:
        raise ValueError(f"No GRAB motion files found under {grab_root}")
    model = load_mano_model(mano_path)
    fk = RightHandFK(right_xml)
    template_cache: dict[Path, np.ndarray] = {}
    manifest: dict = {
        "format_version": 1,
        "description": "GRAB right hand -> Wuji SDK joint targets -> Wuji FK fingertips in r_wrist frame",
        "fps": output_fps,
        "joint_names": list(FIRMWARE_JOINT_NAMES),
        "tip_names": list(TIP_SITE_NAMES),
        "train_subjects": ["s1", "s2", "s3", "s4", "s5"],
        "validation_subjects": ["s8"],
        "sdk_version": SDK_VERSION.decode(),
        "mano_sha256": _sha256(mano_path),
        "right_xml_sha256": _sha256(right_xml),
        "accepted": 0,
        "rejected": 0,
        "total_frames": 0,
        "motions": [],
        "errors": [],
    }
    with WujiRetargetSession(sdk_path) as session:
        for source in sources:
            try:
                if not source.is_relative_to(grab_root) or source.parent.name not in {
                    "s1", "s2", "s3", "s4", "s5", "s8"
                }:
                    raise ValueError("Motion must be in a known GRAB subject directory")
                with np.load(source, allow_pickle=True) as raw:
                    hand = raw["rhand"].item()
                    source_fps = float(raw["framerate"])
                    fullpose = np.asarray(hand["params"]["fullpose"], dtype=np.float32)
                    if fullpose.ndim != 2 or fullpose.shape[1] != 45:
                        raise ValueError("Expected GRAB right-hand fullpose[T,45]")
                    indices = sample_indices(len(fullpose), source_fps, output_fps)
                    template_path = (data_root / hand["vtemp"]).resolve()
                    if not template_path.is_relative_to(data_root):
                        raise ValueError("Subject template path escapes data root")
                if template_path not in template_cache:
                    template_cache[template_path] = load_subject_template(template_path)
                keypoints = mano_keypoints(fullpose[indices], template_cache[template_path], model)
                q, tips, metrics = retarget_sequence(keypoints, session, fk)
                if not np.isfinite(q).all() or not np.isfinite(tips).all():
                    raise ValueError("Retargeted motion contains nonfinite data")
                if np.any(q < fk.lower - 1e-6) or np.any(q > fk.upper + 1e-6):
                    raise ValueError("Retargeted joint lies outside the robot model limits")
                relative = source.relative_to(grab_root)
                filename = f"{relative.parent.name}_{source.stem}.npz"
                target = output_dir / filename
                temporary = output_dir / f"{filename}.tmp"
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        q_ref=q,
                        tip_ref=tips,
                        source_indices=indices,
                        fps=np.float32(output_fps),
                    )
                temporary.replace(target)
                tip_step = float(np.max(np.linalg.norm(np.diff(tips, axis=0), axis=-1))) if len(tips) > 1 else 0.0
                entry = {
                    "source": str(relative),
                    "template": str(template_path.relative_to(data_root)),
                    "file": filename,
                    "split": "validation" if relative.parent.name == "s8" else "train",
                    "source_fps": source_fps,
                    "source_frames": len(fullpose),
                    "frames": len(q),
                    "max_tip_step_m": tip_step,
                    **metrics,
                }
                manifest["motions"].append(entry)
                manifest["accepted"] += 1
                manifest["total_frames"] += len(q)
            except (KeyError, ValueError, OSError, RuntimeError) as error:
                manifest["errors"].append({"source": str(source), "error": f"{type(error).__name__}: {error}"})
                manifest["rejected"] += 1
    temporary_manifest = output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(output_dir / "manifest.json")
    return manifest
