"""Build a task-space bank from recorded 50 Hz Tracker and MANUS sequences."""

import argparse
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from ..wholebody_s0.bag import digest
from ..wholebody_s0.intent import source_orders
from ..wholebody_s0.library import MotionLibrary, SPLITS
from ..wholebody_s0.prepare import write_json, write_npz
from . import DT, SCHEMA, TASK_NAME
from .targets import TargetMapper, command_bounds, joint_names, native_state, recorded_home

HARD_DEPTH = .003


def reset_depths(model, q):
    data = mujoco.MjData(model)
    out = np.empty(len(q), dtype=np.float32)
    for i, state in enumerate(q):
        data.qpos[:] = state
        mujoco.mj_fwdPosition(model, data)
        out[i] = max(0., -float(data.contact.dist.min(initial=0.)))
    return out


def prepare(args):
    lib = MotionLibrary.open(args.manifest)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if lib.manifest["rate_hz"] != round(1 / DT):
        raise ValueError("Expected synchronized 50 Hz recordings")
    names = joint_names(model)
    order = [lib.manifest["joint_names"].index(name) for name in names]
    lo, hi = command_bounds(model, names)
    home = recorded_home(model, args.arm_home, args.hand_home)
    home_data = native_state(model, home)
    home_depth = max(0., -float(home_data.contact.dist.min(initial=0.)))
    if home_depth > HARD_DEPTH:
        raise ValueError("Recorded Home starts with a hard collision in the teleoperation model")
    mapper = TargetMapper(model)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": SCHEMA, "task": TASK_NAME, "rate_hz": lib.manifest["rate_hz"], "joint_names": names,
        "source_manifest": str(args.manifest.resolve()), "source_manifest_sha256": digest(args.manifest),
        "teleop_model": str(args.model.resolve()), "teleop_model_sha256": digest(args.model),
        "home": home.tolist(), "arm_home_file": str(args.arm_home.resolve()), "arm_home_sha256": digest(args.arm_home),
        "hand_home_file": str(args.hand_home.resolve()), "hand_home_sha256": digest(args.hand_home),
        "home_reset_depth_m": home_depth, "position_lower_rad": lo.tolist(), "position_upper_rad": hi.tolist(),
        "frame": {"wrist": "fixed_robot_world", "fingers": "same_side_tcp", "relative": "right_tcp_in_left_tcp",
                  "fingers_order": ["thumb", "index", "middle", "ring", "little"], "landmarks": ["joint_2", "joint_3", "tip"]},
        "splits": {},
    }
    episode_splits, orders = {}, {}
    for split in SPLITS:
        entries = lib.entries(split)
        if args.limit_per_split is not None:
            entries = entries[:args.limit_per_split]
        rows, arrays, offset, counts = [], {}, 0, Counter()
        for entry in entries:
            episode = entry["source_episode"]
            if episode_splits.setdefault(episode, split) != split:
                raise ValueError(f"Source recording crosses splits: {episode}")
            if split != "train" and entry.get("timing", "original") != "original":
                raise ValueError(f"Retimed clip is in held-out split: {entry['id']}")
            source_path = (lib.root / entry["path"]).resolve()
            if not source_path.is_relative_to(lib.root) or digest(source_path) != entry["sha256"]:
                raise ValueError(f"Unexpected source clip: {entry['id']}")
            clip = lib.load_clip(entry)
            if episode not in orders:
                orders[episode] = source_orders(lib, entry)
            wrist, rotation, hand = mapper.from_clip(clip, orders[episode])
            inverse_left = rotation[:, 0].transpose(0, 2, 1)
            q = clip["q_ref"][:, order]
            depth = reset_depths(model, q)
            reset_safe = (depth <= HARD_DEPTH) & ((q >= lo - 1e-6) & (q <= hi + 1e-6)).all(-1)
            values = {
                "wrist": wrist, "rotation": rotation, "hand": hand,
                "relative_position": np.einsum("tij,tj->ti", inverse_left, wrist[:, 1] - wrist[:, 0]),
                "relative_rotation": inverse_left @ rotation[:, 1],
                "rsi_q": q, "rsi_qd": clip["qd_ref"][:, order], "reset_depth_m": depth, "reset_safe": reset_safe,
                "time_ns": clip["time_ns"],
            }
            for name, array in values.items():
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite {name} in {entry['id']}")
                arrays.setdefault(name, []).append(array.astype(np.float32) if array.dtype.kind == "f" else array)
            n = len(q)
            rows.append({"id": entry["id"], "episode": episode, "start": offset, "stop": offset + n,
                         "timing": entry.get("timing", "original")})
            offset += n
            counts.update(safe_frames=int(reset_safe.sum()), safe_starts=int(reset_safe[0]))
        if not rows:
            raise ValueError(f"No data in {split}")
        path = args.output / f"{split}.npz"
        write_npz(path, {key: np.concatenate(value, axis=0) for key, value in arrays.items()})
        metadata["splits"][split] = {"file": path.name, "sha256": digest(path), "clips": rows, "frames": offset,
                                     **counts}
    write_json(args.output / "manifest.json", metadata)
    print({split: {key: value[key] for key in ("frames", "safe_frames", "safe_starts")}
           for split, value in metadata["splits"].items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("manifest", "model", "arm-home", "hand-home", "output"):
        p.add_argument(f"--{flag}", required=True, type=Path)
    p.add_argument("--limit-per-split", type=int)
    args = p.parse_args()
    if args.limit_per_split is not None and args.limit_per_split < 1:
        p.error("--limit-per-split must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
