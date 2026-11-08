"""Runtime sequence indexing and split isolation."""

import json

import numpy as np
import torch
from dexgym.data.motion_bank import MotionBank, advance_goal_frames


def test_motion_bank_keeps_train_validation_separate_and_indexes_frames(tmp_path):
    records = []
    for name, split, length, marker in (("train", "train", 8, 1.0), ("heldout", "validation", 6, 2.0)):
        q = np.full((length, 20), marker, dtype=np.float32)
        tips = np.full((length, 5, 3), marker, dtype=np.float32)
        np.savez_compressed(tmp_path / f"{name}.npz", q_ref=q, tip_ref=tips, fps=np.float32(30))
        records.append({"file": f"{name}.npz", "split": split, "frames": length})
    (tmp_path / "manifest.json").write_text(json.dumps({"motions": records, "fps": 30.0}))

    bank = MotionBank(tmp_path / "manifest.json", split="train", device="cpu")
    assert bank.motion_count == 1
    ids, frames = bank.sample_starts(32, stride=3)
    assert ids.shape == (32,) and frames.shape == (32,)
    assert torch.all(ids == 0)
    assert torch.all(frames <= 4)
    assert torch.all(bank.joints(ids, frames) == 1.0)
    assert torch.all(bank.fingertips(ids, frames + 3) == 1.0)
    heldout = MotionBank(tmp_path / "manifest.json", split="validation", device="cpu")
    assert heldout.motion_count == 1
    assert torch.all(heldout.joints(torch.tensor([0]), torch.tensor([0])) == 2.0)


def test_goal_advance_stops_at_last_frame_without_wraparound():
    current = torch.tensor([3, 7, 4])
    length = torch.tensor([8, 8, 8])
    reached = torch.tensor([True, True, False])
    next_frame, completed = advance_goal_frames(current, length, reached, stride=3)
    assert next_frame.tolist() == [6, 7, 4]
    assert completed.tolist() == [False, True, False]


def test_motion_bank_can_skip_short_segments_without_changing_manifest(tmp_path):
    records = []
    for name, length in (("short", 3), ("usable", 8)):
        np.savez_compressed(
            tmp_path / f"{name}.npz",
            q_ref=np.zeros((length, 20), dtype=np.float32),
            tip_ref=np.zeros((length, 5, 3), dtype=np.float32),
        )
        records.append({"file": f"{name}.npz", "split": "train", "frames": length})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"motions": records, "fps": 30.0}))

    bank = MotionBank(manifest, split="train", device="cpu", min_frames=6)
    assert bank.motion_count == 1
    assert bank.records[0]["file"] == "usable.npz"
    assert bank.sample_starts(16, stride=4)[0].tolist() == [0] * 16
