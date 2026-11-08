"""A clean library contains only fully replayed, contact-refined motions."""

import hashlib

import pytest
from dexgym.data.quality_filter import fingerprint_files, select_clean_records, verify_replay_file_hashes


def test_excludes_tracking_failures_and_sdk_fallbacks():
    records = [
        {
            "source": "s8/flashlight_on_1.npz",
            "split": "validation",
            "frames": 3,
            "postoptimization": {"fallback_reasons": []},
        },
        {
            "source": "s8/fryingpan_cook_2.npz",
            "split": "validation",
            "frames": 3,
            "postoptimization": {"fallback_reasons": []},
        },
        {
            "source": "s8/hammer_use_1.npz",
            "split": "validation",
            "frames": 3,
            "postoptimization": {"fallback_reasons": []},
        },
        {
            "source": "s1/fallback.npz",
            "split": "train",
            "frames": 3,
            "postoptimization": {"fallback_reasons": ["tip_change_over_12mm"]},
        },
    ]
    replay = {
        "s8/flashlight_on_1.npz": {"sampled_frames": 3, "all_five_within_10mm_frames": 2},
        "s8/fryingpan_cook_2.npz": {"sampled_frames": 3, "all_five_within_10mm_frames": 2},
        "s8/hammer_use_1.npz": {"sampled_frames": 3, "all_five_within_10mm_frames": 3},
        "s1/fallback.npz": {"sampled_frames": 3, "all_five_within_10mm_frames": 3},
    }
    kept, excluded = select_clean_records(records, replay)
    assert [entry["source"] for entry in kept] == ["s8/hammer_use_1.npz"]
    assert set(excluded) == {"s8/flashlight_on_1.npz", "s8/fryingpan_cook_2.npz", "s1/fallback.npz"}


def test_rejects_missing_or_incomplete_replay_evidence():
    record = {"source": "s8/test.npz", "split": "validation", "frames": 4, "postoptimization": {"fallback_reasons": []}}
    with pytest.raises(ValueError, match="Missing replay"):
        select_clean_records([record], {})
    with pytest.raises(ValueError, match="Incomplete replay"):
        select_clean_records([record], {"s8/test.npz": {"sampled_frames": 3, "all_five_within_10mm_frames": 3}})


def test_replay_evidence_is_bound_to_motion_file_contents(tmp_path):
    file = tmp_path / "motion.npz"
    file.write_bytes(b"original motion")
    entry = {"source": "s8/motion.npz", "file": file.name}
    replay = {entry["source"]: {"input_npz_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}}
    verify_replay_file_hashes([entry], replay, tmp_path)
    file.write_bytes(b"changed motion")
    with pytest.raises(ValueError, match="hash"):
        verify_replay_file_hashes([entry], replay, tmp_path)


def test_controller_fingerprint_changes_when_asset_changes(tmp_path):
    file = tmp_path / "hand_x2.usda"
    file.write_text("gain=1")
    before = fingerprint_files([file])
    file.write_text("gain=2")
    assert fingerprint_files([file]) != before
