"""Reference bank controls that do not touch device drivers or create physics episodes."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.v1.rmb.domains import contiguous_segments, radial_lower_bound
from dexgym.tasks.direct.wholebody_s05.v1.rmb.mapping import canonical_xyzw, packed_geometry, pinch_points, smooth_weights
from dexgym.tasks.direct.wholebody_s05.v1.rmb.reader import RobotReferenceBank
from dexgym.tasks.direct.wholebody_s05.v1.rmb.schema import SCHEMA, file_sha, write_partition


def test_motion_segments_cannot_bridge_invalid_ticks_or_generation():
    mask = np.array([True, True, False, True, True, True, True])
    splits = np.array([0, 0, 0, 1, 1, 1, 1])
    generation = np.array([4, 4, 4, 5, 5, 6, 6])
    assert contiguous_segments(mask, splits, generation) == [(0, 2), (3, 5), (5, 7)]


def test_hard_pinch_is_exact_on_active_pairs():
    p = np.random.default_rng(49).normal(size=(5, 4, 3))
    d = np.random.default_rng(50).normal(size=(4, 3))
    hard = pinch_points(p, d, np.array([1, 0, 1, 0]))
    relative = hard[1:, -1] - hard[0, -1]
    np.testing.assert_allclose(relative[[0, 2]], d[[0, 2]], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(hard[[2, 4]], p[[2, 4]])


def test_smooth_candidates_are_continuous_at_band_boundaries():
    low, high = 0.018, 0.022
    assert smooth_weights(np.array([low, high]), (low, high)).tolist() == [1.0, 0.0]
    eps = 1e-8
    np.testing.assert_allclose(smooth_weights(np.array([low + eps, high - eps]), (low, high)),
                               [1.0, 0.0], rtol=0, atol=1e-8)


def test_cross_hand_uses_left_major_then_right_order():
    world = np.zeros((2, 5, 4, 3))
    world[0, :, -1, 0] = np.arange(5)
    world[1, :, -1, 0] = 10 * np.arange(5)
    p = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 2.0]])
    r = Rotation.from_rotvec([[0, 0, np.pi / 2], [0, 0, 0]]).as_matrix()
    data = packed_geometry(p, r, np.zeros_like(world), world)
    np.testing.assert_allclose(data["relative_wrist_xyz"], [1, 0, 0], atol=1e-12)
    assert data["cross_xyz"][1, 2, 0] == 19
    assert data["cross_xyz"][2, 1, 0] == 8
    assert canonical_xyzw(np.eye(3))[3] == 1


def test_reach_bound_is_only_a_strict_exclusion():
    positions = np.zeros((2, 5, 3))
    positions[:, :, 0] = 0.5
    bounds = radial_lower_bound(positions, (np.zeros_like(positions), np.ones((2, 5))))
    assert not bounds.any()
    positions[1, 2, 0] = 1.01
    bounds = radial_lower_bound(positions, (np.zeros_like(positions), np.ones((2, 5))))
    assert 0 < bounds[1, 2] < 0.01
    assert np.count_nonzero(bounds) == 1


def test_reader_distinguishes_smoke_and_fingerprint(tmp_path):
    rel = "partitions/Train/0001"
    arrays = write_partition(tmp_path / rel, {"tick_ns": np.arange(5, dtype="<i8") * 20_000_000})
    manifest = {"schema": SCHEMA, "qualification": "SMOKE_ONLY", "partitions": [
        {"split": "Train", "root": "root1", "key": "part1", "relative_path": rel, "arrays": arrays},
    ]}
    path = tmp_path / "rmb_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not eligible"):
        RobotReferenceBank(tmp_path, expected_sha=file_sha(path))
    bank = RobotReferenceBank(tmp_path, expected_sha=file_sha(path), require_qualified=False)
    part = next(bank.partitions("Train"))
    assert part.arrays["tick_ns"][-1] == 80_000_000
    assert not part.arrays["tick_ns"].flags.writeable
    with pytest.raises(ValueError, match="from this split"):
        bank.read("part1", "Validation")
    array_path = tmp_path / rel / "tick_ns.npy"
    with Path(array_path).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        bank.read("part1", "Train")
