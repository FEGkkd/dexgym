from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dexteroushands_lab.tasks.direct.allegro_hand_track.reference_loader import (
    GrabReferenceError,
    allegro_source_to_sim_joint_order,
    load_grab_reference,
)


def _write_reference(path: Path, *, frames: int = 3) -> None:
    hand = np.arange(frames * 22, dtype=np.float32).reshape(frames, 22)
    object_translation = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    object_quaternion = np.tile(np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32), (frames, 1))
    np.save(
        path,
        {
            "robot_delta_states_weights_np": hand,
            "object_transl": object_translation,
            "object_rot_quat": object_quaternion,
        },
    )


def test_load_reference_validates_shapes_normalizes_quaternions_and_clamps_frames(tmp_path):
    path = tmp_path / "sample.npy"
    _write_reference(path)

    reference = load_grab_reference(path)

    assert reference.hand_target.shape == (3, 22)
    assert reference.object_translation.shape == (3, 3)
    assert reference.object_quaternion_xyzw.shape == (3, 4)
    assert torch.isfinite(reference.hand_target).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(reference.object_quaternion_xyzw, dim=-1),
        torch.ones(3),
    )

    hand, translation, quaternion = reference.frame(torch.tensor([-2, 1, 99]))
    torch.testing.assert_close(hand[0], reference.hand_target[0])
    torch.testing.assert_close(hand[2], reference.hand_target[-1])
    torch.testing.assert_close(translation[1], reference.object_translation[1])
    torch.testing.assert_close(quaternion[2], reference.object_quaternion_xyzw[-1])


def test_load_reference_rejects_missing_keys_and_nonfinite_values(tmp_path):
    missing = tmp_path / "missing.npy"
    np.save(missing, {"object_transl": np.zeros((2, 3), dtype=np.float32)})
    with pytest.raises(GrabReferenceError, match="required key"):
        load_grab_reference(missing)

    invalid = tmp_path / "invalid.npy"
    np.save(
        invalid,
        {
            "robot_delta_states_weights_np": np.full((2, 22), np.nan, dtype=np.float32),
            "object_transl": np.zeros((2, 3), dtype=np.float32),
            "object_rot_quat": np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        },
    )
    with pytest.raises(GrabReferenceError, match="finite"):
        load_grab_reference(invalid)


def test_joint_mapping_uses_explicit_source_and_destination_orders():
    values = np.arange(16, dtype=np.float32)
    source_names = tuple(f"source_{idx}" for idx in range(16))
    destination_names = source_names[4:] + source_names[:4]

    mapped = allegro_source_to_sim_joint_order(
        values,
        source_names=source_names,
        destination_names=destination_names,
    )

    np.testing.assert_array_equal(mapped, np.arange(16, dtype=np.float32)[4:].tolist() + np.arange(16, dtype=np.float32)[:4].tolist())

    with pytest.raises(ValueError, match="duplicate"):
        allegro_source_to_sim_joint_order(
            values,
            source_names=source_names,
            destination_names=source_names[:-1] + (source_names[-2],),
        )


@pytest.mark.skipif(
    not list(Path("/home/ubuntu/dexgym/data").glob("**/*.npy")),
    reason="processed GRAB archive has not been extracted into the workspace",
)
def test_real_downloaded_grab_sample():
    sample = next(Path("/home/ubuntu/dexgym/data").glob("**/*.npy"))
    reference = load_grab_reference(sample)
    assert reference.hand_target.shape[-1] == 22
