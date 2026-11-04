"""Contact pair indexing must never mix identical bodies across env rows."""

import torch

from dexteroushands_lab.scenes.batched_contacts import gather_local_contact_forces


def test_gather_local_contact_forces_ignores_cross_environment_columns():
    # Three source objects, two named hand bodies in each environment.
    matrix = torch.zeros(3, 6, 3)
    matrix[0, 1, 2] = 2.0
    matrix[1, 2, 2] = 3.0
    matrix[2, 5, 2] = 4.0
    matrix[1, 0, 2] = 99.0  # force from an incorrect cross-env filter
    got = gather_local_contact_forces(matrix, num_bodies=2)
    assert got.shape == (3, 2, 3)
    assert got[:, :, 2].tolist() == [[0., 2.], [3., 0.], [0., 4.]]


def test_gather_rejects_bad_matrix_shape():
    try:
        gather_local_contact_forces(torch.zeros(2, 3, 3), num_bodies=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid contact filter mapping was accepted")
