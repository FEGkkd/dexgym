"""Numerical regression against the untouched M4.1 actor implementation."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from dexgym.algorithms.supermimic.checkpoint import load_legacy_m41_actor


CHECKPOINT = Path("/home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/selected_best.pt")
EXPECTED_SHA = "fd0054a7ef707d46a30b0235713c9e00cc4d7e69d6687fc3015e52aac2cb773f"


def test_legacy_selected_actor_matches_m41_source_output():
    if not CHECKPOINT.is_file():
        pytest.skip("Immutable M4.1 selected checkpoint is not mounted")
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_SHA
    fixture = np.load(Path(__file__).with_name("legacy_m41_actions.npz"))
    actor = load_legacy_m41_actor(CHECKPOINT)
    with torch.no_grad():
        actual = actor(torch.from_numpy(fixture["history"]), torch.from_numpy(fixture["goal"]))
    torch.testing.assert_close(actual, torch.from_numpy(fixture["action"]), rtol=0, atol=2e-6)
