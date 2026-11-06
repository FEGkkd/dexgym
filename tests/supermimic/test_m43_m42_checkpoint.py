import pytest
import torch

from dexgym.algorithms.supermimic.checkpoint import load_legacy_m42_actor
from dexgym.tasks.marvin_supermimic.robot_adapter import DEFAULT_MJCF


CHECKPOINT = "/home/ubuntu/dexgym_data/marvin_supermimic_m42/pure_v3_seed44/iteration_000200.pt"
REFERENCE = "/home/ubuntu/dexgym_data/marvin_supermimic_m42/second_task/procedural_bimanual_reach_v1.npz"


def test_m42_procedural_actor_strictly_loads_and_rejects_wave_reference():
    actor, metadata = load_legacy_m42_actor(
        CHECKPOINT, reference_path=REFERENCE, robot_asset_path=DEFAULT_MJCF)
    assert metadata["action_schema"] == "robot_fixed_v3"
    with torch.no_grad():
        action = actor(torch.zeros(1, 10, 56), torch.zeros(1, 10, 90))
    assert action.shape == (1, 14) and torch.isfinite(action).all()
    with pytest.raises(ValueError, match="reference"):
        load_legacy_m42_actor(
            CHECKPOINT,
            reference_path="/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz",
            robot_asset_path=DEFAULT_MJCF)
