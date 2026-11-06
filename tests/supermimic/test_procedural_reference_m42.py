import numpy as np
import torch

from dexgym.motion.motion_manager import MotionManager
from dexgym.motion.procedural import generate_procedural_bimanual_reference


def test_procedural_bimanual_reference_is_distinct_smooth_and_fk_consistent(tmp_path):
    path = tmp_path / "procedural.npz"
    metadata = generate_procedural_bimanual_reference(path, frames=101)
    assert metadata["provenance"] == "procedural_test"
    motion = MotionManager(path, 1, "cpu", motion_id="procedural_bimanual_reach_v1")
    with np.load(path, allow_pickle=False) as data:
        q = data["q_arm"]
        assert q.shape == (101, 14)
        assert np.max(np.abs(np.diff(q, axis=0))) < 0.04
        assert np.ptp(q[:, :7]) > 0.04 and np.ptp(q[:, 7:]) > 0.04
        assert np.allclose(data["tcp_pos"], data["link_pos"][:, :, 2], atol=1e-6)
    assert motion.length == 101
    assert torch.isfinite(motion.future_goal_at(torch.tensor([0]))).all()
