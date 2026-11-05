import numpy as np
import torch
from dexgym.algorithms.supermimic.action import ArmActionMap
from dexgym.motion.motion_manager import MotionManager, quaternion_xyzw_to_6d

PATH = "/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz"


def test_public_task_registration():
    import dexgym.tasks
    import gymnasium as gym

    assert "DexGym-Marvin-SuperMimic-Wave-Direct-v0" in dexgym.tasks.TASK_IDS
    assert gym.spec("DexGym-Marvin-SuperMimic-Wave-Direct-v0").entry_point.endswith("MarvinWaveDirectEnv")


def test_future_end_holds_pose_and_zeros_held_velocities():
    manager = MotionManager(PATH, 3, "cpu")
    manager.reset(torch.tensor([0, 1, 2]), torch.tensor([0, 1355, 1356]))
    future = manager.future_links()
    assert manager.goal().shape == (3, 10, 90)
    assert future["link_pos"].shape == (3, 10, 2, 3, 3)
    torch.testing.assert_close(future["link_pos"][2], manager.data["link_pos"][-1].expand(10, -1, -1, -1))
    assert torch.count_nonzero(future["link_linear_vel"][2]) == 0
    assert torch.count_nonzero(future["link_angular_vel"][2]) == 0
    torch.testing.assert_close(future["link_quat_xyzw"][2], manager.data["link_quat_xyzw"][-1].expand(10, -1, -1, -1))


def test_rotation_6d_identity_and_action_gradients():
    six = quaternion_xyzw_to_6d(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(six, torch.tensor([[1.0, 0, 0, 0, 1.0, 0]]))
    origin = torch.zeros(14)
    action_map = ArmActionMap(origin, -torch.ones(14), torch.ones(14), torch.ones(14), max_delta=0.08)
    action = torch.full((2, 14), 0.02, requires_grad=True)
    command = action_map(action, torch.zeros(2, 14))
    assert command.shape == (2, 14)
    assert np.isfinite(command.detach().numpy()).all()
    assert command.abs().max() <= 0.08
    command.sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
