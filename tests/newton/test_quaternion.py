import torch
from dexgym.utils.quaternion import wxyz_to_xyzw, xyzw_to_wxyz


def test_quaternion_round_trip():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    torch.testing.assert_close(wxyz_to_xyzw(q)[0], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(xyzw_to_wxyz(wxyz_to_xyzw(q)), q)
