import torch
from dexgym.algorithms.supermimic.differentiable_fk import DifferentiableMarvinFK
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES, RewardWeights, so3_geodesic, tracking_reward
from dexgym.motion.motion_manager import MotionManager

MJCF = "/home/ubuntu/dexgym/source/dexgym/assets/marvin_wuji/source_model/dual_arm_marvin.xml"
NPZ = "/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz"


def test_reward_profiles_are_distinct_and_tcp_priority_is_stronger():
    assert set(REWARD_PROFILES) == {"openarm_style", "marvin_tcp_priority"}
    baseline = REWARD_PROFILES["openarm_style"]
    priority = REWARD_PROFILES["marvin_tcp_priority"]
    assert priority.tcp_position > baseline.tcp_position
    assert priority.tcp_rotation > baseline.tcp_rotation
    assert priority.link_position < baseline.link_position


def test_so3_orientation_distance_and_reward_ascend():
    ref = MotionManager(NPZ, 1, "cpu")
    fk = DifferentiableMarvinFK(MJCF)
    ref.reset(torch.tensor([0]), 200)
    target = {key: ref.current(key) for key in ("link_pos", "link_rotmat", "link_linear_vel", "link_angular_vel")}
    baseline = ref.current("q_arm").clone()
    q = baseline.clone()
    q[0, 0] += 0.08
    q[0, 9] -= 0.08
    q.requires_grad_()
    weights = RewardWeights(
        tcp_position=0,
        tcp_rotation=1,
        link_position=0,
        link_rotation=0,
        linear_velocity=0,
        angular_velocity=0,
        action_rate=0,
        joint_limit=0,
    )
    lower = torch.full((14,), -10.0)
    upper = -lower
    reward, diagnostics = tracking_reward(q, q.detach(), q, q.detach(), target, fk, lower, upper, weights)
    assert diagnostics["tcp_rotation_error_rad"].min() > 0
    assert torch.isfinite(reward).all()
    gradient = torch.autograd.grad(reward.sum(), q)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    improved, _ = tracking_reward(
        q.detach() + 0.05 * gradient, q.detach(), q.detach(), q.detach(), target, fk, lower, upper, weights
    )
    assert improved.item() > reward.item()
    identity = torch.eye(3)[None]
    torch.testing.assert_close(so3_geodesic(identity, identity), torch.zeros(1))
    half_turn = torch.diag(torch.tensor([1.0, -1.0, -1.0]))[None]
    torch.testing.assert_close(so3_geodesic(half_turn, identity), torch.full((1,), torch.pi))
