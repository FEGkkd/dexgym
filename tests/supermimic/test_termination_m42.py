import torch

from dexgym.algorithms.supermimic.termination import (
    TerminationThresholds, analytic_soft_termination, termination_calibration,
)
from dexgym.tasks.marvin_supermimic.contact_audit import unsafe_arm_states
from dexgym.algorithms.supermimic.imagination import imagined_rollout
from dexgym.algorithms.supermimic.action import ArmActionMap
from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from dexgym.algorithms.supermimic.differentiable_fk import DifferentiableMarvinFK
from dexgym.motion.motion_manager import MotionManager


def test_analytic_probability_tracks_hard_tcp_joint_and_velocity_boundaries():
    thresholds = TerminationThresholds()
    q = torch.zeros(4, 14, requires_grad=True)
    qd = torch.zeros(4, 14)
    lower, upper = -torch.ones(14), torch.ones(14)
    tcp = torch.tensor([[0.01, 0.01], [0.13, 0.01], [0.01, 0.01], [0.01, 0.01]])
    with torch.no_grad():
        q[2, 0] = 1.01
    qd[3, 0] = 5.1
    probability = analytic_soft_termination(q, qd, lower, upper, tcp, thresholds)
    hard = unsafe_arm_states(q.detach(), qd, lower, upper, tcp,
                             torch.zeros(4, dtype=torch.int32), thresholds=thresholds)
    assert hard.tolist() == [False, True, True, True]
    assert probability[0] < 0.01
    assert (probability[1:] > 0.5).all()
    probability[2].backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad[2, 0].abs() > 0


def test_termination_metrics_report_class_balance_and_brier_score():
    scores = torch.tensor([0.1, 0.9, 0.6, 0.2])
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
    result = termination_calibration(scores, labels)
    assert result["positive_count"] == 2 and result["negative_count"] == 2
    assert result["precision"] == 0.5 and result["recall"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert abs(result["brier_score"] - 0.255) < 1e-6


def test_imagination_analytic_mode_uses_next_state_and_arrival_reference():
    motion = MotionManager("/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/"
                           "cmu_141_16_wave_bimanual.npz", 1, "cpu")
    fk = DifferentiableMarvinFK("/home/ubuntu/dexgym/source/dexgym/assets/"
                                "marvin_wuji/source_model/dual_arm_marvin.xml")
    model = DeltaWorldModel(hidden=32)
    actor, critic = GoalActor(), PrivilegedCritic()
    origin = motion.data["q_arm"][0]
    mapping = ArmActionMap(origin, origin - 3, origin + 3, torch.ones(14))
    state = torch.cat((origin, torch.zeros(14), origin))[None]
    start = {"state": state, "history": torch.zeros(1, 10, 56), "frame": torch.tensor([0])}
    rollout = imagined_rollout(actor, model, critic, fk, mapping, motion, start,
                               horizon=1, termination_mode="analytic")
    q = rollout["terminal_state_detached"][:, :14]
    qd = rollout["terminal_state_detached"][:, 14:28]
    tcp = (fk(q)[0][:, :, 2] - motion.reference_at(torch.tensor([1]))["link_pos"][:, :, 2]).norm(dim=-1)
    expected = analytic_soft_termination(q, qd, mapping.lower, mapping.upper, tcp)
    torch.testing.assert_close(rollout["termination_probabilities"][0], expected)
