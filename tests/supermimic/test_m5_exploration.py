import torch

from dexgym.tasks.marvin_supermimic.se3_collector import TemporalJointExploration


def test_strict_pure_exploration_is_seeded_bounded_and_correlated():
    a = TemporalJointExploration(3, 14, device="cpu", seed=77, rho=0.95, sigma=0.8)
    b = TemporalJointExploration(3, 14, device="cpu", seed=77, rho=0.95, sigma=0.8)
    samples_a = torch.stack([a(torch.zeros(3, 14)) for _ in range(100)])
    samples_b = torch.stack([b(torch.zeros(3, 14)) for _ in range(100)])
    assert torch.equal(samples_a, samples_b)
    assert samples_a.std() > 0.1
    assert torch.corrcoef(torch.stack((samples_a[:-1].flatten(), samples_a[1:].flatten())))[0, 1] > 0.5
    a.reset(torch.tensor([1]))
    assert torch.all(a.state[1] == 0)


def test_heldout_and_train_actor_lanes_remain_deterministic():
    exploration = TemporalJointExploration(4, 14, device="cpu", seed=91,
                                           rho=0.95, sigma=0.8)
    clean = torch.full((4, 14), 0.25)
    active = torch.tensor([True, False, False, False])
    first = exploration(clean, active_mask=active)
    second = exploration(clean, active_mask=active)
    assert not torch.equal(first[0], clean[0])
    assert not torch.equal(first[0], second[0])
    assert torch.equal(first[1:], clean[1:])
    assert torch.equal(second[1:], clean[1:])
    assert torch.equal(exploration.state[1:], torch.zeros_like(exploration.state[1:]))
