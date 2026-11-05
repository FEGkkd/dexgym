import torch

from dexgym.algorithms.supermimic.critic_optimizer import optimize_critic
from dexgym.algorithms.supermimic.imagination import critic_td_lambda_batch, lambda_returns
from dexgym.algorithms.supermimic.networks import PrivilegedCritic


class TinyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward_features(self, features):
        return self.linear(features).squeeze(-1)


def test_critic_optimizer_runs_requested_epoch_minibatch_schedule():
    torch.manual_seed(4)
    critic = TinyCritic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)
    features = torch.randn(16, 2)
    targets = features[:, 0] - features[:, 1]
    before = (critic.forward_features(features) - targets).square().mean().item()
    metrics = optimize_critic(critic, optimizer, features, targets, epochs=5, minibatches=4)
    after = (critic.forward_features(features) - targets).square().mean().item()
    assert metrics["updates"] == 20 and metrics["skipped_nonfinite"] == 0
    assert after < before
    assert metrics["target_abs_max"] == targets.abs().max().item()


def test_nonfinite_critic_targets_skip_updates_without_parameter_pollution():
    critic = TinyCritic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in critic.parameters()]
    metrics = optimize_critic(critic, optimizer, torch.ones(4, 2),
                              torch.full((4,), float("nan")), epochs=1, minibatches=1)
    assert metrics["updates"] == 0 and metrics["skipped_nonfinite"] == 1
    for parameter, expected in zip(critic.parameters(), before):
        torch.testing.assert_close(parameter, expected)


def test_critic_targets_are_fixed_at_rollout_time_and_use_current_frame_context():
    class Motion:
        data = {"q_arm": torch.arange(4).float()[:, None].repeat(1, 14)}

        def goal_at(self, frame):
            return frame.float()[:, None, None].repeat(1, 10, 90)

    critic = PrivilegedCritic()
    rollout = {
        "states_detached": torch.zeros(2, 1, 42),
        "frames": torch.tensor([[0], [1]]),
        "terminal_state_detached": torch.zeros(1, 42),
        "terminal_frame": torch.tensor([2]),
        "imagined_rewards": torch.tensor([[1.0], [2.0]]),
        "termination_probabilities": torch.tensor([[0.1], [0.2]]),
    }
    features, targets, bootstrap = critic_td_lambda_batch(critic, Motion(), rollout)
    assert features.shape == (2, 146) and targets.shape == (2,)
    assert features[0, 42] == 0 and features[1, 42] == 1
    assert features[0, 56] == 0 and features[1, 56] == 1
    with torch.no_grad():
        values = torch.cat((critic.forward_features(features).reshape(2, 1), bootstrap[None]), dim=0)
    expected = lambda_returns(rollout["imagined_rewards"], values,
                              rollout["termination_probabilities"]).flatten()
    torch.testing.assert_close(targets, expected)
