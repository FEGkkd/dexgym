import torch

from dexgym.algorithms.supermimic.closed_loop import model_actor_rollout


class ToyReference:
    length = 100

    def future_goal_at(self, frame):
        return torch.zeros(len(frame), 10, 90)


class ToyMapping:
    origin = torch.zeros(14)
    scale = torch.ones(14)

    def __call__(self, action, command):
        return command + action


class ToyActor:
    def __call__(self, history, goal):
        return history[:, -1, 28:42] + 0.1


class ToyModel:
    def __call__(self, state, action, mapping):
        out = state.clone()
        out[:, :14] += action
        out[:, 28:42] = mapping(action, state[:, 28:42])
        return out


def test_closed_loop_actor_reacts_to_predicted_history_and_differs_from_open_loop_actions():
    state = torch.zeros(2, 42)
    history = torch.zeros(2, 10, 56)
    frame = torch.zeros(2, dtype=torch.long)
    closed = model_actor_rollout(ToyActor(), ToyModel(), ToyMapping(), ToyReference(),
                                state, history, frame, horizon=3)
    assert closed["kind"] == "policy_closed_loop"
    first = torch.tensor(0.1)
    second = first.tanh() + 0.1
    third = second.tanh() + 0.1
    torch.testing.assert_close(closed["actions"][:, 0, 0], torch.stack((first, second, third)))
    open_loop = model_actor_rollout(ToyActor(), ToyModel(), ToyMapping(), ToyReference(),
                                    state, history, frame, horizon=3,
                                    physical_actions=torch.full((3, 2, 14), 0.1))
    assert open_loop["kind"] == "action_conditioned_open_loop"
    torch.testing.assert_close(open_loop["states"][-1, 0, 0], torch.tensor(0.3))
    assert closed["states"][-1, 0, 0] > open_loop["states"][-1, 0, 0]
