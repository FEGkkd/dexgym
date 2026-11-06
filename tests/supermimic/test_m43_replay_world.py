"""Generic replay preserves transition time, task identity and short episodes."""

import pytest
import torch

from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.models import StateDeltaWorldModel
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, RobotSpec, TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState, TransitionBatch


ROBOT = RobotSpec("toy", 3, 1, 3, 1, "toy_v1", "state3_v1")
TASK = TaskSpec("reach", 4, 2, 2, 1, 2, 5, "goal2_v1")


def transition(step: int, *, done=(False, False), motion=(0, 0), task=(1, 1)) -> TransitionBatch:
    frame = torch.full((2,), step, dtype=torch.long)
    ids = torch.tensor(task)
    motions = torch.tensor(motion)
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state = torch.full((2, 3), float(step))
    action = torch.full((2, 1), 0.1)
    task_state = TaskState(ids, motions, frame, target)
    next_task_state = TaskState(ids, motions, frame + 1, target)
    terminated = torch.tensor(done)
    return TransitionBatch(
        state=state, action=action, previous_action=torch.zeros_like(action),
        executed_command=action, next_state=state + 1,
        history=torch.full((2, 2, 4), float(step)),
        dynamics_context=torch.ones(2, 1), task_state=task_state,
        next_task_state=next_task_state, terminated=terminated,
        truncated=torch.zeros(2, dtype=torch.bool),
        source_id=torch.zeros(2, dtype=torch.int8),
        failure_reason=torch.zeros(2, dtype=torch.int32),
    )


@pytest.mark.parametrize("valid_length", [1, 2, 3, 5, 8])
def test_masked_replay_uses_terminal_transition_without_crossing_reset(valid_length):
    replay = SchemaReplay(16, 2, ROBOT, TASK, "cpu")
    for step in range(10):
        replay.append(transition(step, done=(step == valid_length - 1, False)))
    window = replay.window_at(torch.tensor([0]), torch.tensor([0]), 8)
    assert int(window.mask.sum()) == valid_length
    assert bool(window.terminated[0, valid_length - 1])
    if valid_length < 8:
        assert not bool(window.mask[0, valid_length:].any())
    assert window.task_state.data.shape == (1, 8, 2)
    assert window.dynamics_context.shape == (1, 8, 1)


def test_partial_reset_task_change_and_wraparound_do_not_stitch_windows():
    replay = SchemaReplay(3, 2, ROBOT, TASK, "cpu")
    replay.append(transition(0, done=(True, False)))
    replay.append(transition(1, motion=(1, 0)))
    replay.append(transition(2, motion=(1, 0)))
    window = replay.window_at(torch.tensor([0, 0]), torch.tensor([0, 1]), 3)
    assert window.mask[0].tolist() == [1.0, 0.0, 0.0]
    assert window.mask[1].tolist() == [1.0, 1.0, 1.0]
    replay.append(transition(3, motion=(1, 0), task=(2, 1)))
    window = replay.window_at(torch.tensor([1]), torch.tensor([0]), 3)
    assert window.mask[0].tolist() == [1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="overwritten"):
        replay.window_at(torch.tensor([0]), torch.tensor([1]), 2)


def test_replay_schema_and_restored_episode_counter_are_strict():
    replay = SchemaReplay(4, 2, ROBOT, TASK, "cpu")
    replay.append(transition(0))
    restored = SchemaReplay(4, 2, ROBOT, TASK, "cpu")
    restored.load_state_dict(replay.state_dict())
    assert restored.steps == 1
    torch.testing.assert_close(restored.state[0], replay.state[0])
    torch.testing.assert_close(restored.episode_counter, replay.episode_counter + 1)
    other = SchemaReplay(4, 2, ROBOT, TaskSpec("other", 4, 2, 3, 1, 3, 5, "goal3"), "cpu")
    with pytest.raises(ValueError, match="schema"):
        other.load_state_dict(replay.state_dict())


def test_transition_rejects_wrong_next_time_and_nonfinite_state():
    batch = transition(4)
    batch.next_task_state.frame[0] = 6
    with pytest.raises(ValueError, match="frame"):
        batch.validate(ROBOT, TASK, num_envs=2)
    batch = transition(4)
    batch.next_state[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        batch.validate(ROBOT, TASK, num_envs=2)


def test_world_model_gets_gradient_without_any_full_eight_step_episode():
    class ToyRobot:
        spec = ROBOT

        @staticmethod
        def integrate(state, physical_delta, action, context):
            return state + physical_delta

        @staticmethod
        def physical_delta(state, next_state):
            return next_state - state

    replay = SchemaReplay(16, 2, ROBOT, TASK, "cpu")
    for step in range(7):
        batch = transition(step, done=(True, True))
        batch.next_state = batch.state + torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        replay.append(batch)
    model = StateDeltaWorldModel(ROBOT, NetworkSpec(world_hidden=(16,), goal_hidden=(8,),
                                                   actor_hidden=(8,), critic_hidden=(8,), latent_dim=4))
    window = replay.sample_windows(16, 8)
    assert int(window.mask.sum()) == 16
    loss, errors = model.autoregressive_loss(window, ToyRobot())
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    assert errors.shape == (16, 8)
    loss.backward()
    assert model.network[-1].weight.grad is not None
    assert float(model.network[-1].weight.grad.abs().sum()) > 0
