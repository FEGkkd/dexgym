"""Label one real one-step Newton action probe for masked dynamics learning."""

import torch

from dexgym.algorithms.supermimic.core.transition import TransitionBatch


def make_probe_transition(robot, task, *, state, action, previous_action,
                          next_state, history, task_state, dynamics_context,
                          terminated, failure_reason):
    """End each probe after its physical transition without inventing dynamics.

    ``truncated`` marks the deliberate one-step experiment boundary, while
    ``terminated`` retains any real Newton safety failure. The goal witness
    joint angles are deliberately absent from this interface and Replay.
    """
    next_task_state = task.advance(task_state)
    transition = TransitionBatch(
        state=state, action=action, previous_action=previous_action,
        executed_command=robot.executed_command(state, action),
        next_state=next_state, history=history,
        dynamics_context=dynamics_context, task_state=task_state,
        next_task_state=next_task_state, terminated=terminated,
        truncated=~terminated,
        source_id=torch.full_like(terminated, 4, dtype=torch.int8),
        failure_reason=failure_reason,
        reward=task.reward(state, next_state, action, next_task_state,
                           previous_action),
    )
    transition.validate(robot.spec, task.spec, num_envs=len(state))
    return transition
