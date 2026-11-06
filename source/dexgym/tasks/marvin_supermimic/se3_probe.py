"""Label one real one-step Newton action probe for masked dynamics learning."""

from dataclasses import asdict
import torch

from dexgym.algorithms.supermimic.core.transition import TransitionBatch


def validate_probe_metadata(metadata: dict, robot, task):
    if (metadata.get("kind") != "strict_pure_newton_paired_action_probe"
            or metadata.get("teacher_or_ik_used") is not False
            or metadata.get("goal_joint_witness_stored") is not False):
        raise ValueError("Probe data must be physical and free of teacher or IK labels")
    if metadata.get("robot") != asdict(robot.spec):
        raise ValueError("Probe Robot dynamics contract differs")
    if metadata.get("task") != asdict(task.spec):
        raise ValueError("Probe Task schema differs")
    if metadata.get("robot_contract") != robot.checkpoint_contract():
        raise ValueError("Probe Robot execution contract differs")
    if metadata.get("task_contract") != task.checkpoint_contract():
        raise ValueError("Probe Task execution contract differs")


def paired_probe_actions(num_envs: int, action_dim: int, *, epsilon: float,
                         device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair legal raw-action perturbations around zero at independent starts."""
    if num_envs < 1 or action_dim < 1 or not 0 < epsilon <= 1:
        raise ValueError("Probe count, action dimension or amplitude is invalid")
    group_size = 2 * action_dim + 2
    env = torch.arange(num_envs, device=device)
    offset = env % group_size
    actions = torch.zeros((num_envs, action_dim), device=device)
    positive = (offset >= 1) & (offset <= action_dim)
    negative = (offset > action_dim) & (offset <= 2 * action_dim)
    joint = torch.where(positive, offset - 1, offset - action_dim - 1)
    active = positive | negative
    actions[env[active], joint[active]] = torch.where(
        positive[active], epsilon, -epsilon)
    return actions, env // group_size


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
