import pytest
import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES, tracking_reward
from dexgym.motion.library import MotionLibrary
from dexgym.tasks.marvin_supermimic.core_plugins import (
    MarvinCoreRobot, MarvinMotionTracking, MarvinDualTCPReach)


WAVE = "/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/cmu_141_16_wave_bimanual.npz"
PROCEDURAL = "/home/ubuntu/dexgym_data/marvin_supermimic_m42/second_task/procedural_bimanual_reach_v1.npz"


@pytest.fixture(scope="module")
def plugins():
    library = MotionLibrary({"cmu_141_16_wave_bimanual": WAVE,
                             "procedural_bimanual_reach_v1": PROCEDURAL}, "cpu")
    robot = MarvinCoreRobot(library.clips[0], "cpu", action_schema="robot_fixed_v3")
    return library, robot


def test_marvin_contract_and_tracking_reward_parity(plugins):
    library, robot = plugins
    task = MarvinMotionTracking(library, robot, reward_profile="marvin_tcp_priority")
    assert (robot.spec.state_dim, robot.spec.action_dim, robot.spec.delta_dim) == (42, 14, 28)
    assert (task.spec.observation_dim, task.spec.goal_dim, task.spec.critic_dim) == (56, 90, 146)
    frame = torch.tensor([0, 20])
    motion = torch.tensor([0, 1])
    state = torch.cat((library.reference_at(motion, frame)["q_arm"],
                       torch.zeros(2, 14), library.reference_at(motion, frame)["q_arm"]), -1)
    action = torch.zeros(2, 14)
    delta = torch.full((2, 28), 0.0001)
    next_state = robot.integrate(state, delta, action, torch.empty(2, 0))
    assert torch.allclose(next_state[:, 28:], robot.action_map(action, state[:, 28:]))
    task_state = TaskState(torch.zeros(2, dtype=torch.long), motion, frame, torch.empty(2, 0))
    arrival = task.advance(task_state)
    actual = task.reward(state, next_state, action, arrival)
    ref = library.reference_at(motion, frame + 1)
    expected, _ = tracking_reward(next_state[:, :14], state[:, :14],
                                  next_state[:, 28:42], state[:, 28:42],
                                  {key: ref[key] for key in ("link_pos", "link_rotmat",
                                                              "link_linear_vel", "link_angular_vel")},
                                  robot.fk, robot.action_map.lower, robot.action_map.upper,
                                  weights=REWARD_PROFILES["marvin_tcp_priority"],
                                  bounded_action=action.tanh(),
                                  previous_bounded_action=action.tanh())
    assert torch.allclose(actual, expected, atol=1e-7)
    assert task.goal(state, task_state).shape == (2, 10, 90)
    assert task.goal(state, task_state)[0, 0].equal(library.reference_features_at(motion[:1], frame[:1] + 1)[0])
    assert torch.equal(task.valid_imagination_start(task_state, 32), torch.tensor([True, True]))


def test_reach_has_distinct_goal_observation_and_bptt(plugins):
    library, robot = plugins
    reach = MarvinDualTCPReach(robot)
    tracking = MarvinMotionTracking(library, robot)
    assert reach.spec.goal_dim == 6 and reach.spec.observation_dim == 48
    assert reach.spec.goal_dim != tracking.spec.goal_dim
    modules = NetworkFactory.build(robot.spec, reach.spec, NetworkSpec(
        world_hidden=(32, 32), goal_hidden=(16,), actor_hidden=(32,),
        critic_hidden=(16,), latent_dim=8))
    q = library.reference_at(torch.tensor([1, 1]), torch.tensor([0, 0]))["q_arm"]
    state = torch.cat((q, torch.zeros_like(q), q), -1)
    task_state = reach.task_state_from_reset(state)
    observation = reach.observation(state, torch.zeros(2, 14), task_state)
    goal = reach.goal(state, task_state)
    assert observation.shape == (2, 48) and goal.shape == (2, 1, 6)
    action = modules.actor(observation[:, None].repeat(1, 4, 1), goal)
    assert action.shape == (2, 14)
    # A learned model must expose action-conditioned physical deltas; a zero
    # delta model and a rate-limited command can legitimately have zero slope.
    physical_delta = torch.cat((0.01 * action, torch.zeros(2, 14)), dim=-1)
    next_state = robot.integrate(state, physical_delta, action, torch.empty(2, 0))
    reward = reach.reward(state, next_state, action, reach.advance(task_state))
    assert reward.shape == (2,) and torch.isfinite(reward).all()
    assert reach.soft_termination(next_state, task_state).shape == (2,)
    assert reach.critic_features(state, task_state).shape == (2, reach.spec.critic_dim)
    assert torch.autograd.grad(reward.sum(), action, retain_graph=True)[0].abs().sum() > 0
