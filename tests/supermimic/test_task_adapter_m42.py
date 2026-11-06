import torch

from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from dexgym.motion.motion_manager import MotionManager
from dexgym.tasks.marvin_supermimic.robot_adapter import MarvinArmDynamicsAdapter
from dexgym.tasks.marvin_supermimic.task_adapter import MarvinTrackingTask


def test_tracking_adapter_uses_current_reference_at_arrival_and_same_imagination_core():
    motion = MotionManager("/home/ubuntu/dexgym_data/marvin_motion_library_v1/validated_motions/"
                           "cmu_141_16_wave_bimanual.npz", 1, "cpu")
    robot = MarvinArmDynamicsAdapter(motion, "cpu", action_schema="robot_fixed_v2")
    task = MarvinTrackingTask(motion, robot, reward_profile="paper_multilink",
                              termination_mode="analytic")
    frame = torch.tensor([5])
    torch.testing.assert_close(task.arrival_goal(frame), motion.reference_features_at(frame + 1))
    state = torch.cat((motion.data["q_arm"][5], torch.zeros(14), motion.data["q_arm"][5]))[None]
    start = {"state": state, "history": torch.zeros(1, 10, 56), "frame": frame}
    rollout = task.imagine(GoalActor(), DeltaWorldModel(hidden=32), PrivilegedCritic(), start, horizon=1)
    assert rollout["imagined_rewards"].shape == (1, 1)
    assert rollout["termination_probabilities"].shape == (1, 1)
