import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig
from dexgym.tasks.supermimic_toy import ToyCollector, ToyGoalTask, ToyRobot


class TwoLengthToyMotion(ToyGoalTask):
    lengths = torch.tensor([12, 40])

    def __init__(self):
        super().__init__("position")

    def initial_task_state(self, num_envs, device="cpu"):
        base = super().initial_task_state(num_envs, device)
        return TaskState(base.task_id, torch.arange(num_envs, device=device) % 2,
                         base.frame, base.data)

    def valid_imagination_start(self, task_state, horizon):
        return task_state.frame + horizon <= self.lengths[task_state.motion_id]

    def episode_end(self, task_state):
        return task_state.frame >= self.lengths[task_state.motion_id]


def test_same_trainer_collects_and_trains_mixed_motion_ids_without_episode_splicing():
    torch.manual_seed(43)
    robot, task = ToyRobot(), TwoLengthToyMotion()
    modules = NetworkFactory.build(robot.spec, task.spec, NetworkSpec(
        world_hidden=(32, 32), goal_hidden=(16,), actor_hidden=(32,),
        critic_hidden=(16,), latent_dim=8))
    trainer = SuperMimicTrainer(
        collector=ToyCollector(task, 8),
        replay=SchemaReplay(64, 8, robot.spec, task.spec, "cpu"),
        robot=robot, task=task, modules=modules,
        optimizers={"world_model": torch.optim.Adam(modules.world_model.parameters(), lr=1e-3),
                    "actor": torch.optim.Adam(modules.actor.parameters(), lr=1e-3),
                    "critic": torch.optim.Adam(modules.critic.parameters(), lr=1e-3)},
        trust_evaluator=ModelTrustEvaluator(TrustConfig(
            min_distinct_starts=1, max_rmse=(0.5, 0.8, 1.5),
            require_baseline_improvement=False)),
        config=TrainerConfig(world_batch=16, imagination_batch=8,
                             world_warmup_updates=2, trust_interval=1),
        train_env_range=(0, 6), validation_env_range=(6, 8))
    for _ in range(45):
        trainer.run_iteration()
    assert trainer.counts["world_updates"] == 45
    assert trainer.counts["actor_updates"] > 0
    for motion_id in (0, 1):
        sample = trainer.replay.sample_windows(12, 8, motion_id=motion_id)
        assert (sample.task_state.motion_id[:, 0] == motion_id).all()
        assert (sample.task_state.frame[:, 0] < task.lengths[motion_id]).all()
        assert ((sample.task_state.frame[:, 0] + sample.mask.sum(1).long()) <=
                task.lengths[motion_id]).all()
        start = trainer.replay.sample_starts(
            8, horizon=8, valid_start=task.valid_imagination_start,
            motion_id=motion_id)
        assert start.state.shape[0] == 8
        assert (start.task_state.motion_id == motion_id).all()
