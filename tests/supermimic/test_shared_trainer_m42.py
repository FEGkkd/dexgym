import torch

from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.algorithms.supermimic.trainer import SuperMimicTrainer, TrainerConfig


class ToyCollector:
    def __init__(self, replay):
        self.replay = replay
        self.calls = []

    def step(self, *, policy):
        self.calls.append(policy)
        frame = torch.tensor([1, 2])
        state = torch.zeros(2, 42)
        action = torch.zeros(2, 14)
        self.replay.append(state, action, state, torch.zeros(2, 10, 56), frame,
                           torch.zeros(2, dtype=torch.bool),
                           source_id=torch.full((2,), 0 if policy == "actor" else 1))
        return {"reward_mean": 0.0}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def update_norms(self, *args):
        pass

    def autoregressive_loss(self, windows, mapping, **kwargs):
        self.last_windows = windows
        return self.weight.square(), torch.ones_like(windows["mask"])


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def update_norms(self, *args):
        pass


class ToyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def update_norms(self, *args):
        pass

    def forward_features(self, features):
        return features[:, 0] * self.weight


class ToyTask:
    termination_mode = "analytic"
    thresholds = None

    def __init__(self):
        self.reference = type("Reference", (), {"length": 100,
                         "data": {"q_arm": torch.zeros(100, 14)}})()

    def future_goal(self, frame):
        return torch.zeros(len(frame), 10, 90)

    def imagine(self, actor, world_model, critic, start, *, horizon):
        assert horizon == 1
        assert not world_model.weight.requires_grad
        assert not critic.weight.requires_grad
        return {"actor_loss": actor.weight.square()}

    def critic_batch(self, critic, rollout):
        return torch.ones(4, 146), torch.ones(4), torch.zeros(4)


class ToyTrust:
    def evaluate_available(self, *args, **kwargs):
        assert kwargs["source_id"] == 0
        return {"1": {"distinct_start_count": 8, "joint_position_rmse_rad": 0.0,
                      "tcp_position_mean_mm": 0.0, "constant_velocity_tcp_mean_mm": 10.0}}


def test_strict_pure_starts_world_actor_critic_without_teacher_supervision():
    replay = MotionReplay(16, 2, "cpu")
    collector = ToyCollector(replay)
    world, actor, critic = ToyModel(), ToyActor(), ToyCritic()
    trainer = SuperMimicTrainer(
        collector=collector, replay=replay, robot=type("Robot", (), {"action_map": None})(),
        task=ToyTask(), world_model=world, actor=actor, critic=critic,
        world_optimizer=torch.optim.SGD(world.parameters(), lr=0.01),
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.01),
        critic_optimizer=torch.optim.SGD(critic.parameters(), lr=0.01),
        trust_evaluator=ToyTrust(),
        config=TrainerConfig(world_warmup_updates=1, world_batch=2, imagination_batch=2,
                             critic_epochs=1, critic_minibatches=1, trust_interval=1))
    report = trainer.run_iteration()
    assert collector.calls == ["actor"]
    assert report["counts"]["world_updates"] == 1
    assert report["counts"]["actor_updates"] == 1
    assert report["counts"]["critic_updates"] == 1
    assert world.weight.item() < 1 and actor.weight.item() < 1


def test_strict_pure_rejects_reference_pd_bootstrap():
    try:
        TrainerConfig(mode="pure_model_based", reference_pd_bootstrap_steps=1)
    except ValueError:
        return
    assert False, "Strict pure mode accepted a reference PD collector"


def test_learned_head_uses_arrival_reference_once_even_at_final_clip_transition():
    replay = MotionReplay(8, 2, "cpu")
    state = torch.zeros(2, 42)
    replay.append(state, torch.zeros(2, 14), state, torch.zeros(2, 10, 56),
                  torch.tensor([98, 98]), torch.tensor([True, True]),
                  terminated=torch.tensor([True, True]))

    class LearnedTask(ToyTask):
        termination_mode = "learned"

        def arrival_goal(self, frame):
            if bool((frame + 1 >= self.reference.length).any()):
                raise ValueError("Double arrival-frame offset")
            return (frame + 1).float()[:, None].expand(-1, 90)

    world, actor, critic = ToyModel(), ToyActor(), ToyCritic()
    trainer = SuperMimicTrainer(
        collector=ToyCollector(replay), replay=replay,
        robot=type("Robot", (), {"action_map": None})(), task=LearnedTask(),
        world_model=world, actor=actor, critic=critic,
        world_optimizer=torch.optim.SGD(world.parameters(), lr=0.01),
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.01),
        critic_optimizer=torch.optim.SGD(critic.parameters(), lr=0.01),
        trust_evaluator=ToyTrust(), config=TrainerConfig(world_batch=2))
    assert trainer.update_world_model()["updated"]
    torch.testing.assert_close(world.last_windows["termination_goal"][:, 0, 0], torch.tensor([99.0, 99.0]))
