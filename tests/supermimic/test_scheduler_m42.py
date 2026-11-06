import torch

from dexgym.algorithms.supermimic.scheduling import step_schedulers_after_updates


def test_scheduler_advances_only_after_its_optimizer_updates():
    models = {name: torch.nn.Linear(1, 1) for name in ("actor", "world_model", "critic")}
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=0.1)
                  for name, model in models.items()}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
                  for name, optimizer in optimizers.items()}
    before = {"world_updates": 0, "actor_updates": 0, "critic_updates": 0}
    after = {"world_updates": 1, "actor_updates": 0, "critic_updates": 0}
    optimizers["world_model"].step()
    step_schedulers_after_updates(schedulers, before, after, max_steps=3)
    assert schedulers["world_model"].last_epoch == 1
    assert schedulers["actor"].last_epoch == 0
    assert schedulers["critic"].last_epoch == 0
    for name in ("actor", "world_model", "critic"):
        optimizers[name].step()
    later = {"world_updates": 2, "actor_updates": 1, "critic_updates": 4}
    step_schedulers_after_updates(schedulers, after, later, max_steps=3)
    assert [schedulers[name].last_epoch for name in ("actor", "world_model", "critic")] == [1, 2, 1]
    step_schedulers_after_updates(schedulers, later, later, max_steps=3)
    assert schedulers["actor"].last_epoch == 1
