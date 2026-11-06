"""Advance each LR schedule only after its optimizer performed a valid update."""


UPDATE_COUNT = {"world_model": "world_updates", "actor": "actor_updates",
                "critic": "critic_updates"}


def step_schedulers_after_updates(schedulers, before_counts, after_counts, *, max_steps):
    for name, scheduler in schedulers.items():
        count_key = UPDATE_COUNT[name]
        if (after_counts[count_key] > before_counts[count_key]
                and scheduler.last_epoch < max_steps):
            scheduler.step()
