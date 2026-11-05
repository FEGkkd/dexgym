"""Alternate real physics collection and all SuperMimic learning updates."""


def prepare_model_windows(replay, batch, updates, env_range):
    """Reserve only complete episode-safe windows before starting WM updates."""
    try:
        return [replay.sample_windows(batch, 8, env_range=env_range) for _ in range(updates)]
    except ValueError:
        return None


class AlternatingLoop:
    def __init__(self, world_updates_per_iteration=1, actor_updates_per_iteration=1):
        if world_updates_per_iteration < 1 or actor_updates_per_iteration < 1:
            raise ValueError("Aligned learning update counts must be positive")
        self.world_updates_per_iteration = world_updates_per_iteration
        self.actor_updates_per_iteration = actor_updates_per_iteration
        self.counts = {"collections": 0, "world_updates": 0, "actor_updates": 0, "critic_updates": 0}

    def run_one(self, *, collect, replay_ready, model_update, actor_ready, actor_update, critic_update):
        """Execute one collect→model→actor→critic cycle with bounded warmup gates."""
        collect()
        self.counts["collections"] += 1
        if not replay_ready():
            return self.counts.copy()
        for _ in range(self.world_updates_per_iteration):
            model_update()
            self.counts["world_updates"] += 1
        if actor_ready():
            for _ in range(self.actor_updates_per_iteration):
                actor_update()
                self.counts["actor_updates"] += 1
                critic_update()
                self.counts["critic_updates"] += 1
        return self.counts.copy()
