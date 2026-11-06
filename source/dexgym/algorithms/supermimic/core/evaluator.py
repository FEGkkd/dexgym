"""Independent, task-owned physical metrics over real pre-reset transitions."""

import torch


class PhysicalEvaluator:
    def __init__(self, collector_factory, task):
        self.collector_factory = collector_factory
        self.task = task

    @torch.no_grad()
    def evaluate(self, actor, *, steps: int):
        if steps < 1:
            raise ValueError("Evaluation needs positive physical steps")
        collector = self.collector_factory()  # Fresh physical reset and history.
        was_training = actor.training
        actor.eval()
        try:
            trajectory = [collector.step(actor, policy="actor") for _ in range(steps)]
        finally:
            actor.train(was_training)
        report = {"physical_steps": steps,
                  "physical_transitions": steps * collector.num_envs,
                  "reward_mean": float(torch.stack([item.reward for item in trajectory]).mean()),
                  "hard_terminations": sum(int(item.terminated.sum()) for item in trajectory),
                  "truncations": sum(int(item.truncated.sum()) for item in trajectory),
                  "failure_reason_counts": {}}
        reasons = torch.cat([item.failure_reason.flatten() for item in trajectory])
        for value in reasons.unique():
            if int(value):
                report["failure_reason_counts"][str(int(value))] = int((reasons == value).sum())
        report.update(self.task.evaluate_metrics(trajectory))
        return report
