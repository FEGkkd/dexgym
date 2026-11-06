"""Episode-aware physical evaluation of pre-reset real transitions."""

from copy import copy
from pathlib import Path

import torch

from .transition import TaskState


def _one_env(item, env_id: int, num_envs: int):
    """Preserve task metric tensor shapes while isolating one physical episode."""
    selected = copy(item)
    for name, value in vars(item).items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == num_envs:
            setattr(selected, name, value[env_id:env_id + 1].detach().clone())
        elif isinstance(value, TaskState):
            setattr(selected, name, value.index(slice(env_id, env_id + 1)))
    return selected


class PhysicalEvaluator:
    def __init__(self, collector_factory, task, *, evaluation_kind="inprocess_smoke"):
        if evaluation_kind not in ("inprocess_smoke", "independent_physics_evaluation"):
            raise ValueError("Unknown physical evaluation kind")
        self.collector_factory = collector_factory
        self.task = task
        self.evaluation_kind = evaluation_kind

    @torch.no_grad()
    def evaluate(self, actor, *, steps: int, trace_path: str | Path | None = None,
                 one_episode_per_env: bool = False):
        if steps < 1:
            raise ValueError("Evaluation needs positive physical steps")
        collector = self.collector_factory()
        if getattr(collector, "action_noise_std", 0.0) != 0:
            raise ValueError("Physical evaluation must disable exploration noise")
        if not callable(getattr(self.task, "success_mask", None)):
            raise ValueError("Task must define success_mask for episode evaluation")
        num_envs = collector.num_envs
        was_training = actor.training
        actor.eval()
        pending = [[] for _ in range(num_envs)]
        episode_ids = [0] * num_envs
        episodes = []
        raw_trace = []
        evaluated_masks = []
        inactive = set()
        evaluated_transitions = 0
        reward_sum = 0.0
        reason_counts = {}
        hard_terminations = truncations = 0
        hold_steps = int(getattr(self.task, "success_hold_steps", 1))
        if hold_steps < 1:
            raise ValueError("Task success_hold_steps must be positive")

        def close_episode(env_id, *, complete):
            items = pending[env_id]
            if not items:
                return
            success_steps = [bool(self.task.success_mask(item)[0]) for item in items]
            first_success = next((index + 1 for index, value in enumerate(success_steps) if value), None)
            streak = longest = 0
            for value in success_steps:
                streak = streak + 1 if value else 0
                longest = max(longest, streak)
            last = items[-1]
            failed = bool(last.terminated[0])
            truncated = bool(last.truncated[0])
            metrics = self.task.evaluate_metrics(items)
            record = {
                "env_id": env_id, "episode_id": episode_ids[env_id],
                "episode_start_step": global_step - len(items) + 1,
                "episode_length": len(items),
                "episode_return": float(sum(float(item.reward[0]) for item in items)),
                "complete": complete, "success": longest >= hold_steps,
                "first_success_step": first_success,
                "first_success_time_s": (
                    first_success * collector.robot.spec.control_dt
                    if first_success is not None and hasattr(collector, "robot") else None),
                "first_failure_step": len(items) if failed else None,
                "terminal_reason": int(last.failure_reason[0]) if failed else None,
                "truncation_reason": (self.task.truncation_reason(last)
                                      if truncated and callable(getattr(self.task, "truncation_reason", None))
                                      else "time_limit" if truncated else None),
                "goal_hold_duration": longest,
                "metrics": metrics,
            }
            episodes.append(record)
            pending[env_id] = []
            episode_ids[env_id] += 1

        try:
            for global_step in range(1, steps + 1):
                batch = collector.step(actor, policy="actor")
                raw_trace.append(batch)
                evaluated_masks.append(torch.tensor(
                    [env_id not in inactive for env_id in range(num_envs)], dtype=torch.bool))
                for env_id in range(num_envs):
                    if env_id in inactive:
                        continue
                    item = _one_env(batch, env_id, num_envs)
                    evaluated_transitions += 1
                    reward_sum += float(item.reward[0])
                    hard_terminations += int(item.terminated[0])
                    truncations += int(item.truncated[0])
                    pending[env_id].append(item)
                    reason = int(item.failure_reason[0])
                    if reason:
                        key = str(reason)
                        reason_counts[key] = reason_counts.get(key, 0) + 1
                    if bool(item.terminated[0] or item.truncated[0]):
                        close_episode(env_id, complete=True)
                        if one_episode_per_env:
                            inactive.add(env_id)
                if len(inactive) == num_envs:
                    break
            for env_id in range(num_envs):
                close_episode(env_id, complete=False)
        finally:
            actor.train(was_training)
        if trace_path is not None:
            trace_path = Path(trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"evaluation_kind": self.evaluation_kind,
                        "episodes": episodes, "transitions": raw_trace,
                        "evaluated_mask": evaluated_masks}, trace_path)
        report = {
            "evaluation_kind": self.evaluation_kind,
            "physical_steps": global_step, "physical_transitions": global_step * num_envs,
            "evaluated_transitions": evaluated_transitions,
            "reward_mean": reward_sum / evaluated_transitions,
            "hard_terminations": hard_terminations,
            "truncations": truncations,
            "failure_reason_counts": reason_counts,
            "episode_count": len(episodes),
            "completed_episode_count": sum(item["complete"] for item in episodes),
            "episode_success_rate": sum(item["success"] for item in episodes) / len(episodes),
            "episode_failure_rate": sum(item["terminal_reason"] is not None for item in episodes) / len(episodes),
            "episodes": episodes,
        }
        metric_keys = set.intersection(*(set(item["metrics"]) for item in episodes))
        for key in metric_keys:
            values = [item["metrics"][key] for item in episodes]
            try:
                report[key] = torch.as_tensor(values, dtype=torch.float32).mean(0).tolist()
            except (TypeError, ValueError):
                pass
        return report
