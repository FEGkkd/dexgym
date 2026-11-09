"""Record outcomes for privileged replay against the live task's goal rules."""

import torch


class GoalReplayAudit:
    def __init__(self, num_envs: int, device: str | torch.device):
        self.first_within_step = torch.zeros(num_envs, dtype=torch.long, device=device)

    def record_step(
        self, *, span: torch.Tensor, age: torch.Tensor, within: torch.Tensor,
        reached: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor,
        fail_frames: torch.Tensor, fail_budget: torch.Tensor, dwell_required: torch.Tensor,
    ) -> list[dict]:
        elapsed = age + 1
        first = within & (self.first_within_step == 0)
        self.first_within_step[first] = elapsed[first]
        finished = reached | terminated | truncated
        ids = torch.nonzero(finished).flatten()
        if not len(ids):
            return []
        data = torch.stack((
            ids, span[ids], elapsed[ids], self.first_within_step[ids],
            fail_frames[ids], dwell_required[ids], reached[ids].long(),
            terminated[ids].long(), truncated[ids].long(),
            (fail_frames[ids] > fail_budget[ids]).long(),
        ), dim=1).cpu().tolist()
        budgets = fail_budget[ids].cpu().tolist()
        events = []
        for row, budget in zip(data, budgets):
            env_id, goal_span, steps, first_step, failed, dwell, success, terminal, cutoff, exceeded = row
            outcome = (
                "reached" if success else "budget_timeout" if terminal and exceeded
                else "other_termination" if terminal else "episode_truncation"
            )
            events.append(dict(
                env_id=env_id, span=goal_span, outcome=outcome, elapsed_steps=steps,
                first_within_step=first_step or None, fail_frames=failed,
                fail_budget=budget, dwell_required=dwell,
            ))
        self.first_within_step[ids] = 0
        return events
