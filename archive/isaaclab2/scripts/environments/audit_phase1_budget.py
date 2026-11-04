"""Audit logged Phase 1 PPO transitions without double-counting resumed runs.

This is a lower bound on all environment transitions: deterministic
evaluations, smoke tests, physical probes, and interrupted unlogged rollouts
are not included. Run with the project's Isaac/RSL-RL Python environment,
which provides TensorBoard and PyYAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import yaml
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_RUNS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "logs/rsl_rl/dexteroushands_bimanual_bottle_phase1"
)


def logged_updates(run_dir: Path) -> tuple[int, list[str]]:
    """Count distinct logger iterations for this run, including resumes."""
    steps: set[int] = set()
    paths = sorted(run_dir.glob("events.out*"))
    if not paths:
        raise ValueError(f"No TensorBoard event file: {run_dir}")
    for path in paths:
        events = event_accumulator.EventAccumulator(
            str(path), size_guidance={"scalars": 0})
        events.Reload()
        tags = events.Tags()["scalars"]
        tag = next((name for name in tags if name.startswith("phase1/")),
                   tags[0] if tags else None)
        if tag is None:
            raise ValueError(f"No scalar iteration records: {path}")
        steps.update(item.step for item in events.Scalars(tag))
    if any(step < 0 for step in steps):
        raise ValueError(f"Negative logged iteration: {run_dir}")
    return len(steps), [str(path) for path in paths]


def num_envs_from_saved_cfg(path: Path) -> int:
    # Isaac's saved YAML includes !!python/tuple tags. Only read this
    # simple scalar; do not execute arbitrary tagged YAML constructors.
    matches = re.findall(r"^\s*num_envs:\s*(\d+)\s*$",
                         path.read_text(), flags=re.MULTILINE)
    if len(matches) != 1 or int(matches[0]) < 1:
        raise ValueError(f"Expected one positive scene.num_envs in {path}")
    return int(matches[0])


def audit(runs_root: Path, budget: int) -> dict:
    if budget <= 0 or not runs_root.is_dir():
        raise ValueError("Positive budget and existing run directory required")
    rows = []
    for run_dir in sorted(runs_root.iterdir()):
        agent_path = run_dir / "params/agent.yaml"
        env_path = run_dir / "params/env.yaml"
        if not agent_path.is_file():
            continue
        if not env_path.is_file():
            raise ValueError(f"Training run lacks saved environment config: {run_dir}")
        agent = yaml.safe_load(agent_path.read_text())
        rollout = agent.get("num_steps_per_env")
        if not isinstance(rollout, int) or rollout < 1:
            raise ValueError(f"Invalid rollout length in {agent_path}")
        environments = num_envs_from_saved_cfg(env_path)
        updates, event_files = logged_updates(run_dir)
        rows.append({
            "run": run_dir.name,
            "num_envs": environments,
            "rollout_steps_per_env": rollout,
            "logged_updates": updates,
            "logged_ppo_transitions": environments * rollout * updates,
            "event_files": event_files,
        })
    if not rows:
        raise ValueError("No logged Phase 1 PPO runs found")
    total = sum(row["logged_ppo_transitions"] for row in rows)
    return {
        "schema": "dexgym-phase1-logged-ppo-budget-audit-v1",
        "runs_root": str(runs_root),
        "budget_environment_transitions": budget,
        "logged_ppo_transitions": total,
        "upper_bound_remaining_total_transitions": max(0, budget - total),
        "all_transitions_exhausted_proven": total >= budget,
        "limitations": [
            "Deterministic evaluation, smoke and physical-probe transitions are omitted.",
            "Interrupted rollouts without logger records are omitted.",
            "The reported remaining count is only an upper bound, not spendable PPO budget.",
        ],
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--budget", type=int, default=100_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.runs_root, args.budget)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PHASE1_BUDGET_AUDIT", json.dumps({
        key: result[key] for key in (
            "logged_ppo_transitions", "upper_bound_remaining_total_transitions",
            "all_transitions_exhausted_proven")
    }))


if __name__ == "__main__":
    main()
