"""Run standard Isaac Lab RSL-RL PPO while auditing latent and executed actions.

Example:
  PYTHONPATH=source /home/ubuntu/dexgym/.venv/bin/python scripts/train_hand21_action_audit.py \
    --task DexGym-Wuji-Hand21-TeleDexter-Direct-v0 --num_envs 4096 \
    --max_iterations 300 --seed 42 --run_name hand21_action_audit_s42 physics=newton_mjwarp
"""

import csv
import sys
from pathlib import Path

from dexgym.algorithms.action_audit import ActionAudit
from isaaclab_rl.entrypoints import dispatch
from isaaclab_rl.entrypoints.backends import train_rsl_rl


def attach_action_audit(runner) -> None:
    """Instrument rollout sampling and iteration logging without changing PPO updates."""
    env = runner.env.unwrapped
    if not env.cfg.teledexter_style:
        raise ValueError("Action audit requires the TELEDEXTER right-hand task")
    audit = ActionAudit(
        action_dim=runner.env.num_actions,
        max_span=env.cfg.curriculum_stride_max,
        device=runner.env.device,
    )
    output = Path(runner.logger.log_dir) / "action_audit_exact_span.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("iteration", "goal_span", *ActionAudit.FIELDS)
    with output.open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()

    original_act = runner.alg.act
    original_log = runner.logger.log

    def act_with_audit(obs):
        actions = original_act(obs)
        latent_mean = runner.alg.actor.output_distribution_params[0]
        audit.record(latent_mean, actions, env.goal_span)
        return actions

    def log_with_audit(*args, **kwargs):
        rows = audit.drain(iteration=kwargs["it"])
        with output.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writerows(rows)
        original_log(*args, **kwargs)

    runner.alg.act = act_with_audit
    runner.logger.log = log_with_audit
    print(f"[ACTION_AUDIT] per-iteration exact-span counts: {output}", flush=True)


def main() -> int:
    create_runner = train_rsl_rl.create_rsl_rl_runner

    def audited_runner(*args, **kwargs):
        runner = create_runner(*args, **kwargs)
        attach_action_audit(runner)
        return runner

    train_rsl_rl.create_rsl_rl_runner = audited_runner
    return dispatch.run_train_cli(["--rl_library", "rsl_rl", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
