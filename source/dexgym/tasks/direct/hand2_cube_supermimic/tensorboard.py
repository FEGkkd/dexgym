"""Mirror JSONL training metrics and native evaluations into live TensorBoard events."""

import argparse
import json
import math
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

TAGS = {
    "reward": "train/reward",
    "drop_rate": "train/drop_fraction_per_control_step",
    "rot_error": "train/rotation_error_rad",
    "wm_loss": "world_model/train_loss",
    "holdout/wm_loss": "world_model/holdout_loss",
    "state_loss": "world_model/train_state_loss",
    "holdout/state_loss": "world_model/holdout_state_loss",
    "contact_digit_loss": "world_model/train_contact_loss",
    "holdout/contact_digit_loss": "world_model/holdout_contact_loss",
    "failure_loss": "failure_head/train_loss",
    "holdout/failure_loss": "failure_head/holdout_loss",
    "actor_loss": "actor/loss",
    "critic_loss": "critic/loss",
    "survival": "actor/imagined_survival",
    "lr": "optimization/learning_rate",
    "sim_steps": "data/real_transitions",
    "wall_seconds": "time/elapsed_seconds",
}
EVAL_TAGS = {
    "mean_successes": "eval/goals_per_episode",
    "fraction_at_least_one": "eval/fraction_at_least_one_goal",
    "fraction_at_least_three": "eval/fraction_at_least_three_goals",
    "drop_fraction": "eval/drop_fraction_per_episode",
    "mean_episode_seconds": "eval/episode_duration_seconds",
    "mean_angle_rad": "eval/rotation_error_rad",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    source = args.run_dir / "metrics.jsonl"
    output = args.run_dir / "tensorboard"
    output.mkdir(parents=True, exist_ok=True)
    last = -1
    seen_evaluations = set()
    initial = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    epoch = source.stat().st_mtime - initial[-1]["wall_seconds"] if initial else time.time()
    writer = SummaryWriter(str(output), flush_secs=5)
    try:
        while True:
            for line in source.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A concurrent writer may not have finished its last line.
                step = int(row["iteration"])
                if step <= last:
                    continue
                for key, value in row.items():
                    if key == "iteration" or not isinstance(value, (int, float)) or not math.isfinite(value):
                        continue
                    tag = TAGS.get(key, "gradients/" + key[5:] + "_before_clipping" if key.startswith("grad/") else key)
                    writer.add_scalar(tag, value, step, walltime=epoch + row["wall_seconds"])
                last = step
            if args.evaluations:
                for path in sorted(args.evaluations.glob("eval_iter*_scene_v2_seed123.json")):
                    if path.name in seen_evaluations:
                        continue
                    try:
                        result = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        continue
                    if result.get("checkpoint_kind") != "native":
                        continue
                    if Path(result["checkpoint"]).parent.resolve() != args.run_dir.resolve():
                        continue
                    for key, tag in EVAL_TAGS.items():
                        writer.add_scalar(tag, result[key], int(result["checkpoint_iteration"]))
                    seen_evaluations.add(path.name)
            writer.flush()
            (output / "sync_status.json").write_text(
                json.dumps(
                    dict(
                        last_iteration=last,
                        evaluation_files=sorted(seen_evaluations),
                        updated_unix=time.time(),
                    ),
                    indent=2,
                )
            )
            if not args.follow:
                break
            time.sleep(5)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
