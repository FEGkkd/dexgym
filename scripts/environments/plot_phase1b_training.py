"""Render reproducible Phase 1B PPO diagnostics from TensorBoard events."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("log_dirs", type=Path, nargs="+")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--std-output", type=Path, required=True)
parser.add_argument("--reward-output", type=Path)
parser.add_argument("--health-output", type=Path)
args = parser.parse_args()

fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
std_fig, std_axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
reward_fig, reward_axes = plt.subplots(5, 2, figsize=(14, 18), constrained_layout=True)
health_fig, health_axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
panels = (
    ("Train/mean_reward", "Episode reward"),
    ("phase1b/right_contact_rate", "Right cap contact rate"),
    ("phase1b/bilateral_support_rate", "Bilateral stable support rate"),
    ("phase1b/raw_action_element_outside_fraction", "Raw action element outside [-1, 1]"),
    ("Loss/approx_kl", "PPO approximate KL"),
    ("Loss/ppo_clip_fraction", "PPO clip fraction"),
)
reward_panels = tuple((f"reward/{name}_mean", f"{name.title()} reward / step")
                      for name in ("contact", "holding", "stability", "approach",
                                   "progress", "action", "constraint", "success", "failure"))
health_panels = (
    ("phase1b/joint_target_clip_fraction", "Finger target clipping"),
    ("phase1b/wrist_workspace_clip_fraction", "Wrist workspace clipping"),
    ("Loss/policy_mean_element_outside_fraction", "Policy mean outside [-1, 1]"),
    ("Loss/raw_action_element_outside_fraction", "Sampled raw action outside [-1, 1]"),
    ("Loss/entropy", "Actor entropy"),
    ("Loss/value", "Critic value loss"),
    ("Loss/explained_variance", "Explained variance"),
    ("Perf/total_fps", "Environment transitions / second"),
)
for log_dir in args.log_dirs:
    run_name = log_dir.name
    label = ("scratch, 1024" if "seed1101_A_1024" in run_name else
             "contact coverage, 1024" if "contact_coverage" in run_name else
             "warm seed 1101, 1024" if "seed1101" in run_name else
             "warm seed 2202, 2048" if "seed2202" in run_name else run_name)
    files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard events in {log_dir}")
    event = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
    event.Reload()
    tags = set(event.Tags()["scalars"])
    for axis_group, panel_group in ((axes.flat, panels),
                                    (reward_axes.flat, reward_panels),
                                    (health_axes.flat, health_panels)):
        for axis, (tag, title) in zip(axis_group, panel_group):
            if tag not in tags:
                continue
            rows = event.Scalars(tag)
            axis.plot([r.step for r in rows], [r.value for r in rows], label=label)
            axis.set_title(title)
            axis.set_xlabel("PPO update")
            axis.grid(alpha=.25)
            axis.legend(fontsize=7)
    for axis, group in zip(std_axes.flat, (
            "left_wrist_translation", "left_wrist_rotation", "left_finger",
            "right_wrist_translation", "right_wrist_rotation", "right_finger")):
        tag = f"Loss/{group}_std"
        if tag not in tags:
            continue
        rows = event.Scalars(tag)
        axis.plot([r.step for r in rows], [r.value for r in rows], label=label)
        axis.set_title(group.replace("_", " ").title() + " Gaussian std")
        axis.set_xlabel("PPO update")
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=160)
args.std_output.parent.mkdir(parents=True, exist_ok=True)
std_fig.savefig(args.std_output, dpi=160)
if args.reward_output:
    args.reward_output.parent.mkdir(parents=True, exist_ok=True)
    reward_fig.savefig(args.reward_output, dpi=160)
if args.health_output:
    args.health_output.parent.mkdir(parents=True, exist_ok=True)
    health_fig.savefig(args.health_output, dpi=160)
print(args.output)
print(args.std_output)
if args.reward_output:
    print(args.reward_output)
if args.health_output:
    print(args.health_output)
