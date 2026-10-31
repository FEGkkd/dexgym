"""Print grouped RSL-RL Gaussian action std from a Phase 1 checkpoint."""

import argparse
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state = checkpoint["actor_state_dict"]
if "distribution.std_param" in state:
    std = state["distribution.std_param"]
    parameterization = "scalar"
elif "distribution.log_std_param" in state:
    std = state["distribution.log_std_param"].exp()
    parameterization = "log"
else:
    raise SystemExit("No homogeneous Gaussian std found in actor checkpoint")
if std.numel() != 52:
    raise SystemExit(f"Expected 52 action std values, got {std.numel()}")
groups = {"left_wrist_translation": std[0:3], "left_wrist_rotation": std[3:6],
          "left_fingers": std[6:26], "right_wrist_translation": std[26:29],
          "right_wrist_rotation": std[29:32], "right_fingers": std[32:52]}
print(json.dumps({"checkpoint": str(args.checkpoint), "iteration": checkpoint.get("iter"),
                  "parameterization": parameterization,
                  "groups": {name: {"mean": float(values.mean()), "min": float(values.min()),
                                    "max": float(values.max())} for name, values in groups.items()}},
                 sort_keys=True))
