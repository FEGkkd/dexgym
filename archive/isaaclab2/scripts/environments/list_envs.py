"""List task IDs registered by the DexterousHands extension."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source"))

import gymnasium as gym

import dexteroushands_lab.tasks as dexteroushands_tasks

dexteroushands_tasks.register_tasks()

for spec in sorted(gym.registry.values(), key=lambda item: item.id):
    if spec.id.startswith("DexterousHands-"):
        print(spec.id)
