from __future__ import annotations

import os
from pathlib import Path

import pytest


REQUIRED_TAGS = {
    "tracking/hand_error",
    "tracking/object_pos_error",
    "control/action_norm",
    "tracking/success",
}


def tensorboard_scalar_tags(run_dir: str | Path) -> set[str]:
    """Read scalar tags from an RSL-RL TensorBoard run directory."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = Path(run_dir)
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event file found under {run_dir}")
    tags: set[str] = set()
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags.update(accumulator.Tags().get("scalars", []))
    return tags


@pytest.mark.integration
def test_short_ppo_run_contains_tracking_scalars():
    run_dir = os.environ.get("DEXTRACK_RUN_DIR")
    if not run_dir:
        pytest.skip("set DEXTRACK_RUN_DIR after a short RSL-RL run")
    tags = tensorboard_scalar_tags(run_dir)
    assert REQUIRED_TAGS.issubset(tags)
    assert any(tag.startswith("termination/") for tag in tags)
