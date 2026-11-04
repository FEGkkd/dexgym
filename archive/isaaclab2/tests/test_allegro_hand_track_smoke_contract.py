from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.environments.smoke_allegro_hand_track import (  # noqa: E402
    assert_finite_step_tensors,
    build_parser,
)


def test_smoke_parser_defaults_to_allegro_task():
    args = build_parser().parse_args([])
    assert args.task == "DexterousHands-Allegro-Hand-Track-Direct-v0"
    assert args.num_envs == 1
    assert args.steps == 8


def test_smoke_finite_helper_rejects_nan():
    assert_finite_step_tensors(
        {"policy": torch.zeros(1, 2)},
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    with pytest.raises(RuntimeError, match="observation"):
        assert_finite_step_tensors(
            {"policy": torch.tensor([[float("nan")]])},
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
        )
