"""Controls for matching event timing, contact labels, and quantitative boundary limits."""

from types import SimpleNamespace

import torch

from dexgym.tasks.direct.wholebody_s05.v1.task.boundary_parity import json_finite, pair_case


def fk(q):
    return SimpleNamespace(tips=q[:, None, :3])


def fixture():
    return {
        "pass": True, "events": [64], "substeps": [1], "contact": [[0., 1., 1.]],
        "depth_m": [.001], "fallback": [False], "terminated": [True],
    }


def test_identical_boundary_physics_and_stop_are_accepted():
    state = torch.zeros((1, 108))
    assert pair_case(fixture(), fixture(), state, state, fk)["pass"]


def test_equal_position_cannot_mask_event_timing_or_contact_drift():
    state = torch.zeros((1, 108))
    native = fixture()
    for altered in ({"events": [0]}, {"substeps": [2]}, {"contact": [[0., 1., 0.]]}):
        result = pair_case(native, dict(native, **altered), state, state, fk)
        assert not result["discrete_exact_match"]
        assert not result["pass"]


def test_matching_events_do_not_bypass_physical_or_nonfinite_limits():
    state = torch.zeros((1, 108))
    altered_state = state.clone()
    altered_state[0, 54] = .051
    assert not pair_case(fixture(), fixture(), state, altered_state, fk)["pass"]
    result = pair_case(fixture(), dict(fixture(), depth_m=[float("nan")]), state, state, fk)
    assert not result["pass"]
    assert json_finite(result)["maxima"]["contact_depth_m"] is None
