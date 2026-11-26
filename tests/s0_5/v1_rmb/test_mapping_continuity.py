"""Synthetic controls for paired mapping statistics, without IK or physical stepping."""

from types import SimpleNamespace

import numpy as np

from dexgym.tasks.direct.wholebody_s05.v1.rmb import mapping_continuity as audit
from dexgym.tasks.direct.wholebody_s05.v1.rmb.mapping import IntentMapper, smooth_weights


ORDER = ((0, np.arange(1, 21).reshape(5, 4)),) * 2
VERSIONS = ("clock_generation", "calibration_id", "topology_id")


def fixture(distances):
    home = np.zeros((2, 5, 4, 3))
    home[..., 0] = np.array([-.045, 0., .031, .051, .068])[None, :, None]
    home[..., 1] = np.array([.044, .063, .082, .101])[None, None, :]
    rotation = np.tile(np.eye(3), (2, 1, 1))
    calibrated = {"wrist_to_palm_translation": np.zeros((2, 3)),
                  "wrist_to_palm_rotation": rotation, "hand_scale": np.ones(2)}
    config = {"scale": 1, "rotation": np.eye(3), "translation": np.zeros(3),
              "mount": rotation, "pivot": np.zeros(3)}
    fit = {"fit_scope": "Train", "configs": {"M0": config, "M1": config}}

    n = len(distances)
    pose = np.zeros((n, 4, 21, 7))
    pose[..., -1] = 1
    pose[:, :2, 1:, :3] = home.reshape(2, 20, 3)
    pose[:, :2, 8, 0] = home[0, 0, -1, 0] + np.asarray(distances)[:, None]
    pose[:, 2:4, 0, :3] = np.array([[0., 0., .1], [0., .3, .1]])
    pose[:, 2:4, 0, 0] += np.arange(n)[:, None] * .10
    raw = {"pose": pose, "joint_valid": np.ones(n, dtype=bool), "segment_id": np.zeros(n, dtype=int),
           "tick_ns": np.arange(n, dtype=np.int64) * audit.TICK_NS}
    raw.update({name: np.zeros((n, 4), dtype=int) for name in VERSIONS})
    return raw, IntentMapper(SimpleNamespace(local=home), calibrated, fit), (home, calibrated, fit)


def test_paired_crossings_remove_common_motion_and_keep_function_tradeoff(monkeypatch):
    raw, mapper, _ = fixture([.021, .019, .021])
    monkeypatch.setattr(audit, "compiled_topologies", lambda *_: (ORDER, None))
    samples = audit.new_samples()
    counts = audit.collect_partition(raw, {}, np.ones(3, dtype=bool), mapper, samples, version_fields=VERSIONS)
    assert (counts["continuous_pairs"], counts["entry_ticks"], counts["exit_ticks"], counts["multi_pair_ticks"]) == (
        2, 1, 1, 2)
    results = audit.summarize(samples, active_roots=1)
    hard = results[audit.HARD]
    assert 5 < hard["extra_world_tip_step_mm"]["max"] < 20
    assert not hard["extra_world_tip_step_mm"]["within_numeric_limits"]
    assert set(results) == {audit.HARD, *audit.BANDS}
    for name in audit.BANDS:
        row = results[name]
        assert row["extra_world_tip_step_mm"]["max"] < hard["extra_world_tip_step_mm"]["max"]
        assert row["function_on_hard_active_ticks"]["pinch_vector_mm"]["p95"] > .25
        assert not row["function_on_hard_active_ticks"]["pinch_vector_mm"]["within_numeric_limit"]
        assert not row["function_on_hard_active_ticks"]["pinch_vector_mm"]["adequate_support"]
        assert row["function_on_hard_active_ticks"]["relative_wrist_deg"]["p95"] < 1e-12


def test_boundaries_do_not_create_crossings_and_missing_data_has_no_pass(monkeypatch):
    raw, mapper, _ = fixture([.021, .019, .021, .019, .021, .019])
    monkeypatch.setattr(audit, "compiled_topologies", lambda *_: (ORDER, None))
    raw["tick_ns"][2:] += audit.TICK_NS
    raw["clock_generation"][3:] += 1
    raw["segment_id"][4:] += 1
    samples = audit.new_samples()
    mask = np.array([True, False, True, True, True, True])
    counts = audit.collect_partition(raw, {}, mask, mapper, samples, version_fields=VERSIONS)
    assert counts["continuous_pairs"] == 1
    assert counts["crossing_ticks"] == 1 and counts["entry_ticks"] == 1
    assert audit.summarize(audit.new_samples(), active_roots=0)[audit.HARD]["extra_world_tip_step_mm"] == {
        "n": 0, "p95": None, "max": None, "within_numeric_limits": None,
    }


def test_comparison_reuses_exact_mapper_for_all_three_bands():
    raw, mapper, (home, calibrated, fit) = fixture([.019])
    wrists, palms = raw["pose"][0, 2:4, 0], raw["pose"][0, :2]
    _, _, _, variants = audit.compare_tick(mapper, wrists, palms, ORDER)
    for name, band in audit.BANDS.items():
        reference = IntentMapper(SimpleNamespace(local=home), calibrated, fit, smooth_band=band)
        expected = reference.map(wrists, palms, ORDER)["mapped"]
        for field, value in expected.items():
            np.testing.assert_array_equal(variants[name][field], value)
        lo, hi = band
        step = (hi - lo) * 1e-6
        near = smooth_weights(np.array([lo - step, lo, lo + step, hi - step, hi, hi + step]), band)
        np.testing.assert_allclose(near, [1., 1., 1., 0., 0., 0.], rtol=0, atol=4e-12)


def test_degenerate_geometry_keeps_denominator_and_breaks_pairing(monkeypatch):
    raw, mapper, _ = fixture([.019, .019, .021])
    raw["pose"][1, 0, 2, :3] = raw["pose"][1, 0, 1, :3]
    monkeypatch.setattr(audit, "compiled_topologies", lambda *_: (ORDER, None))
    samples = audit.new_samples()
    counts = audit.collect_partition(raw, {}, np.ones(3, dtype=bool), mapper, samples, version_fields=VERSIONS)
    assert (counts["eligible_ticks"], counts["mapped_ticks"], counts["paired_mapping_invalid_ticks"]) == (3, 2, 1)
    assert counts["crossing_ticks"] == 0
    assert all(not row["extra_step_mm"] for row in samples.values())
