"""Behavior checks for contact-aware retargeting on the pinned Wuji hand."""

from pathlib import Path

import numpy as np
import pytest
from dexgym.data.self_collision_postopt import HandPostOptimizer, PostOptWeights
from dexgym.data.wuji_retarget import RightHandFK

RIGHT_XML = Path("/home/ubuntu/teleop_ws/src/whole_body_mujoco_driver/model/source/wujihand2_beta2/mjcf/right.xml")
PILOT = Path(__file__).resolve().parents[2] / "data/processed/hand_only_right/s8_flashlight_on_1.npz"
OUTLIER = Path(__file__).resolve().parents[2] / "data/processed/hand_only_right/s4_fryingpan_cook_3.npz"
COLLIDING_Q = np.array(
    [
        -0.29796,
        0.69793,
        0.48191,
        0.63261,
        0.08863,
        0.39888,
        1.5206,
        0.17246,
        0.85988,
        -0.10049,
        1.52822,
        0.40168,
        1.09656,
        -0.44051,
        1.14228,
        0.36928,
        1.56843,
        -0.38825,
        0.29463,
        0.61476,
    ],
    dtype=np.float64,
)


@pytest.mark.skipif(not RIGHT_XML.is_file(), reason="pinned local Wuji model unavailable")
def test_contact_gradient_and_postopt_reduce_penetration():
    fk = RightHandFK(RIGHT_XML)
    optimizer = HandPostOptimizer(fk, PostOptWeights())
    assert optimizer.audit(np.zeros(20))["deepest_penetration_mm"] == 0.0
    q0 = COLLIDING_Q.copy()
    tip0 = fk.tips(q0)
    initial = optimizer.audit(q0)
    assert initial["deepest_penetration_mm"] > 10
    value, gradient = optimizer.frame_loss_and_grad(q0, q0, tip0, q0, q0)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    for joint in (5, 9, 13, 17):
        shifted = q0.copy()
        shifted[joint] += 1e-5
        shifted_value, _ = optimizer.frame_loss_and_grad(shifted, q0, tip0, q0, q0)
        np.testing.assert_allclose(gradient[joint], (shifted_value - value) / 1e-5, rtol=0.06, atol=0.15)

    result = optimizer.optimize_sequence(np.repeat(q0[None], 3, axis=0))
    q_star = result[0]
    assert optimizer.audit(q_star[1])["deepest_penetration_mm"] < initial["deepest_penetration_mm"] / 2
    assert np.max(np.linalg.norm(fk.tips(q_star[1]) - tip0, axis=-1)) < 0.01
    assert np.all(q_star >= fk.lower - 1e-7) and np.all(q_star <= fk.upper + 1e-7)


@pytest.mark.skipif(not RIGHT_XML.is_file() or not PILOT.is_file(), reason="local validation motion unavailable")
def test_sequence_refinement_does_not_create_large_joint_step():
    with np.load(PILOT, allow_pickle=False) as source:
        q_sdk = source["q_ref"]
    optimizer = HandPostOptimizer(RightHandFK(RIGHT_XML), PostOptWeights())
    q_star, _, _ = optimizer.optimize_sequence(q_sdk)
    original_step = float(np.abs(np.diff(q_sdk, axis=0)).max())
    optimized_step = float(np.abs(np.diff(q_star, axis=0)).max())
    assert optimized_step <= original_step + 0.015


@pytest.mark.skipif(not RIGHT_XML.is_file() or not OUTLIER.is_file(), reason="local collision outlier unavailable")
def test_postopt_never_accepts_worse_max_penetration_or_large_tip_shift():
    with np.load(OUTLIER, allow_pickle=False) as source:
        q_sdk = source["q_ref"]
    optimizer = HandPostOptimizer(RightHandFK(RIGHT_XML), PostOptWeights())
    _, _, metrics = optimizer.optimize_sequence(q_sdk)
    assert metrics["final_max_penetration_mm"] <= metrics["initial_max_penetration_mm"] + 0.1
    assert metrics["max_tip_change_mm"] <= 12.0
