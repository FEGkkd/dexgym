import unittest

import numpy as np

from dexgym.tasks.direct.wholebody_s05.v1.rmb.audit_nominal_workspace import (
    BOUND_MARGIN_M,
    audit_partition,
    upper_bound,
)


def raw_hmb():
    ticks = np.arange(8) * 20_000_000
    raw = {
        "tick_ns": ticks,
        "cutoff_ns": ticks,
        "joint_valid": np.r_[np.ones(7, dtype=bool), False],
        "safety_permission": np.ones(8, dtype=bool),
        "pose_valid": np.ones((8, 4), dtype=bool),
        "source_index": np.zeros((8, 4, 1), dtype=int),
        "capture_bounds_ns": np.zeros((8, 4, 2), dtype=int),
        "arrival_bounds_ns": np.zeros((8, 4, 2), dtype=int),
        "segment_id": np.zeros(8, dtype=int),
        "pose": np.zeros((8, 4, 1, 7)),
    }
    # Left-only, right-only, both, inside but adjusted by the SDK, within
    # numerical margin, missing native workspace, future capture, invalid HMB.
    raw["pose"][:, 2, 0, 0] = [.7, .4, .7, .598, .6 + BOUND_MARGIN_M / 2, .4, .9, .9]
    raw["pose"][:, 3, 0, 0] = [.4, .7, .7, .598, .599, .45, .9, .9]
    raw["capture_bounds_ns"][6, 2, 1] = ticks[6] + 1
    return raw


class FrozenWristMapping:
    def place(self, layer, raw_wrists, hand_shape):
        if layer != "M1" or np.any(hand_shape):
            raise AssertionError("The wrist audit accepts only the frozen mapping and inert hand shape")
        return {"wrist_xyz": raw_wrists[:, :3], "wrist_xyzw": [[0, 0, 0, 1], [0, 0, 0, 1]]}


class Workspace:
    def workspace_probe(self, side, pose):
        radius = pose[0, 3]
        if side == 1 and radius == .45:
            return None
        # This deliberately reports a transition offset even inside the
        # physical 0.6 m outer bound.
        return {"radius_request_m": radius, "radius_max_m": .6, "offset_m": .003 if radius > .597 else 0.}


class NominalWorkspaceTests(unittest.TestCase):
    def test_bilateral_upper_bound_deduplicates_sides_and_keeps_unknown(self):
        bridges = [(np.eye(4), np.eye(4), .6)] * 2
        result = upper_bound(audit_partition(raw_hmb(), False, FrozenWristMapping(), Workspace(), bridges))
        self.assertEqual(result["planned_ticks"], 8)
        self.assertEqual(result["hmb_joint_valid_ticks"], 7)
        self.assertEqual(result["rmb_input_mask_eligible_ticks"], 6)
        self.assertEqual(result["strict_outside_side_0_ticks"], 2)
        self.assertEqual(result["strict_outside_side_1_ticks"], 2)
        self.assertEqual(result["strict_outside_both_sides_ticks"], 1)
        self.assertEqual(result["strict_outside_either_side_ticks"], 3)
        self.assertEqual(result["not_excluded_but_workspace_unknown_ticks"], 1)
        self.assertEqual(result["both_sides_known_not_excluded_ticks"], 2)
        self.assertEqual(result["inside_adjusted_by_controller_side_0_ticks"], 2)
        self.assertEqual(result["exact_direct_ik_hmb_joint_valid_ticks_upper"], 4)
        self.assertEqual(result["exact_direct_ik_hmb_joint_valid_fraction_upper"], 4 / 7)
        self.assertEqual(result["exact_direct_ik_planned_fraction_upper"], .5)

    def test_denied_source_keeps_historical_denominator_without_loading_wrist(self):
        result = upper_bound(audit_partition(raw_hmb(), True, None, None, None))
        self.assertEqual(result["hmb_joint_valid_ticks"], 7)
        self.assertEqual(result["denied_hmb_joint_valid_ticks"], 7)
        self.assertEqual(result["rmb_input_mask_eligible_ticks"], 0)
        self.assertEqual(result["exact_direct_ik_hmb_joint_valid_fraction_upper"], 1.)
