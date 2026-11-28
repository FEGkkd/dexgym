import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from home_anchored_start_anchor import (
    AGE_NS, ARRIVAL_DELAY_NS, PRIMARY, SENSITIVITY, STABLE_IDENTITY_FIELDS, TICK_NS, Row, choose_anchor,
    clock_status, hmb_five_rows, motion_events, palm_class, select_six, stable_window, start_transform, train_reader,
)


ABSENT = "UNOBSERVED_IN_SIGNED_NATIVE_INPUTS"


def rows(length, stride=1):
    target = np.repeat(np.eye(4)[None], 2, axis=0)
    target[:, 0, 3] = [-.2, .2]
    return [Row("part", i, i * TICK_NS, 2, ((0,) * 4,) * 3, "VALID",
                tuple((str(role), i // stride) for role in range(4)),
                np.full(4, (i // stride) * TICK_NS), target.copy(), np.zeros((2, 5, 3)))
            for i in range(length)]


class TestHomeAnchoredPilot(unittest.TestCase):
    def test_first_stable_terminal_is_causal(self):
        data = rows(30)
        data[0] = replace(data[0], status="HMB_JOINT_INVALID_OR_UNSAFE")
        a, receipt = choose_anchor(data, PRIMARY, ABSENT)
        b, sensitivity = choose_anchor(data, SENSITIVITY, ABSENT)
        self.assertEqual((a.tick, b.tick), (11, 21))
        self.assertEqual(receipt["status"], "RECONSTRUCTED_START")
        self.assertEqual(receipt["last_cutoff_ns"] - receipt["first_cutoff_ns"], 200_000_000)
        self.assertEqual(sensitivity["last_cutoff_ns"] - sensitivity["first_cutoff_ns"], 400_000_000)
        self.assertEqual(choose_anchor(data[:12], PRIMARY, ABSENT)[1], receipt)
        data[19] = replace(data[19], status="SOURCE_TIME_OR_AGE")
        self.assertEqual(choose_anchor(data, PRIMARY, ABSENT)[1], receipt)

    def test_explicit_bad_event_cannot_fall_through(self):
        data = rows(24)
        self.assertIsNone(choose_anchor(data, PRIMARY, "PRESENT_INELIGIBLE")[0])
        bad, receipt = choose_anchor(data, PRIMARY, "PRESENT_QUALIFIED_EVENT", 5 * TICK_NS)
        self.assertIsNone(bad)
        self.assertEqual(receipt["reason"], "EXPLICIT_EVENT_WINDOW_UNQUALIFIED")
        good, receipt = choose_anchor(data, PRIMARY, "PRESENT_QUALIFIED_EVENT", 13 * TICK_NS)
        self.assertEqual((good.tick, receipt["status"]), (13, "EXPLICIT_START"))

    def test_duplicate_samples_and_version_barriers(self):
        sparse = rows(21, stride=4)
        self.assertEqual(stable_window(sparse[:11], PRIMARY)[1], "SOURCE_SUPPORT_TOO_LOW")
        self.assertEqual(stable_window(sparse, SENSITIVITY)[1], "SOURCE_SUPPORT_TOO_LOW")
        dense = rows(21, stride=2)
        dense[1].tips[0, 2, 1] = .001
        self.assertEqual(stable_window(dense, SENSITIVITY)[1], "REPEATED_SOURCE_GEOMETRY_CHANGED")
        dense = rows(11)
        dense[7] = replace(dense[7], part="other")
        self.assertEqual(stable_window(dense, PRIMARY)[1], "DISCONTINUOUS_WINDOW")

    def test_stability_is_distinct_source_geometry(self):
        data = rows(11)
        data[4].wrists[1, 0, 3] += .003
        self.assertEqual(stable_window(data, PRIMARY)[1], "ANATOMICAL_WRIST_MOTION_EXCEEDS_STABLE_WINDOW")
        data = rows(11)
        data[4].tips[1, 2, 0] += .002
        self.assertEqual(stable_window(data, PRIMARY)[1], "MANUS_MOTION_EXCEEDS_STABLE_WINDOW")

    def test_clock_limits(self):
        tick = 100_000_000
        cap = np.full((1, 4, 2), tick - 25_000_000)
        sample = {"tick_ns": np.array([tick]), "capture_bounds_ns": cap.copy(),
                  "arrival_bounds_ns": cap + ARRIVAL_DELAY_NS, "age_ns": tick - cap[..., 0]}
        self.assertEqual(clock_status(sample, 0), "VALID")
        sample["arrival_bounds_ns"][0, 0, 1] = tick + 1
        self.assertEqual(clock_status(sample, 0), "SOURCE_TIME_OR_AGE")
        cap[0, 0] = tick - AGE_NS - 1
        sample.update(capture_bounds_ns=cap, arrival_bounds_ns=cap + ARRIVAL_DELAY_NS, age_ns=tick - cap[..., 0])
        self.assertEqual(clock_status(sample, 0), "SOURCE_TIME_OR_AGE")
        cap[0, 0] = tick - AGE_NS
        cap[0, 1] = tick - 10_000_000
        sample.update(arrival_bounds_ns=cap + ARRIVAL_DELAY_NS, age_ns=tick - cap[..., 0])
        self.assertEqual(clock_status(sample, 0), "SOURCE_SKEW")

    def test_matched_anatomical_home_preserves_pose_and_wrist_span(self):
        anchor, home = rows(1)[0].wrists, rows(1)[0].wrists.copy()
        anchor[:, 0, 3] *= 1.6
        anchor[1, :3, :3] = Rotation.from_euler("x", 47, degrees=True).as_matrix()
        home[:, :3, 3] += [.14, -.31, .9]
        r_cal = Rotation.from_euler("xyz", [.4, .6, -.15]).as_matrix()
        shared = start_transform(r_cal, anchor, home)
        mapped = shared @ anchor
        np.testing.assert_allclose(mapped[:, :3, 3].mean(axis=0), home[:, :3, 3].mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(mapped[0]) @ mapped[1], np.linalg.inv(anchor[0]) @ anchor[1], atol=1e-12)
        np.testing.assert_allclose(mapped[1, :3, :3], r_cal @ anchor[1, :3, :3], atol=1e-12)
        self.assertGreater(np.linalg.norm(mapped[0, :3, 3] - home[0, :3, 3]), .1)
        with self.assertRaisesRegex(ValueError, "right-handed"):
            start_transform(np.diag([-1., 1., 1.]), anchor, home)

    def test_local_axes_are_not_global_palm_down_force(self):
        wrists = rows(1)[0].wrists
        self.assertEqual(palm_class(wrists, np.eye(3), [[0,0,-1],[0,0,-1]])["class"], "down")
        self.assertEqual(palm_class(wrists, np.eye(3), [[0,0,1],[0,0,1]])["class"], "up")
        self.assertEqual(palm_class(wrists, np.eye(3), [[1,0,0],[-1,0,0]])["class"], "facing")
        self.assertEqual(palm_class(wrists, np.eye(3), [[0,0,-1],[0,0,1]])["class"], "mixed")

    def test_five_frames_do_not_cross_hmb_version_or_loss(self):
        count = 18
        data = {"joint_valid": np.ones(count, bool), "safety_permission": np.ones(count, bool),
                "pose_valid": np.ones((count,4), bool), "segment_id": np.ones(count, int),
                "tick_ns": np.arange(count)*TICK_NS, **{key: np.zeros((count,4),int) for key in STABLE_IDENTITY_FIELDS}}
        np.testing.assert_array_equal(np.flatnonzero(hmb_five_rows(data)), np.arange(4, count))
        data["clock_generation"][8:] += 1
        data["safety_permission"][13] = False
        np.testing.assert_array_equal(np.flatnonzero(hmb_five_rows(data)), [4,5,6,7,12])

    def test_motion_labels_are_from_input_and_exclude_uncovered_windows(self):
        times = np.arange(61) * TICK_NS
        poses = np.zeros((61,2,3))
        poses[:,0,0] = np.minimum(times/1e9,.55) * .30
        sources = np.repeat(times[:,None], 2, axis=1)
        valid = np.ones(61,bool)
        events = motion_events(times, np.zeros(61), poses, sources, valid, valid)
        self.assertTrue(any(e["kind"] == "stop" for e in events))
        self.assertEqual(motion_events(times, np.zeros(61), poses, sources, valid, times < 200_000_000), [])

    def test_six_and_repeats_deterministic_or_inconclusive(self):
        palm = ["down","up","mixed","facing","up","up","down","mixed"]
        records=[{"root": f"capture:{idx}", "split": "Train", "median_anatomical_gap_m": .2+.05*idx,
                  "ineligibility_reason": "NO_STABLE_WINDOW" if idx == 1 else None,
                  "palm": {"class": kind}, "contains_start_stop_or_reversal": idx in (2,3)}
                 for idx, kind in enumerate(palm)]
        chosen, repeats, _ = select_six(list(reversed(records)))
        self.assertEqual(len(chosen), 6)
        self.assertEqual(set(repeats), {"capture:0", "capture:6"})
        self.assertNotIn("capture:1", {r["root"] for r in chosen})
        records[3]["ineligibility_reason"] = "SOURCE_SUPPORT_TOO_LOW"
        self.assertEqual(select_six(records)[0], ())
        records[3]["split"] = "Validation"
        with self.assertRaisesRegex(ValueError, "only access Train"):
            select_six(records)


if __name__ == "__main__":
    unittest.main()
