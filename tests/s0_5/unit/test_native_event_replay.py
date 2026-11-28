"""Causality and state-continuity controls for ROOT05 native replay scheduling."""

from collections import Counter
from io import StringIO
import json
import unittest

from scripts.s0_5_campaign_teleop import native_event_replay as replay


class HandCallbacks:
    def __init__(self):
        self.counts = Counter()
        self.calls = []

    def step(self, row, clock, *, reconstructed, gate_open):
        side = row["side"]
        self.counts[side] += 1
        self.calls.append((side, clock))
        publishes = not row.get("reject", False)
        return {"stage": "PUBLISHED" if publishes else "REJECTED", "published": publishes,
                "filtered_rad": [self.counts[side]] * 20 if publishes else None}


def event(side, receipt, physical, writer=None, **extra):
    return {"side": side, "receive_ns": receipt, "collector_mono_ns": 100 + receipt,
            "source_index": f"{side}:{physical}", "source_instance": f"writer:{side}",
            "writer_sequence": physical if writer is None else writer, "physical_message": physical,
            "in_recorded_interval": True, "sdk_source_stamp_raw": 13, "cdr_sha256": "same", **extra}


def tick(partition, cutoff, part_index=0):
    return {"partition": partition, "partition_tick": part_index, "tick_ns": cutoff,
            "in_start_window": True, "joint_valid": False, "safety_permission": False,
            "nominal_pair_eligible": False, "history_eligible": False, "pose_valid": [False, False],
            "source_index": [None, None], "writer_identity": [None, None], "arrival_ns": [None, None]}


class NativeReplayTests(unittest.TestCase):
    def make_replay(self, events):
        native = HandCallbacks()
        output = StringIO()
        return replay.EventReplay(native, events, output, 100), native, output

    def test_future_excluded_and_cutoff_tie_consumes_each_hand_in_file_order(self):
        route, native, _ = self.make_replay([
            event("left", 1, 0), event("left", 2, 2), event("left", 2, 4), event("right", 2, 0),
        ])
        self.assertEqual(route.at(0), [None, None])
        self.assertEqual(native.calls, [])
        left, right = route.at(1)
        self.assertEqual((left["source_index"], right), ("left:0", None))
        left, right = route.at(2)
        self.assertEqual((left["source_index"], right["source_index"]), ("left:4", "right:0"))
        self.assertEqual(native.calls, [("left", 1), ("left", 2), ("left", 2), ("right", 2)])
        self.assertEqual(left["age_ns"], 0)
        self.assertEqual(right["event_ordinal"], 3)

    def test_tie_out_of_order_is_rejected(self):
        route, _, _ = self.make_replay([event("right", 0, 0), event("left", 0, 0)])
        with self.assertRaisesRegex(ValueError, "file order"):
            route.at(0)

    def test_repeated_writer_identity_is_rejected(self):
        route, native, _ = self.make_replay([event("left", 0, 0), event("left", 1, 2, writer=0)])
        with self.assertRaisesRegex(ValueError, "Repeated native event identity"):
            route.at(1)
        self.assertEqual(native.calls, [("left", 0)])

    def test_sdk_stamp_and_payload_repeats_are_executed_for_distinct_writer_events(self):
        route, native, _ = self.make_replay([event("left", 0, 0), event("left", 1, 2)])
        left, _ = route.at(1)
        self.assertEqual(left["filtered_rad"], [2] * 20)
        self.assertEqual(native.calls, [("left", 0), ("left", 1)])

    def test_rejected_callback_keeps_prior_publication(self):
        route, native, _ = self.make_replay([event("left", 0, 0), event("left", 1, 2, reject=True)])
        left, _ = route.at(1)
        self.assertEqual(left["source_index"], "left:0")
        self.assertEqual(left["age_ns"], 1)
        self.assertEqual(len(native.calls), 2)

    def test_partition_boundary_keeps_state_and_sample_uses_preceding_500hz_row(self):
        route, native, _ = self.make_replay([
            event("left", 0, 0), event("right", 1, 0),
            event("left", replay.SAMPLE_NS - 1, 2), event("right", replay.SAMPLE_NS + 1, 2),
        ])
        zoh, samples = StringIO(), StringIO()
        ticks = [tick("old", 0), tick("new", replay.SAMPLE_NS + 2)]
        grids = replay.sample_grids(route, ticks, replay.SAMPLE_NS + replay.ZOH_NS, zoh, samples)
        self.assertEqual(grids["rows_500hz"], 11)
        first, second = [json.loads(line) for line in samples.getvalue().splitlines()]
        self.assertEqual(first["partition"], "old")
        self.assertEqual(second["partition"], "new")
        self.assertEqual(second["sample_lag_ns"], 2)
        self.assertEqual([hand["source_index"] for hand in second["hands"]], ["left:2", "right:0"])
        self.assertEqual(second["hands"][0]["filtered_rad"], [2] * 20)
        self.assertEqual(second["hands"][1]["filtered_rad"], [1] * 20)
        self.assertEqual(len(native.calls), 3)
        route.finish()
        self.assertEqual(native.calls[-1], ("right", replay.SAMPLE_NS + 1))


if __name__ == "__main__":
    unittest.main()
