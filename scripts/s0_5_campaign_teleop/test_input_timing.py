from dataclasses import asdict
from hashlib import sha256
from types import SimpleNamespace
import unittest

from input_timing import CLOCK_TAG, ROS_TYPES, TELEOP_REVISION, Source, iter_input_timing, resolve_review
from preflight import HUMAN


class TimingWitnessTests(unittest.TestCase):
    def setUp(self):
        self.schema_bytes = b"definition"
        self.ids = {
            topic: Source(i + 1, ROS_TYPES[topic], "ros2msg", sha256(self.schema_bytes).hexdigest())
            for i, topic in enumerate(HUMAN)
        }
        self.topics = list(HUMAN)

    def record(self, topic, stamp, *, data=b"test", schema_bytes=None, channel_id=None):
        source = self.ids.get(topic, Source(50, "tianji_interfaces/msg/JointCommand", "ros2msg", ""))
        schema = SimpleNamespace(id=source.channel_id, name=source.schema_name, encoding=source.schema_encoding,
                                 data=self.schema_bytes if schema_bytes is None else schema_bytes)
        channel = SimpleNamespace(id=source.channel_id if channel_id is None else channel_id,
                                  topic=topic, message_encoding="cdr")
        message = SimpleNamespace(log_time=stamp, data=data, publish_time=0)
        return schema, channel, message

    def test_equal_times_keep_original_indices_and_quarantine_other_topics(self):
        rows = [self.record("/tianji/joint_command", 90), self.record(self.topics[0], 100),
                self.record("/tianji/robot_state", 91), self.record(self.topics[1], 100),
                self.record(self.topics[2], 110), self.record(self.topics[3], 125)]
        timeline = list(iter_input_timing(rows, self.ids))
        self.assertEqual([entry.record_index for entry in timeline], [1, 3, 4, 5])
        self.assertEqual([entry.relative_deadline_ns for entry in timeline], [0, 0, 10, 25])
        self.assertEqual([entry.topic for entry in timeline], self.topics)

    def test_clock_backstep_is_rejected_without_reordering(self):
        with self.assertRaisesRegex(ValueError, "Noncausal selected log time"):
            list(iter_input_timing([self.record(self.topics[0], 10), self.record(self.topics[1], 9)], self.ids))

    def test_rejects_schema_drift_and_unreviewed_channel(self):
        with self.assertRaisesRegex(ValueError, "Embedded declaration changed"):
            list(iter_input_timing([self.record(self.topics[0], 10, schema_bytes=b"drift")], self.ids))
        with self.assertRaisesRegex(ValueError, "Embedded declaration changed"):
            list(iter_input_timing(
                [self.record(self.topics[0], 10), self.record(self.topics[0], 11, schema_bytes=b"drift")],
                self.ids,
            ))
        with self.assertRaisesRegex(ValueError, "Unreviewed channel"):
            list(iter_input_timing([self.record(self.topics[0], 10, channel_id=77)], self.ids))

    def test_review_excludes_command_alias_and_requires_pinned_clock(self):
        bag_sha = "b" * 64
        review = {
            "schema": "s05.teleop_timing_witness.review.v1", "teleop_revision": TELEOP_REVISION,
            "bag_sha256": bag_sha, "clock": CLOCK_TAG, "clock_evidence_sha256": "a" * 64,
            "tracker_basis": "REGISTERED_INPUT",
            "channels": {topic: asdict(source) for topic, source in self.ids.items()},
        }
        self.assertEqual(resolve_review(review, bag_sha), self.ids)
        with self.assertRaisesRegex(ValueError, "Review identity"):
            resolve_review({**review, "clock": "publish_time"}, bag_sha)
        review["channels"]["/tianji/joint_command"] = asdict(next(iter(self.ids.values())))
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            resolve_review(review, bag_sha)

    def test_partial_input_is_flagged(self):
        with self.assertRaisesRegex(ValueError, "Selected human input absent"):
            list(iter_input_timing([self.record(self.topics[0], 10)], self.ids))
