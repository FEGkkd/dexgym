import unittest
from types import SimpleNamespace

from preflight import HUMAN, inventory_messages, topic_role


def record(topic, kind, stamp=1, channel_id=0, schema_bytes=b"schema", encoding="cdr"):
    schema = SimpleNamespace(name=f"robot_msgs/msg/{kind}", encoding="ros2msg", data=schema_bytes)
    channel = SimpleNamespace(topic=topic, id=channel_id, message_encoding=encoding, metadata={"qos": "recorded"})
    message = SimpleNamespace(log_time=stamp, publish_time=stamp + 1)
    return schema, channel, message


class PreflightTests(unittest.TestCase):
    def test_recognizes_human_candidates_without_authorizing_replay(self):
        result = inventory_messages(record(topic, kind, channel_id=idx) for idx, (topic, kind) in enumerate(HUMAN.items()))
        self.assertEqual(result["file_order_message_count"], 4)
        self.assertEqual(result["identity_or_schema_findings"], [])
        self.assertFalse(result["replay_whitelist_approved"])
        self.assertIn("UNKNOWN", result["start_event_status"])

    def test_reports_source_order_and_schema_conflict(self):
        topic, kind = next(iter(HUMAN.items()))
        rows = [record(topic, kind, 30), record(topic, kind, 15), record(topic, kind, 15),
                record(topic, kind, 1, 4, schema_bytes=b"other schema")]
        result = inventory_messages(rows)
        report = result["channels"][0]
        self.assertEqual((report["log_backsteps"], report["same_log_time"]), (1, 1))
        self.assertEqual((report["log_first_ns"], report["log_last_ns"]), (15, 30))
        self.assertTrue(any("Conflicting" in finding for finding in result["identity_or_schema_findings"]))

    def test_recorded_commands_and_feedback_are_quarantined(self):
        self.assertEqual(topic_role("/arm/joint_commands"), "ROBOT_COMMAND_AUDIT_ONLY")
        self.assertEqual(topic_role("/teleop/robot_feedback"), "HISTORICAL_FEEDBACK_AUDIT_ONLY")
        self.assertEqual(topic_role("/teleop/workflow/state"), "EVENT_CANDIDATE_REQUIRES_DECODE")
        self.assertEqual(topic_role("/copied/manus_glove_left"), "DIAGNOSTIC_OR_UNKNOWN")

    def test_missing_stream_and_non_cdr_block_inventory(self):
        topic, kind = next(iter(HUMAN.items()))
        result = inventory_messages([record(topic, kind, encoding="json")])
        self.assertEqual(len(result["identity_or_schema_findings"]), 4)
        self.assertTrue(any("CDR" in finding for finding in result["identity_or_schema_findings"]))
