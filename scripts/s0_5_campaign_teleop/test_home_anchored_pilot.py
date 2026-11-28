"""Offline controls for the DCR-009 native runtime and isolated overlay."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_home_anchored_pilot_overlay import prepare
from replay_dcr009_guarded_diagnostic import CENTER_OFFSET_SEQUENCE_MM, VIRTUAL_R, registered_targets
from replay_home_anchored_pilot import (
    AUTHORITY_SCHEMA, DESIGN, RUN_ID, FeedbackDwell, digest, expected_arm_publish,
    load_prerequisites, lock_home_frames, map_file, proper, read_frames, validate_environment,
)


class PilotRuntimeTest(unittest.TestCase):
    def test_strict_start_only_stops_without_claiming_historical_replay(self):
        from run_dcr009_guarded_diagnostic import evidence_summary

        samples = [
            {"kind": "start_handoff", "stage": "preposition_start", "status": "START_REACHED"},
            {"kind": "start_only_completed", "stage": "preposition_start",
             "profile": "segmented", "original_history_started": False},
            {"kind": "pilot_stop", "stage": "tail", "late_observed_command_count": 0},
        ]
        with TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
            result = evidence_summary(trace)
            self.assertEqual(result["observed_phase"], "START_REACHED_NO_HISTORY_BY_SCOPE")
            self.assertEqual(result["historical_packets"], 0)
            samples[0]["status"] = "START_NOT_REACHED"
            trace.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "START-only completion"):
                evidence_summary(trace)

    def test_center_offset_translates_both_registered_and_engineering_targets_in_world(self):
        frames = np.repeat(np.eye(4)[None], 7, axis=0)
        frames[0, :3, :3] = VIRTUAL_R
        frames[1, :3, :3] = Rotation.from_euler("z", 32, degrees=True).as_matrix()
        frames[2, :3, :3] = Rotation.from_euler("x", -18, degrees=True).as_matrix()
        frames[3:7, :3, 3] = [[.04, -.05, .03], [.02, .08, -.06], [.06, .06, .01], [.09, .01, -.02]]
        home = np.repeat(np.eye(4)[None], 2, axis=0)
        home[:, :3, 3] = [[.2, -.3, .4], [-.2, -.1, .4]]
        raw = home.copy()
        raw[1, :3, :3] = Rotation.from_euler("y", 38, degrees=True).as_matrix()
        initial, initial_registered, initial_native = registered_targets(frames, home, raw)
        for offset in CENTER_OFFSET_SEQUENCE_MM:
            with self.subTest(offset=offset):
                locked, registered, native = registered_targets(frames, home, raw, offset)
                delta = np.repeat([[offset / 1000, 0, 0]], 2, axis=0)
                np.testing.assert_array_equal(locked[:, :3, :3], initial[:, :3, :3])
                np.testing.assert_array_equal(locked[1:], initial[1:])
                np.testing.assert_array_equal(registered[:, :3, :3], initial_registered[:, :3, :3])
                np.testing.assert_allclose(registered[:, :3, 3] - initial_registered[:, :3, 3], delta, atol=1e-14)
                np.testing.assert_allclose((frames[1:3] @ native)[:, :3, 3]
                                           - (frames[1:3] @ initial_native)[:, :3, 3], delta, atol=1e-14)

    def test_shared_map_only_translates_the_fixed_rotation(self):
        frames = np.repeat(np.eye(4)[None], 7, axis=0)
        frames[0, :3, :3] = Rotation.from_euler("z", 49, degrees=True).as_matrix()
        home = np.repeat(np.eye(4)[None], 2, axis=0)
        home[:, :3, 3] = [[.2, -.3, .4], [-.2, -.1, .4]]
        start = np.repeat(np.eye(4)[None], 2, axis=0)
        start[:, :3, 3] = [[.4, .6, .3], [.5, -.1, .2]]
        start[1, :3, :3] = Rotation.from_euler("x", -23, degrees=True).as_matrix()
        fixed, target, goal = lock_home_frames(frames, home, start)
        np.testing.assert_array_equal(fixed[:, :3, :3], frames[:, :3, :3])
        np.testing.assert_allclose(target[:, :3, 3].mean(axis=0), home[:, :3, 3].mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(np.linalg.inv(target[0]) @ target[1], np.linalg.inv(start[0]) @ start[1], atol=1e-14)
        np.testing.assert_array_equal(target, goal)
        frames[3, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "Reflected"):
            proper(frames, 7)

    def test_dwell_cannot_count_polling_or_invalid_samples(self):
        state = FeedbackDwell()
        for _ in range(20):
            self.assertFalse(state.observe(10, True))
        for tick in range(1, 10):
            self.assertFalse(state.observe(tick * 20_000_000, True))
        self.assertFalse(state.observe(180_000_000, False))
        self.assertFalse(state.observe(200_000_000, True))
        for tick in range(11, 20):
            self.assertFalse(state.observe(tick * 20_000_000, True))
        self.assertTrue(state.observe(400_000_000, True))
        self.assertFalse(state.observe(480_000_000, True))
        self.assertFalse(state.observe(470_000_000, True))

    def test_issue_matching_rejects_holds_and_reused_diagnostic_sequences(self):
        row = {"session": "s", "pilot_publish_advanced": "true", "pilot_arm_status": "LIMITED", "sequence": "13",
               "pilot_previous_sequence": "12", "pilot_command_header_ns": "100"}
        command = (100, {"sequence": 13, "session_id": "s"})
        self.assertTrue(expected_arm_publish(row, "s", command))
        for patch in ({"pilot_arm_status": "HOLD"}, {"pilot_arm_status": "FALLBACK"},
                      {"pilot_publish_advanced": "false"}, {"pilot_command_header_ns": "101"},
                      {"pilot_previous_sequence": "13"}, {"session": "other"}):
            self.assertFalse(expected_arm_publish(dict(row, **patch), "s", command))
        self.assertFalse(expected_arm_publish(row, "s", None))

    def test_isolated_runtime_requires_clean_scope_and_fresh_map(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "session-map.txt"
            env = {"ROS_DOMAIN_ID": "208", "ROS_LOCALHOST_ONLY": "1", "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
                   "S05_HOME_ANCHORED_SESSION_MAP": str(output)}
            validate_environment(env, output)
            for change in ({"ROS_DOMAIN_ID": "205"}, {"ROS_LOCALHOST_ONLY": "0"}, {"ROS_DISCOVERY_SERVER": "external"},
                           {"S05_SHARED_RIGID_MAP": "old.txt"}):
                with self.assertRaisesRegex(ValueError, "isolated"):
                    validate_environment(dict(env, **change), output)
            frames = np.repeat(np.eye(4)[None], 7, axis=0)
            self.assertEqual(map_file(output, "s05-dcr009-synthetic", frames), digest(output))
            with self.assertRaisesRegex(ValueError, "fresh"):
                validate_environment(env, output)
            frames_path = output.with_name("frames.txt")
            np.savetxt(frames_path, frames[:, :3].reshape(21, 4))
            np.testing.assert_array_equal(read_frames(frames_path), frames)
            output.chmod(0o644)

    def test_unscoped_authority_fails_before_assets_or_ros(self):
        with TemporaryDirectory() as directory:
            file = Path(directory) / "authority.json"
            authority = {"schema": AUTHORITY_SCHEMA, "run_id": RUN_ID, "design_commit": DESIGN, "scope": "MUJOCO_ONLY",
                         "pilot_execution_authorized": True, "full_campaign_authorized": False, "training_authorized": True,
                         "real_robot_authorized": False, "implementation_commit": "a" * 40}
            file.write_text(json.dumps(authority), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "authority"):
                load_prerequisites(SimpleNamespace(authority=file, authority_sha256=digest(file)))

    def test_blocked_calibration_needs_no_fabricated_frames_or_executables(self):
        with TemporaryDirectory() as directory:
            calibration = Path(directory) / "calibration.json"
            calibration.write_text(json.dumps({"schema_version": 1, "scope": "MUJOCO_ONLY", "status": "BLOCKED"}),
                                   encoding="utf-8")
            file = Path(directory) / "authority.json"
            authority = {"schema": AUTHORITY_SCHEMA, "run_id": RUN_ID, "design_commit": DESIGN, "scope": "MUJOCO_ONLY",
                         "pilot_execution_authorized": True, "full_campaign_authorized": False, "training_authorized": False,
                         "real_robot_authorized": False, "implementation_commit": "a" * 40,
                         "paths_sha256": {"calibration_receipt": digest(calibration)}}
            file.write_text(json.dumps(authority), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration has no qualification: BLOCKED"):
                load_prerequisites(SimpleNamespace(authority=file, authority_sha256=digest(file), calibration=calibration))

    def test_overlay_requires_source_matches_and_only_one_install(self):
        with TemporaryDirectory() as directory:
            package = Path(directory) / "whole_body_teleop"
            source = package / "src"
            source.mkdir(parents=True)
            (source / "teleop_core.cpp").write_text(
                "Transform scaled_delta_target(\n      auto wrist = scaled_delta_target(\n", encoding="utf-8")
            (source / "teleop_controller_node.cpp").write_text('''  config.absolute_orientation = bimanual_.t_heading_enabled ||
    add("mapping_policy",core_->config().absolute_orientation ? "world_absolute_orientation_v1" : "world_spatial_v1");
    add("arm_command_accepted",arm.accepted ? "true" : "false");
''', encoding="utf-8")
            prepare(package)
            fixture = (source / "teleop_core.cpp").read_text(encoding="utf-8")
            self.assertIn("reference.session_id != session", fixture)
            self.assertIn("world_from_base[side].inverse() * world_from_human * tracker", fixture)
            self.assertIn("pilot_publish_advanced", (source / "teleop_controller_node.cpp").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "original reviewed native source"):
                prepare(package)

    def test_parent_rejects_unqualified_recorded_source_and_safety_before_launch(self):
        from run_home_anchored_pilot import load_launch_inputs

        for coverage, expected in (("BLOCKED", "CAL is not qualified"), ("QUALIFIED", "swept-path publish guard")):
            with self.subTest(coverage=coverage), TemporaryDirectory() as directory:
                calibration = Path(directory) / "calibration.json"
                calibration.write_text(json.dumps({
                    "schema_version": 1, "scope": "MUJOCO_ONLY", "status": "QUALIFIED",
                    "qualification": {"recorded_mcap_pilot": coverage},
                }), encoding="utf-8")
                authority = Path(directory) / "authority.json"
                authority.write_text(json.dumps({
                    "schema": AUTHORITY_SCHEMA, "run_id": RUN_ID, "scope": "MUJOCO_ONLY",
                    "pilot_execution_authorized": True, "full_campaign_authorized": False,
                    "training_authorized": False, "real_robot_authorized": False,
                    "paths_sha256": {"calibration_receipt": digest(calibration)},
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    load_launch_inputs(SimpleNamespace(authority=authority, authority_sha256=digest(authority),
                                                       calibration=calibration))


if __name__ == "__main__":
    unittest.main()
