"""Contracts checked against the pinned upstream Wuji description checkout."""

import os
from pathlib import Path
import subprocess
import unittest

from dexteroushands_lab.assets.wuji_hand2_beta import _asset_tree_sha256, load_hand2_spec


ROOT = Path(os.environ.get("WUJI_DESCRIPTION_BETA_ROOT",
                           os.environ.get("WUJI_DESCRIPTION_ROOT", "")))


def _pinned_root_available():
    if not (ROOT / "hand2/hand2_beta1").exists():
        return False
    if not (ROOT / ".git").exists():
        return True  # The adapter still validates the official USD SHA-256.
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.returncode == 0 and result.stdout.strip() == (
        "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95"
    )


@unittest.skipUnless(_pinned_root_available(), "pinned official Beta tag checkout unavailable")
class Hand2BetaAssetTests(unittest.TestCase):
    def test_both_sides_have_distinct_native_usd_and_20_mapped_joints(self):
        left = load_hand2_spec(ROOT, variant="beta1", side="left")
        right = load_hand2_spec(ROOT, variant="beta1", side="right")
        self.assertNotEqual(left.usd_path, right.usd_path)
        self.assertEqual(len(left.joints), 20)
        self.assertEqual(len(right.joints), 20)
        self.assertEqual(left.palm_body, "l_wrist")
        self.assertEqual(right.palm_body, "r_wrist")
        self.assertEqual(left.joints[0].actuator, "l_THJ0")
        self.assertEqual(left.joints[0].name, "l_thumb_cmc_flex")
        self.assertEqual(right.joints[-1].actuator, "r_LFJ3")
        self.assertEqual(right.joints[-1].name, "r_pinky_dip")

    def test_joint_limits_and_efforts_come_from_official_mjcf(self):
        for side in ("left", "right"):
            spec = load_hand2_spec(ROOT, variant="beta1", side=side)
            self.assertTrue(all(j.lower < 0 < j.upper for j in spec.joints))
            self.assertTrue(all(j.mjcf_effort_limit > 0 for j in spec.joints))
            self.assertTrue(all(j.usd_drive_max_force == 2.0 for j in spec.joints))
            self.assertEqual(len(set(j.name for j in spec.joints)), 20)
            self.assertEqual(len(spec.fingertip_bodies), 5)

    def test_beta2_has_sensor_frame_bodies_without_requiring_tactile_runtime(self):
        beta1 = load_hand2_spec(ROOT, variant="beta1", side="left")
        beta2 = load_hand2_spec(ROOT, variant="beta2", side="left")
        self.assertEqual(tuple(j.name for j in beta1.joints), tuple(j.name for j in beta2.joints))
        self.assertFalse(any("tip_sensor_frame" in b for b in beta1.body_names))
        self.assertEqual(sum("tip_sensor_frame" in b for b in beta2.body_names), 5)
        self.assertFalse(beta1.requires_tactile)
        self.assertFalse(beta2.requires_tactile)

    def test_version_and_checksum_guard(self):
        spec = load_hand2_spec(ROOT, variant="beta1", side="left")
        self.assertEqual(spec.source_commit, "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95")
        self.assertEqual(len(spec.usd_sha256), 64)
        self.assertEqual(len(spec.asset_tree_sha256), 64)
        with self.assertRaises(ValueError):
            load_hand2_spec(ROOT, variant="beta3", side="left")

    def test_tree_hash_includes_referenced_layers_and_audited_mjcf(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            body = Path(directory)
            (body / "usd/left").mkdir(parents=True)
            (body / "meshes/left").mkdir(parents=True)
            (body / "mjcf").mkdir()
            layer = body / "usd/left/configuration.usd"
            layer.write_bytes(b"physics A")
            (body / "mjcf/left.xml").write_bytes(b"limits A")
            initial = _asset_tree_sha256(body, "left")
            layer.write_bytes(b"physics B")
            self.assertNotEqual(_asset_tree_sha256(body, "left"), initial)
            layer.write_bytes(b"physics A")
            (body / "mjcf/left.xml").write_bytes(b"limits B")
            self.assertNotEqual(_asset_tree_sha256(body, "left"), initial)


if __name__ == "__main__":
    unittest.main()
