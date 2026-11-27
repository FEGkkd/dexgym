import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from hmb_cohort import INPUT_ROLES, asset_path, cohort


DOCS = Path(__file__).resolve().parents[2] / "docs/s0-5"
CARVEOUT = "capture:8c85b5bd93595270f134bf72"


class HmbCohortTest(unittest.TestCase):
    def test_exact_signed_population(self):
        observed = cohort(DOCS)
        self.assertEqual(observed["selected_root_count"], 32)
        self.assertEqual(observed["signed_raw_asset_count"], 224)
        self.assertEqual(observed["signed_human_input_asset_count"], 128)
        self.assertEqual(sum(root["planned_ticks"] for root in observed["selected_roots"]), 116_613)
        self.assertEqual(sum(asset["bytes"] for root in observed["selected_roots"] for asset in root["inputs"]),
                         2_019_065_488)
        for root in observed["selected_roots"]:
            self.assertEqual([asset["role"] for asset in root["inputs"]], list(INPUT_ROLES))
            self.assertTrue(all(not row["observation_is_replay_input"] for row in root["inputs"]))
        self.assertFalse(observed["registry_checked"])
        self.assertFalse(observed["native_assets_sha256_checked"])

    def test_root_selection_retains_original_population(self):
        observed = cohort(DOCS, root_selection=[CARVEOUT])
        root, = observed["selected_roots"]
        self.assertEqual(root["root"], CARVEOUT)
        self.assertEqual(root["partitions"], 7)
        self.assertEqual(root["planned_ticks"], 2_549)
        self.assertEqual(observed["signed_cohort_denominator"]["Train"]["planned_ticks"], 86_338)
        self.assertEqual(root["hmb_partitions"][0]["archive_relative_path"], "Train/capture_8c85b5bd93595270f134bf72/00000")
        with self.assertRaisesRegex(ValueError, "not signed HMB"):
            cohort(DOCS, root_selection=["capture:not-in-cohort"])

    def test_source_bytes_and_registry_must_agree(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.mcap"
            payload = b"small synthetic asset"
            path.write_bytes(payload)
            asset = {"id": "test", "sha256": sha256(payload).hexdigest(), "bytes": len(payload)}
            registry = {"test": {"canonical_uri": str(path), "file_sha256": asset["sha256"], "size_bytes": len(payload)}}
            result = {}
            asset_path(result, asset, registry, True)
            self.assertEqual(result["asset_check"], "SHA256_AND_SIZE_MATCH")
            path.write_bytes(payload.replace(b"small", b"other"))
            with self.assertRaisesRegex(ValueError, "Actual MCAP bytes differ"):
                asset_path({}, asset, registry, True)


if __name__ == "__main__":
    unittest.main()
