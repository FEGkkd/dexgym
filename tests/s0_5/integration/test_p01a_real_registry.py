"""Verify the real Ubuntu metadata run and its traceable sources."""
import hashlib
import json
import os
from pathlib import Path
import unittest


class RealRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get('S05_P01A_REGISTRY')
        if not value:
            raise unittest.SkipTest('explicit real Ubuntu Run ID required')
        cls.path = Path(value)
        cls.inventory = json.loads((cls.path/'source_inventory.json').read_text())
        cls.records = [json.loads(line) for line in (cls.path/'registry.jsonl').read_text().splitlines()]
        cls.by_id = {row['id']:row for row in cls.records}

    def test_published_hashes_and_status_conservation(self):
        hashes=json.loads((self.path/'output_sha256.json').read_text())
        self.assertEqual(set(hashes), {p.name for p in self.path.iterdir()}-{'output_sha256.json'})
        for name,digest in hashes.items():
            self.assertEqual(hashlib.sha256((self.path/name).read_bytes()).hexdigest(),digest,name)
        counts=self.inventory['counts']
        self.assertEqual(len(self.records),len(self.by_id))
        self.assertEqual(len(self.records),sum(sum(v.values()) for v in counts['status_counts'].values()))
        views=[r for r in self.records if r['record_type']=='recording']
        assets=[r for r in self.records if r['record_type']=='asset']
        self.assertEqual(len(views),counts['logical_recording_views'])
        self.assertEqual(len(assets),counts['mcap_instances'])
        self.assertEqual(sum(r['part_count'] for r in views),len(assets))
        self.assertTrue(all(self.by_id[a['source_id']]['record_type']=='recording' for a in assets))
        self.assertTrue(all(a['asset_id']==a['id'] for a in assets))
        self.assertTrue(all('recorded_interval_ns' in r and 'type_schema_sha256' in r for r in views))

    def test_native_original_and_symlink_alias_are_tied(self):
        rows=[r for r in self.records if r['record_type']=='recording'
              and r.get('episode_ref')=='18e74c646013f1e6-2156b79bc2482857']
        self.assertGreaterEqual(len(rows),2)
        originals=[r for r in rows if r['source_id']=='collector_native']
        self.assertEqual(len(originals),1)
        original=originals[0]
        self.assertEqual(original['format'],'dex-native-v2')
        self.assertTrue(all(original['capabilities']['human_topics'].values()))
        self.assertTrue(any(r.get('alias_of')==original['id'] for r in rows))
        body=next(self.by_id[i] for i in original['source_asset_ids']
                  if self.by_id[i]['role']=='streams/manus_left')
        with Path(body['canonical_uri']).open('rb') as mcap:
            self.assertEqual(mcap.read(8),b'\x89MCAP0\r\n')
            mcap.seek(0)
            self.assertEqual(hashlib.file_digest(mcap,'sha256').hexdigest(),body['file_sha256'])

    def test_legacy_telemetry_is_capability_only_and_sim_excluded(self):
        candidates=[r for r in self.records if r['record_type']=='recording']
        real=next(r for r in candidates if r['locator'].endswith('/teleop_no_camera_20261016_133930'))
        sim=next(r for r in candidates if r['locator'].endswith('/s0_sim_20261117_153201'))
        self.assertEqual(real['format'],'ros2-merged-mcap')
        self.assertTrue(real['capabilities']['robot_topic_capability']['arm_command'])
        self.assertFalse(real['capabilities']['system_id_certified'])
        self.assertEqual(sim['producer_kind'],'SIM_REPLAY')
        self.assertIn(sim['status'],('EXCLUDED','DUPLICATE_ALIAS'))
        final=json.loads((self.path/'final_test_eligibility.json').read_text())
        self.assertEqual((final['status'],final['eligible_root_ids']),('BLOCKED',[]))
        self.assertTrue(all(r['producer_kind']=='LIVE_TELEOP' for r in candidates
                            if r['status']=='ADMITTED_CANDIDATE'))

    def test_real_historical_metadata_and_pending_witness(self):
        hist=self.inventory['history']
        self.assertEqual(set(hist),{'wholebody_s0','wholebody_s0_1','wholebody_s0_2','marvin54','wholebody_s04'})
        for family in hist:
            self.assertGreater(hist[family].get('metadata_documents',0),0,family)
        usage=[json.loads(line) for line in (self.path/'prior_use.jsonl').read_text().splitlines()]
        self.assertTrue(any(x.get('family')=='wholebody_s0' and x['role']=='TRAIN' for x in usage))
        lineage=[json.loads(line) for line in (self.path/'lineage.jsonl').read_text().splitlines()]
        self.assertTrue(any(x['relationship']=='EXACT_COPY' for x in lineage))
        self.assertTrue(all({'source_asset_ids','evidence_ids','split'} <= x.keys() for x in lineage))
        self.assertTrue(all({'usage_id','artifact_path_or_uri','observed_role','root_episode_id',
                             'source_evidence'} <= x.keys() for x in usage))
        self.assertEqual(len({x['usage_id'] for x in usage}),len(usage))
        pending=(self.path/'pending_verification.jsonl').read_text()
        self.assertTrue(pending.strip())


if __name__ == '__main__':
    unittest.main()
