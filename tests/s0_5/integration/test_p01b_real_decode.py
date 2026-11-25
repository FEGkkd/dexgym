"""Verify bounded products against actual MCAPs, not fixture-generated messages."""
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest

from mcap.reader import make_reader

from dexgym.tasks.direct.wholebody_s05.data.adapters.ros2_merged_mcap import open_asset
from dexgym.tasks.direct.wholebody_s05.data.registry_gate import digest,load_rows


class RealP01B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.environ.get('S05_P01B_PRODUCTS')
        if not value:
            raise unittest.SkipTest('S05_P01B_PRODUCTS required for the real Ubuntu run')
        cls.path = Path(value)
        cls.events = load_rows(cls.path/'raw_events.jsonl')
        cls.fixtures = cls.read('fixtures_manifest.json')['recordings']
        cls.raw = {p['part_id']:p for r in cls.fixtures for p in r['parts']}
        cls.by_id = {r['recording_id']:r for r in cls.fixtures}

    @classmethod
    def read(cls,name):
        return json.loads((cls.path/name).read_bytes())

    def test_input_identity_output_hashes_and_byte_denominators(self):
        sums = self.read('output_sha256.json')
        self.assertEqual(set(sums),{p.name for p in self.path.iterdir()}-{'output_sha256.json'})
        for name,key in sums.items():
            self.assertEqual(digest(self.path/name),key,name)
        counts = self.read('dataset_coverage.json')['inherited_a_counts']
        self.assertEqual((counts['mcap_instances'],counts['unique_byte_payloads'],
                          counts['logical_recording_views'],counts['canonical_recordings']),(651,382,181,122))
        registry=self.read('schema_registry.json')
        self.assertEqual(len(registry['part_map']),651)
        self.assertEqual(len(registry['payloads']),382)
        self.assertEqual(self.read('adapter_contract.json')['preflight_A1']['gate'],'PASS')

    def test_native_geometry_with_record_references_and_observation_pairs(self):
        sample = next(x for x in self.events if x.get('manus',{}).get('node_count')==25 and
                      x.get('pairing',{}).get('status')=='VERIFIED_WRITER_PAIR')
        geo = sample['manus']
        self.assertEqual(geo['original_node_order'],list(range(25)))
        self.assertTrue(all(n['joint_type'] and n['chain_type'] for n in geo['nodes']))
        self.assertEqual(geo['nodes'][0]['node_id'],geo['nodes'][0]['parent_node_id'])
        p = self.raw[sample['index']['part_id']]
        self.assertEqual(self.by_id[sample['recording_id']]['format'],'dex-native-v2')
        self.assertEqual(digest(Path(p['path'])),p['sha256'])
        with open_asset({'canonical_uri':p['path']}) as stream:
            reader = make_reader(stream,validate_crcs=True)
            for i,(schema,channel,record) in enumerate(reader.iter_messages(log_time_order=False)):
                if i==sample['index']['physical_message']:
                    self.assertEqual(channel.topic,sample['index']['topic'])
                    self.assertEqual(sha256(schema.data).hexdigest(),sample['index']['schema_sha256'])
                    self.assertEqual(sha256(record.data).hexdigest(),sample['cdr_sha256'])
                    break
            else:
                self.fail('record index not found in original MCAP')
        found_obs = next(x for x in self.events if x['index']==sample['pairing']['observation_index'])
        self.assertEqual(sample['clock']['collector_realtime']['value'],found_obs['fields']['host_realtime_ns'])
        self.assertIsNone(sample['clock']['controller_receive']['value'])
        initial = [e for e in self.events if e['segment']=='INITIAL_STATE']
        self.assertTrue(initial)
        self.assertTrue(all(e['pairing']['status']=='INITIAL_STATE_ONLY' for e in initial))

    def test_old_hand_state_actual_cdr_arrays_and_schema(self):
        merged = next(r for r in self.fixtures if r['locator'].endswith('teleop_no_camera_20261016_133930'))
        rows = [x for x in self.events if x['recording_id']==merged['recording_id'] and
                 x['index']['topic'] in {'/left_hand/state','/right_hand/state'}]
        self.assertEqual({r['index']['topic'] for r in rows},{'/left_hand/state','/right_hand/state'})
        self.assertTrue(all(r['decode_status']=='PASS' and len(r['fields']['position'])==20 for r in rows))
        self.assertTrue(all(len(r['fields']['joint_names'])==20 and
                            'home_sequence' in r['fields'] and 'current' not in r['fields'] for r in rows))
        self.assertTrue(all(r['index']['schema_sha256'] for r in rows))

    def test_diagnostic_and_replay_remain_outside_training_and_system_id(self):
        eligibility = load_rows(self.path/'source_eligibility_updates.jsonl')
        self.assertEqual(len(eligibility),181)
        lookup = {r['recording_id']:r for r in eligibility}
        sim = next(r for r in self.fixtures if r['locator'].endswith('s0_sim_20261117_153201'))
        self.assertEqual(lookup[sim['recording_id']]['level'],'EXCLUDED')
        self.assertTrue(all(not r['formal_training_eligible'] for r in eligibility))
        self.assertEqual(self.read('dataset_coverage.json')['final_test'],'BLOCKED')
        self.assertEqual(len(self.read('telemetry_eligibility.json')['records']),181)

    def test_i17_values_preserved_and_raw_differences_not_applied_as_clock(self):
        report = self.read('raw_timestamp_report.json')
        self.assertFalse(report['mapping_applied'])
        repeats = [row for row in report['rows'] if row['clock_statistics']['sdk_source']['repeat_steps']]
        self.assertTrue(repeats)
        self.assertTrue(all(row['clock_statistics']['sdk_source']['unit'] is None for row in repeats))
        self.assertGreater(len({r['recording_id'] for r in report['rows']}),2)


if __name__ == '__main__':
    unittest.main()
