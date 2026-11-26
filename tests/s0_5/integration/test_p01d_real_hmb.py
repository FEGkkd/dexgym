"""Trace the actual HMB output back to immutable raw MCAP."""
from hashlib import sha256
import json,os,unittest
from pathlib import Path

import numpy as np
from mcap.reader import NonSeekingReader

from dexgym.tasks.direct.wholebody_s05.data.hmb_build import USE,read_partition
from dexgym.tasks.direct.wholebody_s05.data.registry_gate import digest,load_rows


class RealHMB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if 'S05_PD_PRODUCTS' not in os.environ:
            raise unittest.SkipTest('Actual scoped P01-D study output required')
        cls.root=Path(os.environ['S05_PD_PRODUCTS'])
        cls.assets={r['id']:r for r in load_rows(Path(os.environ['S05_PD_A1'])/'registry.jsonl')}

    def read(self,name):
        return json.loads((self.root/name).read_bytes())

    def test_whitelist_and_disjoint_lineage(self):
        findings=self.read('source_qualification.json')
        w=self.read('source_whitelist.json')['sources']
        s=self.read('split.json')
        self.assertEqual((len(findings['rows']),findings['A1_unknown_quarantined']),(32,89))
        self.assertTrue(all(r['issues'] for r in findings['rows'] if r['status']!='QUALIFIED'))
        self.assertEqual({r['root_episode_id'] for r in w},
                         {r['root_episode_id'] for r in findings['rows'] if r['status']=='QUALIFIED'})
        train={r['root_episode_id'] for r in s['Train']}
        val={r['root_episode_id'] for r in s['Validation']}
        self.assertFalse(train&val)
        self.assertEqual(train|val,{r['root_episode_id'] for r in w})
        self.assertEqual(s['Final_Test']['roots'],[])
        self.assertFalse(self.read('qc_threshold_report.json')['validation_used_for_selection'])

    def test_three_production_shards_trace_actual_cdr(self):
        pilot=self.read('resource_budget.json')['pilot']
        self.assertEqual(len(pilot['partitions']),3)
        for shard in pilot['partitions']:
            path=self.root/'pilot-serial'/shard['name']
            c,m,source=read_partition(path,'causal')
            o,_,_=read_partition(path,'offline')
            self.assertTrue(m['identity']['episode_id'])
            self.assertEqual(m['identity']['allowed_use'],USE)
            np.testing.assert_array_equal(c['tick_ns'],o['tick_ns'])
            ptr=int(c['source_index'][c['pose_valid'][:,0],0,0][0])
            ref=source[ptr]['record']
            file=Path(self.assets[ref['part_id']]['canonical_uri'])
            self.assertEqual(digest(file),ref['asset_sha256'])
            with file.open('rb') as f:
                rdr=NonSeekingReader(f,validate_crcs=True,record_size_limit=32*2**20)
                sc,_,payload=next(row for i,row in enumerate(rdr.iter_messages(log_time_order=False))
                                  if i==ref['physical_message'])
            self.assertEqual(sha256(sc.data).hexdigest(),ref['schema_sha256'])
            self.assertEqual(sha256(payload.data).hexdigest(),ref['cdr_sha256'])
            self.assertTrue(all(i>=0 and source[int(i)]['safe']
                                for i in c['safety_source_index'][c['safety_permission']]))

    def test_prefix_and_atomic_pilot_controls(self):
        report=self.read('qc_threshold_report.json')['window_results']
        self.assertTrue(report)
        for r in report:
            p=r['prefix']
            self.assertTrue(all(p[k] for k in ('future_deleted_equal','future_pose_clock_valid_safety_changed_equal',
                                               'streaming_equals_batch','future_anchor_equal')))
        p=self.read('resource_budget.json')['pilot']
        self.assertTrue(p['failed_target_hidden'] and p['idempotent_recovery'] and p['parallel_equal_sha'])
