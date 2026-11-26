"""Reconstruct every emitted causal index and sample one original CDR witness."""
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest

from mcap.reader import NonSeekingReader
import numpy as np

from dexgym.tasks.direct.wholebody_s05.data.registry_gate import digest,load_rows


class ActualCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=os.environ.get('S05_PC_PRODUCTS')
        if not folder:
            raise unittest.SkipTest('Requires isolated Ubuntu output and real source input')
        cls.path=Path(folder)
        cls.trace=load_rows(cls.path/'source_trace.jsonl')

    def read(self,name):
        return json.loads((self.path/name).read_bytes())

    def test_prefix_mutation_and_same_streaming_state_for_every_real_window(self):
        out=self.read('prefix_mutation_results.json')
        self.assertGreaterEqual(len(out['rows']),21)
        for result in out['rows']:
            self.assertTrue(all(result[k] for k in ('future_deleted_equal',
                       'future_pose_clock_valid_safety_changed_equal','streaming_equals_batch','future_anchor_equal')))
            self.assertGreater(result['future_event_vulnerability_controls'],0)
        self.assertEqual(self.read('clock_mapping_report.json')['clock_eligibility'],'BLOCKED')

    def test_each_valid_publication_rebuilds_held_pose_from_raw_index_and_bounds(self):
        original={(r['recording_id'],r['window'],r['stream'],r['index']['raw_index_token']):r['row']
                  for r in self.trace if r['kind']=='raw'}
        by_window={}
        for r in self.read('aligned_causal_sample_manifest.json')['rows']:
            key=r['recording_id'],r['window']
            a_model=r['file']
            raw_path=next(k['file'] for k in self.read('aligned_causal_sample_manifest.json')['raw_windows']
                          if (k['recording_id'],k['window'])==key)
            self.assertEqual(digest(self.path/a_model),r['sha256'])
            with np.load(self.path/a_model,allow_pickle=False) as file:
                poses=file['pose_m_xyzw']; masks=file['node_mask']; ticks=file['tick_ns']
                sidx={s:i for i,s in enumerate(r['streams'])}
                index={int(v):i for i,v in enumerate(ticks)}
            with np.load(self.path/raw_path,allow_pickle=False) as file:
                raw_poses=file['pose_m_xyzw']
            by_window[key]=(poses,masks,index,sidx,raw_poses)
        checked=0
        for r in self.trace:
            if r['kind']!='aligned' or not r['causal']['valid']:
                continue
            c=r['causal']; q=c['ref']
            self.assertLessEqual(c['capture'][1],r['tick_ns'])
            self.assertLessEqual(c['arrival'][1],r['cutoff_ns'])
            self.assertEqual(c['age_ns'],r['tick_ns']-c['capture'][0])
            i=original[(r['recording_id'],r['window'],r['stream'],q['raw_index_token'])]
            out,mask,time,side,raw=by_window[(r['recording_id'],r['window'])]
            a=out[time[r['tick_ns']],side[r['stream']]][mask[time[r['tick_ns']],side[r['stream']]]]
            if not r['stream'].startswith('manus'):
                np.testing.assert_array_equal(a,raw[i,:len(a)])
            else:
                from scipy.spatial.transform import Rotation
                raw_hand=raw[i,:len(a)]
                topology=self.read('manus_topology_report.json')['topologies']
                z=next(t for t in topology if t['recording_id']==r['recording_id'] and
                        t['stream']==r['stream'] and t['version']==c['version'][1])
                palm=next(n['original_order'] for n in z['joints'] if n['parent_node_id'] in {-1,n['node_id']})
                rr=Rotation.from_quat(raw_hand[:,3:])
                np.testing.assert_allclose(a[:,:3],rr[palm].inv().apply(raw_hand[:,:3]-raw_hand[palm,:3]),atol=1e-12)
                self.assertLess(float((Rotation.from_quat(a[:,3:]).inv()*rr[palm].inv()*rr).magnitude().max()),1e-12)
            checked+=1
        self.assertGreater(checked,500)

    def test_sampled_original_mcap_geometry_body_and_source_sha(self):
        actual=next(r for r in self.trace if r['kind']=='raw' and r['window']=='middle' and r['stream']=='manus_left')
        assets={r['id']:r for r in load_rows(Path(os.environ['S05_PC_OLD_A'])/'registry.jsonl')}
        ix=actual['index']
        raw=Path(assets[ix['part_id']]['canonical_uri'])
        self.assertEqual(digest(raw),ix['asset_sha256'])
        with raw.open('rb') as f:
            records=NonSeekingReader(f,validate_crcs=True).iter_messages(log_time_order=False)
            n=ix['physical_message']
            for i,(sc,ch,m) in enumerate(records):
                if i==n:
                    self.assertEqual(sha256(sc.data).hexdigest(),ix['schema_sha256'])
                    self.assertEqual(sha256(m.data).hexdigest(),ix['cdr_sha256'])
                    self.assertEqual(ch.topic,ix['topic'])
                if i==ix['observation_message']:
                    self.assertEqual(sha256(m.data).hexdigest(),ix['observation_cdr_sha256'])
                    break
            else:
                self.fail('Frozen record indexes not present in original MCAP')
