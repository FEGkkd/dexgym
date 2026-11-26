"""Independent checks on the frozen raw-MCAP paired replay receipt."""
from collections import Counter
import json
import os
from pathlib import Path
import unittest


class HistoricalReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('S05_CR_PRODUCTS')
        if not path:
            raise unittest.SkipTest('S05_CR_PRODUCTS must name a completed bounded MCAP replay')
        cls.path=Path(path)
        cls.report=json.loads((cls.path/'paired_regression.json').read_bytes())

    def test_fixed_paired_population_and_unconditional_historical_exclusion(self):
        totals=self.report['totals']
        self.assertEqual((totals['ticks'],totals['recordings'],totals['raw_window_events']),(2300,7,47983))
        self.assertEqual((totals['causal_old'],totals['registered_pairs']),(2248,13640))
        self.assertEqual(totals['causal_old']-totals['joint_lost']+totals['joint_gained'],totals['causal_new'])
        d=json.loads((self.path/'technical_gates.json').read_bytes())
        self.assertEqual(d['gates']['C2-RT'],'BLOCKED')
        self.assertEqual(d['verified_control_ticks'],0)
        self.assertFalse(d['qualifies_hardware'])

    def test_source_age_safety_eligibility_and_offline_span(self):
        counts=Counter()
        with (self.path/'source_trace.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                for stream,state in row['causal'].items():
                    self.assertEqual(state['velocity_valid'],state['velocity_quality']=='SIMULATED')
                    if not row['safety_permission']:
                        self.assertFalse(state['valid'])
                    if state['valid']:
                        self.assertLessEqual(state['capture'][1],row['tick_ns'])
                        self.assertLessEqual(state['arrival'][1],row['cutoff_ns'])
                        self.assertGreaterEqual(state['capture'][0],row['safety_release_ns'])
                        self.assertEqual(state['age_ns'],row['tick_ns']-state['capture'][0])
                        self.assertEqual(len(state['ref']['asset_sha256']),64)
                        if state['velocity_valid']:
                            self.assertEqual(state['velocity_age_ns'],state['age_ns'])
                    off=row['offline'][stream]
                    if off['valid']:
                        low,high=off['time_interval']
                        self.assertLessEqual(low,row['tick_ns'])
                        self.assertLessEqual(row['tick_ns'],high)
                        self.assertEqual(len(off['refs']),2)
                    counts[stream]+=1
        self.assertEqual(len(counts),6)
        self.assertEqual(set(counts.values()),{2300})

    def test_every_window_prefix_controls_and_metrics_tags(self):
        controls=('future_deleted_equal','future_pose_clock_valid_safety_changed_equal',
                  'streaming_equals_batch','future_anchor_equal')
        self.assertEqual(len(self.report['results']),23)
        for row in self.report['results']:
            self.assertTrue(all(row['controls'][name] for name in controls))
            self.assertEqual(row['clock']['clock'],'collector_host_monotonic')
            self.assertEqual(len(row['spatial']['identified_tip_nodes']),2)
            self.assertIn('offline_bracket_span',row['spatial']['metrics'])
            self.assertEqual(row['masks']['streams']['tracker_left_raw']['role'],'DIAGNOSTIC_ONLY')
