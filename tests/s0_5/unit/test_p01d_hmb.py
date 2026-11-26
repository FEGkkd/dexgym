"""Qualification, immutable masks, and build authority tests."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dexgym.tasks.direct.wholebody_s05.data.hmb_build import (SCHEMA,USE,BASE,SYNC_HASH,json_bytes,lock_sync,
    read_partition,schema,segment_ranges,write_partition)
from dexgym.tasks.direct.wholebody_s05.data.p01d_freeze import approved
from dexgym.tasks.direct.wholebody_s05.data.p01d_probes import development_split,narrower_offline_gap
from dexgym.tasks.direct.wholebody_s05.data.synchronize import CORE_STREAMS,OfflineAlignment,batch_causal
from test_p01c_geometry_sync import PARAMS,event,safe


def synthetic():
    rows=[safe(1,0),safe(2,16,False),safe(3,18)]
    rows.extend(event(i,t,s,pos=i) for s in CORE_STREAMS for i,t in enumerate((2,8,12,21,24,28,34),1))
    result=[]
    for e in rows:
        tag=dict(e.ref,raw_index_token=f'{e.stream}/{e.identity.sequence}',asset_sha256='a'*64)
        e=replace(e,ref=tag)
        result.append((e,dict(raw_pose=e.pose,scalar_identity={'sequence':e.identity.sequence},clock={},fields={})))
    events=[e for e,_ in result]
    ticks=[(t,t) for t in range(3,33)]
    online=batch_causal(events,ticks,PARAMS,'SIMULATED')
    align=OfflineAlignment(events,PARAMS)
    off=[align.step(t) for t,_ in ticks]
    return dict(identity={'root_episode_id':'root:fixture','allowed_use':USE},prepared=result,
                topologies=[],causal=online,offline=off,alignment=align,min_segment=2)


class HMBTests(unittest.TestCase):
    def test_locked_sync_and_schema(self):
        lock_sync()
        self.assertEqual(schema()['schema'],SCHEMA)
        self.assertNotIn('VERIFIED',schema()['vocabularies']['quality'])

    def test_round_trip_and_atomic_failure(self):
        with TemporaryDirectory() as td:
            dest=Path(td)/'product'
            args=synthetic()
            with self.assertRaisesRegex(RuntimeError,'atomic publication'):
                write_partition(dest,**args,fail_before_publish=True)
            self.assertFalse(dest.exists())
            m,_=write_partition(dest,**args)
            self.assertEqual(write_partition(dest,**args),(m,0))
            ca,_,src=read_partition(dest,'causal')
            off,_,_=read_partition(dest,'offline')
            self.assertFalse(ca['safety_permission'][ca['tick_ns']==16][0])
            self.assertFalse(off['pose_valid'][off['tick_ns']==17].any())
            self.assertTrue(ca['pose_valid'].any())
            self.assertTrue(ca['velocity_valid'].any())
            for ix in np.unique(ca['source_index'][ca['pose_valid']]):
                if ix>=0:
                    self.assertEqual(src[ix]['record']['asset_sha256'],'a'*64)
            with (dest/'sources.jsonl').open('ab') as f:
                f.write(b' ')
            with self.assertRaises(ValueError):
                read_partition(dest,'offline')

    def test_segment_does_not_cross_hazard(self):
        args=synthetic()
        found=segment_ranges(args['causal'],args['alignment'],2)
        self.assertTrue(found)
        for lo,hi in found:
            selected=args['causal'][lo:hi]
            self.assertTrue(all(x['safety_permission'] for x in selected))
            self.assertFalse(selected[0]['tick_ns']<16<=selected[-1]['tick_ns'])

    def test_gap_sweep_mask_equivalent_to_replaying_core(self):
        args=synthetic()
        for limit in (2,4,5,60):
            narrow=OfflineAlignment([e for e,_ in args['prepared']],dict(PARAMS,gap_ns=limit))
            self.assertEqual([narrower_offline_gap(t,limit) for t in args['offline']],
                             [narrow.step(t['tick_ns'])['joint_valid'] for t in args['offline']])

    def test_split_is_reproducible_disjoint_and_retains_history(self):
        by_parent=[dict(root_episode_id=f'r{i}',recording_id=f'a{i}',task='bi' if i%2 else 'arm',
                         capture_date='12',configs={'teleop':{'revision':'same'}},calibration_sha256='k') for i in range(8)]
        usages={'r0':{'REPEATED_TEST'}}
        first=development_split(by_parent,usages,12)
        second=development_split(by_parent[::-1],usages,12)
        self.assertEqual(first,second)
        train={x['root_episode_id'] for x in first['Train']}
        validation={x['root_episode_id'] for x in first['Validation']}
        self.assertFalse(train&validation)
        self.assertEqual(first['Final_Test']['roots'],[])
        self.assertTrue(any(x['historical_roles']==['REPEATED_TEST'] for x in first['Train']+first['Validation']))

    def test_full_builder_requires_separate_approval(self):
        with TemporaryDirectory() as td:
            cfg=Path(td)/'frozen_config.json'
            cfg.write_bytes(json_bytes(dict(schema='s05.p01d.freeze_candidate.v1',allowed_use=USE,
                                           sync_baseline=BASE,sync_source_sha256=SYNC_HASH)))
            with self.assertRaisesRegex(ValueError,'no approved freeze'):
                approved(cfg,Path('docs/s0-5/phase_state.json'),Path(td)/'must-not-exist')
            self.assertFalse((Path(td)/'must-not-exist').exists())
