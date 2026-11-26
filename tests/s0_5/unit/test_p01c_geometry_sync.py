"""Geometry, timing, and actual streaming boundary controls for P01-C."""
from dataclasses import replace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.data.calibration import verify_pairs
from dexgym.tasks.direct.wholebody_s05.data.clock_domains import Anchor,PrefixClock,qualify
from dexgym.tasks.direct.wholebody_s05.data.geometry import compose,inverse,hand_geometry,rotation
from dexgym.tasks.direct.wholebody_s05.data.sample_identity import Identity,IdentityHistory
from dexgym.tasks.direct.wholebody_s05.data.sync_evidence import compare_prefix
from dexgym.tasks.direct.wholebody_s05.data.synchronize import CausalSynchronizer,Event,OfflineAlignment,STREAMS


PARAMS={'age_ns':60,'skew_ns':35,'safety_age_ns':100,'gap_ns':60}


def event(seq,t,stream='manus_left',pos=0,safe=None,recv=None,epoch='same',source=None):
    identity=Identity(epoch,seq,t if source is None else source,t if recv is None else recv,repr(pos))
    return Event(stream,identity,{'physical_message':seq},t,(t,t),(t,t),
                 'SIMULATED',np.array([[pos,0,0,0,0,0,1.]],dtype=float),
                 safe=safe,version=('a','h'),authenticated=True)


def safe(seq,t,value=True):
    return event(seq,t,'tracker_safety',safe=value)


def nodes():
    return [
        dict(node_id=i,parent_node_id=i-1 if i else 0,original_order=i,joint_type='Tip',
             chain_type='Thumb' if i else 'Hand',node_position=[i*.01,0,0],node_orientation=[0,0,0,1])
        for i in range(4)
    ]


class GeometryTest(unittest.TestCase):
    def test_calibration_order_rotation_inverse_and_determinant(self):
        a=np.r_[[1,0,0],Rotation.from_euler('z',90,degrees=True).as_quat()]
        b=np.array([.1,0,0,0,0,0,1.])
        p=compose(a,b)
        np.testing.assert_allclose(p[:3],[1,.1,0],atol=1e-12)
        np.testing.assert_allclose(compose(p,inverse(p)),[0,0,0,0,0,0,1],atol=1e-12)
        rr=rotation(p[3:]).as_matrix()
        np.testing.assert_allclose(rr.T@rr,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rr),1)
        with self.assertRaises(ValueError):
            rotation([0,0,0,0])

    def test_self_root_and_common_frame_bone_length(self):
        z,palm,bones,edges=hand_geometry(nodes(),'left','Left')
        self.assertEqual(z.shape,(4,7))
        np.testing.assert_allclose(bones,[.01]*3)
        np.testing.assert_allclose(palm[0,:3],0)

    def test_cycle_duplicate_missing_parent_and_wrong_hand(self):
        for mutate in (lambda a:a[1].update(parent_node_id=2),lambda a:a[2].update(node_id=1),
                       lambda a:a[1].update(parent_node_id=9)):
            sample=nodes(); mutate(sample)
            with self.assertRaises(ValueError):
                hand_geometry(sample,'left','Left')
        with self.assertRaises(ValueError):
            hand_geometry(nodes(),'left','Right')

    def test_pair_matching_is_exact_not_nearest(self):
        def row(t,x,registered=False):
            return {'index':{'physical_message':t},'fields':{
                'header':{'frame_id':'openvr_world','stamp':{'sec':0,'nanosec':t}},
                'valid':registered,'pose':{'position':{'x':x,'y':0,'z':0},
                'orientation':{'x':0,'y':0,'z':0,'w':1}}}}
        report=verify_pairs([row(1,0),row(3,0)],[row(1,1,True),row(4,1,True)],np.array([1,0,0,0,0,0,1.]),'openvr_world')
        self.assertEqual(report['matched'],1)
        self.assertEqual(report['raw_unpaired'],1)
        self.assertEqual(report['translation_m']['max'],0)


class CausalTest(unittest.TestCase):
    def test_missing_real_time_cannot_be_certified(self):
        self.assertEqual(qualify((0,2),(4,5),'VERIFIED','external capture'), 'UNUSABLE')
        core=CausalSynchronizer(PARAMS,'VERIFIED')
        core.offer(replace(event(1,8),quality='VERIFIED'))
        row=core.step(10,10)['streams']['manus_left']
        self.assertFalse(row['valid'])
        self.assertEqual(row['input_reason'],'UNSYNC')

    def test_nontrivial_capture_and_arrival_bounds(self):
        c=CausalSynchronizer(PARAMS,'SIMULATED')
        c.offer(safe(1,0))
        c.offer(replace(event(1,5),capture=(5,13),arrival=(6,12)))
        self.assertFalse(c.step(10,11)['streams']['manus_left']['valid'])
        self.assertFalse(c.step(10,14)['streams']['manus_left']['valid'])
        self.assertEqual(c.step(14,14)['streams']['manus_left']['age_ns'],9)

    def test_duplicate_does_not_refresh_age_and_stale_ref_preserved(self):
        c=CausalSynchronizer(PARAMS,'SIMULATED')
        for e in (safe(1,0),event(1,1),event(2,30,recv=1,source=1)):
            c.offer(e)
        row=c.step(32,32)['streams']['manus_left']
        self.assertEqual((row['age_ns'],row['input_reason'],row['ref']['physical_message']),
                         (31,'DUPLICATE_SOURCE',1))
        self.assertEqual(c.step(90,90)['streams']['manus_left']['reason'],'STALE')

    def test_same_pose_different_sampling_and_old_packet(self):
        h=IdentityHistory()
        self.assertEqual(h.observe(event(1,1).identity),'NEW')
        self.assertEqual(h.observe(event(2,2).identity),'NEW')
        self.assertEqual(h.observe(event(1,1).identity),'OUT_OF_ORDER')

    def test_late_packet_does_not_roll_back_pose(self):
        c=CausalSynchronizer(PARAMS,'SIMULATED')
        c.offer(safe(1,1)); c.offer(event(2,10,pos=2)); c.step(12,12)
        c.offer(replace(event(1,2,pos=1),known_ns=14,arrival=(14,14)))
        row=c.step(15,15)['streams']['manus_left']
        self.assertEqual(row['input_reason'],'OUT_OF_ORDER')
        self.assertEqual(row['pose'][0,0],2)
        c.offer(replace(event(4,7,pos=4),known_ns=16,arrival=(16,16)))
        self.assertEqual(c.step(16,16)['streams']['manus_left']['reason'],'CLOCK_JUMP')
        c.offer(event(5,17,pos=3))
        self.assertEqual(c.step(17,17)['streams']['manus_left']['pose'][0,0],3)

    def test_source_restart_discards_history_and_clock_jump_rebuilds(self):
        c=CausalSynchronizer(PARAMS,'SIMULATED')
        for e in (safe(1,0),event(1,2),event(1,5,epoch='new'),event(2,6,epoch='new'),
                  event(3,8,epoch='new',recv=4),event(4,10,epoch='new',recv=11)):
            c.offer(e)
        self.assertTrue(c.step(2,2)['streams']['manus_left']['valid'])
        self.assertEqual(c.step(5,5)['streams']['manus_left']['reason'],'SOURCE_RESTART')
        self.assertTrue(c.step(6,6)['streams']['manus_left']['valid'])
        self.assertEqual(c.step(8,8)['streams']['manus_left']['reason'],'CLOCK_JUMP')
        self.assertTrue(c.step(10,10)['streams']['manus_left']['valid'])

    def test_safety_invalid_and_expiry(self):
        c=CausalSynchronizer(PARAMS,'SIMULATED')
        for e in (event(1,2),safe(1,4),safe(2,10,False),event(2,11),safe(3,20),event(3,119)):
            c.offer(e)
        self.assertEqual(c.step(3,3)['streams']['manus_left']['reason'],'SAFETY_UNKNOWN')
        self.assertFalse(c.step(5,5)['streams']['manus_left']['valid'])
        self.assertEqual(c.step(12,12)['streams']['manus_left']['reason'],'UNSAFE')
        self.assertEqual(c.step(122,122)['streams']['manus_left']['reason'],'SAFETY_UNKNOWN')

    def test_unknown_caches_do_not_initialize(self):
        core=CausalSynchronizer(PARAMS,'SIMULATED')
        core.offer(safe(1,0)); core.offer(replace(event(2,10),segment='INITIAL_STATE'))
        self.assertEqual(core.step(11,11)['streams']['manus_left']['reason'],'SOURCE_UNKNOWN')

    def test_asynchronous_sides_and_tracker_validity(self):
        c=CausalSynchronizer(PARAMS,'SIMULATED')
        for i,s in enumerate(STREAMS):
            if s!='tracker_right':
                c.offer(event(i,10,s))
        c.offer(safe(2,0))
        self.assertFalse(c.step(15,15)['joint_valid'])
        c.offer(replace(event(7,20,'tracker_right'),valid=False))
        self.assertEqual(c.step(22,22)['streams']['tracker_right']['reason'],'TRACKER_INVALID')
        self.assertFalse(c.step(60,60)['joint_valid'])

    def test_future_mutation_and_online_equivalence(self):
        events=[e for i in range(1,40) for e in (safe(i,i),event(i,i,pos=i))]
        _,stats=compare_prefix(events,[(n,n) for n in range(4,38)],PARAMS)
        for k in ('future_deleted_equal','future_pose_clock_valid_safety_changed_equal',
                  'streaming_equals_batch','future_anchor_equal'):
            self.assertTrue(stats[k])
        self.assertGreater(stats['future_event_vulnerability_controls'],0)

    def test_clock_bound_uses_only_available_anchor(self):
        p=PrefixClock(evidence='verified fixture')
        p.add(Anchor(10,100,110,15,'real test fixture'))
        p.add(Anchor(13,1000,1010,30,'future test fixture'))
        self.assertEqual(p.bounds(20,20),((110,120),'MODEL_BOUND',1))
        self.assertEqual(p.bounds(20,31)[1],'CLOCK_JUMP')

    def test_offline_bracket_across_safety_invalid_version_or_gap_rejected(self):
        a,b=event(1,10),event(2,20,pos=1)
        s=[safe(1,0)]
        self.assertAlmostEqual(OfflineAlignment(s+[a,b],PARAMS).step(15)['streams']['manus_left']['pose'][0,0],.5)
        for extra,reason in (
            ([a,b,safe(2,13,False)],'UNSAFE'),
            ([a,replace(b,valid=False)],'MANUS_INVALID'),
            ([a,replace(b,version=('changed','h'))],'VERSION_BOUNDARY'),
            ([a,event(2,90)],'GAP'),
        ):
            self.assertEqual(OfflineAlignment(s+extra,PARAMS).step(15)['streams']['manus_left']['reason'],reason)
