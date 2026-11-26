"""Final P01-C regression controls for trusted offline barrier semantics."""
from dataclasses import replace
import unittest

from dexgym.tasks.direct.wholebody_s05.data.sync_evidence import json_result
from dexgym.tasks.direct.wholebody_s05.data.synchronize import OfflineAlignment,batch_causal
from test_p01c_geometry_sync import PARAMS,event,safe


STREAM='manus_left'


def unavailable(e):
    return replace(e,quality='UNUSABLE',capture=None,arrival=None)


class FinalBoundaryTests(unittest.TestCase):
    def sample(self,events,tick=15):
        return OfflineAlignment(events,PARAMS).step(tick)['streams'][STREAM]

    def test_unusable_trusted_unsafe_blocks_offline_bracket(self):
        deny=unavailable(safe(2,15,False))
        events=[safe(1,0),event(1,10),deny,event(2,20,pos=2)]
        row=self.sample(events)
        self.assertFalse(row['valid'])
        self.assertEqual(row['reason'],'UNSAFE')
        self.assertIsNone(deny.capture)
        later=events+[safe(3,25),event(3,27,pos=3),event(4,33,pos=4)]
        self.assertFalse(self.sample(later,26)['valid'])
        self.assertTrue(self.sample(later,30)['valid'])

    def test_unusable_trusted_invalid_pose_blocks_offline_bracket(self):
        invalid=unavailable(replace(event(2,15),valid=False))
        events=[safe(1,0),event(1,10),invalid,event(3,20,pos=2),event(4,24,pos=3)]
        row=self.sample(events)
        self.assertFalse(row['valid'])
        self.assertEqual(row['reason'],'UNSYNC')
        self.assertIsNone(row['velocity'])
        self.assertTrue(self.sample(events,22)['valid'])

    def test_untrusted_pose_cannot_be_endpoint(self):
        events=[safe(1,0),event(1,10),replace(event(2,20,pos=2),authenticated=False)]
        self.assertFalse(self.sample(events)['valid'])

    def test_untrusted_safety_cannot_grant_permission(self):
        events=[replace(safe(1,0),authenticated=False),event(1,10),event(2,20,pos=2)]
        self.assertFalse(self.sample(events)['valid'])

    def test_same_record_flag_conflicts_are_order_independent_and_deny(self):
        base=event(2,20,pos=2)
        for field in ('valid','fresh','calibrated'):
            with self.subTest(field=field):
                conflict=replace(base,**{field:False})
                rows=[safe(1,0),event(1,10),base,conflict]
                for supplied in (rows,rows[::-1]):
                    self.assertFalse(self.sample(supplied)['valid'])
                left=batch_causal(rows,[(21,21)],PARAMS,'SIMULATED')
                right=batch_causal(rows[::-1],[(21,21)],PARAMS,'SIMULATED')
                self.assertEqual(json_result(left),json_result(right))
                self.assertFalse(left[0]['streams'][STREAM]['valid'])

    def test_obsolete_pose_does_not_block_a_new_segment(self):
        late_old=replace(unavailable(event(0,4)),known_ns=15,valid=False)
        rows=[safe(1,0),event(1,10),late_old,event(2,20,pos=2)]
        self.assertTrue(self.sample(rows)['valid'])

    def test_known_unsafe_recovery_requires_post_release_pose(self):
        rows=[safe(1,0),event(1,10),safe(2,15,False),event(2,18),safe(3,20),
              event(3,22),event(4,28,pos=2)]
        self.assertFalse(self.sample(rows,21)['valid'])
        self.assertTrue(self.sample(rows,26)['valid'])

    def test_clock_generation_cannot_bridge_an_unknown_transition(self):
        rows=[safe(1,0),event(1,10),
              replace(unavailable(event(2,15)),clock_generation='next'),
              replace(event(3,20,pos=2),clock_generation='next'),
              replace(event(4,24,pos=3),clock_generation='next')]
        self.assertFalse(self.sample(rows,15)['valid'])
        self.assertTrue(self.sample(rows,22)['valid'])
