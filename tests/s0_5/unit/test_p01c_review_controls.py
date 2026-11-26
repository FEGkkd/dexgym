"""Executable review counterexamples, against both the original and revised synchronizers."""
from dataclasses import replace
import unittest

from dexgym.tasks.direct.wholebody_s05.data.clock_domains import Anchor,PrefixClock,qualify
from dexgym.tasks.direct.wholebody_s05.data.sync_evidence import json_result
from dexgym.tasks.direct.wholebody_s05.data.synchronize import CausalSynchronizer,batch_causal
from test_p01c_geometry_sync import PARAMS,event,safe


class ReviewCounterexamples(unittest.TestCase):
    def make(self):
        core=CausalSynchronizer(PARAMS,'SIMULATED')
        core.offer(safe(1,0))
        core.offer(event(1,2))
        self.assertTrue(core.step(4,4)['streams']['manus_left']['valid'])
        return core

    def test_unusable_new_input_revokes_old_validity_but_late_old_is_ignored(self):
        core=self.make()
        core.offer(replace(event(0,1),known_ns=5,quality='UNUSABLE',capture=None,arrival=None))
        self.assertTrue(core.step(5,5)['streams']['manus_left']['valid'])
        core.offer(replace(event(2,7),quality='UNUSABLE',capture=None,arrival=None))
        self.assertFalse(core.step(7,7)['streams']['manus_left']['valid'])

    def test_unknown_safety_revokes_permission(self):
        core=self.make()
        core.offer(replace(safe(2,7),quality='UNUSABLE',capture=None,arrival=None))
        row=core.step(8,8)['streams']['manus_left']
        self.assertFalse(row['valid'])

    def test_unsafe_between_ticks_cannot_resurrect_cached_pose(self):
        core=self.make()
        core.offer(safe(2,6,False))
        core.offer(safe(3,7))
        self.assertFalse(core.step(8,8)['streams']['manus_left']['valid'])
        core.offer(event(2,9,pos=1))
        row=core.step(10,10)['streams']['manus_left']
        self.assertTrue(row['valid'])
        self.assertIsNone(row['velocity'])

    def test_forward_sequence_with_capture_and_clock_rollback(self):
        core=self.make()
        rollback=replace(event(2,1,pos=2),known_ns=8,arrival=(8,8))
        core.offer(rollback)
        row=core.step(8,8)['streams']['manus_left']
        self.assertEqual(row['reason'],'CLOCK_JUMP')
        self.assertFalse(row['valid'])

    def test_no_velocity_after_gap_and_uncertain_dt(self):
        core=self.make()
        core.offer(safe(2,70))
        core.offer(event(2,71,pos=2))
        self.assertIsNone(core.step(75,75)['streams']['manus_left']['velocity'])
        core.offer(replace(event(3,79,pos=3),capture=(78,79)))
        self.assertIsNone(core.step(80,80)['streams']['manus_left']['velocity'])

    def test_proof_strings_cannot_grant_verified(self):
        self.assertEqual(qualify((1,2),(3,4),'VERIFIED','assertion A','assertion B'),'UNUSABLE')

    def test_simultaneous_delivery_is_order_independent(self):
        a=event(1,3,pos=1)
        b=event(2,3,pos=2,recv=4,source=4)
        evidence=[safe(1,0),a,b]
        forward=batch_causal(evidence,[(6,6)],PARAMS,'SIMULATED')[0]
        reverse=batch_causal(evidence[::-1],[(6,6)],PARAMS,'SIMULATED')[0]
        self.assertEqual(json_result(forward),json_result(reverse))

    def test_invalidity_from_untrusted_input_cannot_override_trusted_stream(self):
        core=self.make()
        core.offer(replace(safe(2,7,False),authenticated=False))
        self.assertTrue(core.step(8,8)['safety_permission'])

    def test_latch_blocks_next_tick_even_when_recovery_events_arrive_between_ticks(self):
        core=self.make()
        for e in (safe(2,6,False),safe(3,7),event(2,8,pos=2),event(3,9,pos=3)):
            core.offer(e)
        blocked=core.step(10,10)
        self.assertFalse(blocked['safety_permission'])
        self.assertEqual(blocked['safety_reason'],'UNSAFE')
        recovered=core.step(12,12)['streams']['manus_left']
        self.assertTrue(recovered['valid'])
        self.assertFalse(recovered['velocity_valid'])

    def test_unseen_late_unsafe_latches_and_retransmission_cannot_revoke_twice(self):
        core=self.make()
        core.offer(safe(3,7))
        core.step(7,7)
        late=replace(safe(2,6,False),known_ns=8,arrival=(8,8))
        core.offer(late)
        self.assertEqual(core.step(8,8)['safety_reason'],'UNSAFE')
        core.offer(safe(4,10))
        core.offer(event(2,12))
        self.assertTrue(core.step(12,12)['streams']['manus_left']['valid'])
        core.offer(replace(late,known_ns=14,arrival=(14,14)))
        self.assertTrue(core.step(14,14)['streams']['manus_left']['valid'])

    def test_clock_generation_change_and_late_old_generation(self):
        core=self.make()
        core.offer(replace(event(2,7),clock_generation='second'))
        self.assertEqual(core.step(7,7)['streams']['manus_left']['reason'],'CLOCK_GENERATION')
        core.offer(replace(event(3,9),clock_generation='second'))
        self.assertTrue(core.step(10,10)['streams']['manus_left']['valid'])
        core.offer(event(4,12,pos=5))
        row=core.step(12,12)['streams']['manus_left']
        self.assertEqual(row['input_reason'],'OUT_OF_ORDER')
        self.assertEqual(row['ref']['physical_message'],3)

    def test_held_velocity_has_independent_age_and_quality(self):
        core=self.make()
        core.offer(event(2,9,pos=2))
        self.assertTrue(core.step(10,10)['streams']['manus_left']['velocity_valid'])
        row=core.step(20,20)['streams']['manus_left']
        self.assertEqual((row['reason'],row['velocity_quality'],row['velocity_age_ns']),('HELD','SIMULATED',11))
        self.assertFalse(core.step(71,71)['streams']['manus_left']['velocity_valid'])

    def test_first_window_cache_requires_observed_predecessor(self):
        from dexgym.tasks.direct.wholebody_s05.data.sync_evidence import audit_freshness
        observations=[event(1,2),event(2,3,recv=2,source=2),event(3,4),event(5,5)]
        result=audit_freshness([(e,{}) for e in observations])
        self.assertEqual([e.fresh for e,_ in result],[False,False,True,False])

    def test_proxy_grid_uses_past_offset_and_discloses_future_drift_or_jump(self):
        from dexgym.tasks.direct.wholebody_s05.data.clock_domains import proxy_grid
        observations=[(1000,100,'past'),(1010,109,'drift'),(1050,112,'wall-jump')]
        t,r=proxy_grid(1004,1044,observations,0,20)
        self.assertEqual(t,[(104,104),(124,124)])
        self.assertEqual(r['anchor_ref'],'past')
        self.assertEqual(r['max_abs_recorder_residual_ns'],38)
        self.assertEqual(t,proxy_grid(1004,1044,observations[:1],0,20)[0])

    def test_bounded_drift_and_reboot_require_a_new_anchor_model(self):
        clock=PrefixClock(slope_min=0.9,slope_max=1.1,evidence='unit fixture')
        clock.add(Anchor(100,1000,1010,1,'past'))
        self.assertEqual(clock.bounds(200,9),((1090,1120),'MODEL_BOUND',1))
        clock.add(Anchor(6,5000,5010,10,'reset'))
        self.assertEqual(clock.bounds(200,9),((1090,1120),'MODEL_BOUND',1))
        self.assertEqual(clock.bounds(200,10)[1],'CLOCK_JUMP')
        reboot=PrefixClock(evidence='separately initialized fixture')
        self.assertEqual(reboot.bounds(200,10)[1],'UNUSABLE')
