"""HMB actor read boundaries and archive identity negative controls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from dexgym.tasks.direct.wholebody_s05.hmb_reader import (
    ACTOR_FIELDS, HMBReader, SIMULATED, actor_features, eligible_history,
)


def fixture():
    return dict(pose=np.ones((9,6,2,7)), node_mask=np.ones((9,6,2),dtype=bool),
                velocity_m_s=np.zeros((9,6,2,3)), velocity_age_ns=np.zeros((9,6),dtype=int),
                pose_valid=np.ones((9,6),dtype=bool), velocity_valid=np.ones((9,6),dtype=bool),
                age_ns=np.zeros((9,6),dtype=int), safety_permission=np.ones(9,dtype=bool),
                joint_valid=np.ones(9,dtype=bool), tick_ns=np.arange(9)*20_000_000,
                segment_id=np.zeros(9,dtype=int), clock_generation=np.zeros((9,6),dtype=int),
                calibration_id=np.zeros((9,6),dtype=int), topology_id=np.zeros((9,6),dtype=int),
                offline_interpolation=np.ones((9,6)), future_motion_label=np.ones(9))


class HMBReaderTests(unittest.TestCase):
    def test_prefix_only_and_feature_allowlist(self):
        a=fixture()
        original=actor_features(a,4)
        self.assertEqual(tuple(original),ACTOR_FIELDS)
        self.assertEqual(original['pose'].shape,(5,4,2,7))
        for name,data in a.items():
            data[5:]=0
            if name not in ACTOR_FIELDS:
                data[:]=0
        changed=actor_features(a,4)
        for name in original:
            np.testing.assert_array_equal(original[name],changed[name])
            np.testing.assert_array_equal(original[name],actor_features({k:v[:5] for k,v in a.items()},4)[name])
        changed['pose'][:]=4
        self.assertEqual(a['pose'][0,0,0,0],1)

    def test_cannot_read_a_previous_partition(self):
        a=fixture()
        self.assertFalse(eligible_history(a,3))
        self.assertFalse(eligible_history(a,9))
        self.assertTrue(eligible_history(a,4))
        with self.assertRaises(ValueError):
            actor_features(a,3)

    def test_segment_gap_and_versions_are_boundaries(self):
        for field in ('segment_id','tick_ns','clock_generation','calibration_id','topology_id'):
            with self.subTest(field=field):
                a=fixture()
                a[field][2]+=1
                self.assertFalse(eligible_history(a,4))

    def test_unsafe_or_invalid_pose_denies_history(self):
        for field in ('safety_permission','joint_valid','pose_valid'):
            with self.subTest(field=field):
                a=fixture()
                a[field][2]=False
                self.assertFalse(eligible_history(a,4))

    def test_valid_pose_does_not_grant_velocity_history(self):
        a=fixture()
        a['velocity_valid'][2,3]=False
        self.assertTrue(eligible_history(a,4))
        self.assertFalse(eligible_history(a,4,require_velocity=True))

    def test_unapproved_archive_and_legacy_npz_are_denied(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'
            for scope in ('D_BOUNDED_PILOT','E0_TRANSIENT_NON_PRODUCTION','E1_PRODUCTION'):
                p.write_text(json.dumps({'scope':scope,'schema':'s05.hmb.simulation.v1'}))
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    HMBReader(p,p,p,d,'Train',SIMULATED)
            p.write_bytes(b'PK\x03\x04old npz')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                HMBReader(p,p,p,d,'Train',SIMULATED)

    def test_final_test_and_verified_purpose_are_denied(self):
        for split,purpose in [('Final_Test',SIMULATED),('Train','HMB_REALTIME_VERIFIED')]:
            with self.assertRaises(ValueError):
                HMBReader('none','none','none','none',split,purpose)
