"""Structured qualification and staged-publication regression controls."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from dexgym.tasks.direct.wholebody_s05.data import hmb_build
from dexgym.tasks.direct.wholebody_s05.data.p01e_ledger import Ledger
from dexgym.tasks.direct.wholebody_s05.data.p01e_lock import authorize,RUN_BASE
from dexgym.tasks.direct.wholebody_s05.data.p01e_qualification import qualify_partition
from test_p01d_hmb import synthetic


def evidence():
    source=dict(recording_id='source-a',root_episode_id='root-a',episode_id='e0',collector_instance_id='c0',
                manifest_sha256='m',snapshot_sha256='s',calibration_sha256='cal',assets={},
                topologies=[{'stream':'manus_left'},{'stream':'manus_right'}])
    def pair(side):
        return dict(side=side,matched=50,translation_m={'max':0.},rotation_rad={'max':0.},failures=[],
                    raw_unpaired=0,registered_unpaired=0,ambiguous_header_stamps=0)
    witnessed=dict(**deepcopy(source),status='BLOCKED',issues=['Host recorder mentioned in diagnostic text only'],
        source_read={s:{'invalid_schema_rows':0} for s in (*hmb_build.STREAMS,'tracker_safety')},
        capture_qualification='SIMULATED',run_clocks=[{'max_abs_recorder_residual_ns':10}],
        calibrated_side_pairs=[pair('left'),pair('right')],
        production_witness=dict(invalid_geometry=[],observed_clock_generations=['collector-monotonic-v1'],
           monotonic_regressions={},known_registered_invalid=0,
           body_calibration={'00007':[pair('left'),pair('right')]}))
    block=dict(label='00007',start_ns=56_000_000_000,stop_ns=56_140_000_000,tail=True,
               tail_inheritance_max_ns=1_250_000_000,scope_digest='binding')
    policy=dict(archived_clock_pair_limit_ns=1_000_000,calibration_error_limits={'m':1e-5,'rad':1e-5},
                min_registered_pairs_per_side=20)
    return source,witnessed,block,policy


class E0Qualification(unittest.TestCase):
    def qualify(self,source,witness,block,policy,inherited=None):
        return qualify_partition(source,witness,block,policy,['collector-monotonic-v1'],inherited)

    def test_positive_schema_and_unpaired_evidence_despite_string(self):
        s,w,b,p=evidence()
        w['production_witness']['known_registered_invalid']=4
        w['calibrated_side_pairs'][0]['raw_unpaired']=2
        q=self.qualify(s,w,b,p)
        self.assertEqual(q.status,'ELIGIBLE')
        self.assertIn('REGISTERED_INVALID_SAMPLES',q.retained)

    def test_tail_inherits_only_same_binding_and_never_bad_residual(self):
        s,w,b,p=evidence()
        for pair in w['production_witness']['body_calibration'][b['label']]:
            pair['matched']=12
        self.assertEqual(self.qualify(s,w,b,p).status,'INSUFFICIENT')
        prior=dict(recording_id=s['recording_id'],scope_digest=b['scope_digest'],classification='ELIGIBLE',
                   full_block=True,stop_ns=b['start_ns'],manifest_sha256='full')
        q=self.qualify(s,w,b,p,prior)
        self.assertEqual((q.status,q.inherited_from),('TAIL_INHERITED','full'))
        self.assertEqual(self.qualify(s,w,b,p,dict(prior,scope_digest='other')).status,'INSUFFICIENT')
        self.assertEqual(self.qualify(s,w,dict(b,tail=False),p,prior).status,'INSUFFICIENT')
        w['production_witness']['body_calibration'][b['label']][0]['rotation_rad']['max']=0.1
        self.assertEqual(self.qualify(s,w,b,p,prior).status,'HARD_FAILURE')

    def test_unapproved_sources_schemas_topologies_and_clocks(self):
        for expected,change in (
            ('SOURCE_SNAPSHOT_CALIBRATION_BINDING_MISMATCH',lambda w:w.update(snapshot_sha256='different')),
            ('SCHEMA_OR_REQUIRED_GEOMETRY_FIELD_INVALID',lambda w:w['source_read']['manus_left'].update(invalid_schema_rows=1)),
            ('TOPOLOGY_NOT_APPROVED',lambda w:w['topologies'].append({'stream':'manus_left','version':'new'})),
            ('TIME_OR_CLOCK_GENERATION_UNAPPROVED',lambda w:w['production_witness'].update(observed_clock_generations=['new'])),
            ('GEOMETRY_UNIT_OR_COORDINATE_UNPROVEN',lambda w:w['production_witness'].update(invalid_geometry=['unproven'])),
        ):
            with self.subTest(expected=expected):
                s,w,b,p=evidence()
                w['issues']=[]
                change(w)
                self.assertIn(expected,self.qualify(s,w,b,p).hard)


class PublicationControls(unittest.TestCase):
    def test_nonfinite_stage_fails_before_exposing_destination(self):
        with TemporaryDirectory() as temp:
            dest=Path(temp)/'shard'
            original=hmb_build.build_arrays
            def corrupt(*a,**kw):
                arr=original(*a,**kw)
                arr['pose'][arr['node_mask']]=np.nan
                return arr
            with patch.object(hmb_build,'build_arrays',corrupt):
                with self.assertRaisesRegex(ValueError,'Nonfinite valid pose'):
                    hmb_build.write_partition(dest,**synthetic())
            self.assertFalse(dest.exists())
            self.assertTrue(list(dest.parent.glob('.shard.*.partial')))

    def test_recovery_revalidates_semantics_even_if_all_hashes_match(self):
        with TemporaryDirectory() as temp:
            dest=Path(temp)/'shard'
            arguments=synthetic()
            hmb_build.write_partition(dest,**arguments)
            path=dest/'offline.npz'
            with np.load(path,allow_pickle=False) as z:
                data={k:z[k] for k in z.files}
            data['interpolation_weight'][data['pose_valid']]=np.nan
            np.savez_compressed(path,**data)
            mf=dest/'manifest.json'
            obj=json.loads(mf.read_bytes())
            obj['files'][path.name]=hmb_build.digest(path)
            mf.write_bytes(hmb_build.json_bytes(obj))
            hmb_build.inspect_partition(dest)
            with self.assertRaisesRegex(ValueError,'Offline partition lacks safe auditable brackets'):
                hmb_build.write_partition(dest,**arguments)

    def test_failed_pause_resume_ledger_and_nonterminal_index(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'steps.jsonl'
            plan=[{'key':'r/0'}]
            original=Ledger(path,'E0',plan)
            for state in ('RUNNING','FAILED','PAUSED'):
                original.add('r/0',state)
            with self.assertRaises(ValueError):
                original.completed_index()
            resume=Ledger(path,'E0',plan)
            for state in ('RESUMED','RUNNING','PUBLISHED'):
                resume.add('r/0',state)
            self.assertEqual(resume.completed_index()[0]['status'],'PUBLISHED')
            self.assertEqual(len(path.read_bytes().splitlines()),7)

    def test_e0_authorization_cannot_start_full_build(self):
        with self.assertRaisesRegex(ValueError,'E1 requires separate committed human approval'):
            authorize('E1_PRODUCTION',RUN_BASE/'P01-E-NOT-AUTHORIZED/products','x','y')
