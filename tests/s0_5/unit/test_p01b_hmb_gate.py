"""Dependency-gate controls. The review runner separately checks the real recordings."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from dexgym.tasks.direct.wholebody_s05.data.hmb_gate import evaluate


ROOT=Path(__file__).resolve().parents[3]
MATRIX=json.loads((ROOT/'docs/s0-5/evidence/P01_B_review/topic_dependency_matrix.json').read_bytes())
RECORDING='real'
COLLECTOR='recorded-collector'
BODY_INDEX=('part_id','asset_sha256','topic','physical_message','schema_sha256')


def stamped(values):
    return {k:{'value':v} for k,v in values.items()}


class HmbGateTest(unittest.TestCase):
    def setUp(self):
        self.fixture={'recording_id':RECORDING,'format':'dex-native-v2','verified_writer':True,
                      'root_episode_id':'root','parts':[]}
        self.registry={'id':RECORDING,'status':'ADMITTED_CANDIDATE','producer_kind':'LIVE_TELEOP',
                       'root_episode_id':'root','collector_instance':COLLECTOR}
        self.source={'recording_id':RECORDING,'raw_human_source_verified':True,
                     'level':'RAW_SOURCE_CONFIRMED_FIELDS_SEPARATE'}
        self.events=[]
        self.raw={}
        for rule in MATRIX['streams']:
            if rule['class']!='HMB_REQUIRED':
                continue
            for n,topic in enumerate(rule['topics']):
                obs_topic=rule['observation']['topics'][n]
                part='part-'+str(len(self.events))
                digest='a'*64
                self.fixture['parts'].append({'part_id':part,'sha256':digest})
                pointer=dict(zip(BODY_INDEX,(part,digest,obs_topic,1,rule['observation']['schema_sha256'])))
                obs={'stream_id':obs_topic.rsplit('/',1)[-1],'source_instance_id':part,
                     'episode_sample_index':0,'source_sequence_id':7,'host_realtime_ns':25,
                     'host_observed_monotonic_ns':15,'receive_ros_timestamp_ns':18}
                c=stamped(dict(mcap_log=25,mcap_publish=25,ros_header=14,collector_realtime=25,
                               collector_monotonic=15,collector_receive=18,sdk_source=1,sdk_receive=2))
                pairing={'status':'VERIFIED_WRITER_PAIR','observation_index':pointer,
                         'collector_instance_id':COLLECTOR,'source_instance_id':part,
                         'episode_sample_index':0,'source_sequence_id':7}
                row=dict(recording_id=RECORDING,segment='BODY',decode_status='PASS',clock=c,
                         cdr_sha256='b'*64,pairing=pairing,
                         index=dict(zip(BODY_INDEX,(part,digest,topic,0,rule['schema_sha256']))))
                pos={'x':0,'y':0,'z':0}
                rot={'x':0,'y':0,'z':0,'w':1}
                if rule['role']=='manus':
                    row['fields']={'raw_node_count':1}
                    row['manus']=dict(node_count=1,issues=[],original_node_order=[0],
                                      nodes=[dict(node_id=0,parent_node_id=0,original_order=0,joint_type='palm',
                                                  chain_type='ROOT',node_position=[0,0,0],node_orientation=[0,0,0,1])])
                elif rule['role']=='tracker_safety':
                    row['fields']={'safe':True,'code':'TRACKER_OK'}
                else:
                    row['fields']={'valid':True,'role':topic.rsplit('/',1)[-1],'quality':1.0,
                                   'header':{'frame_id':'openvr_world'},'pose':{'position':pos,'orientation':rot}}
                self.raw[topic]=row
                self.events.extend((row,{'recording_id':RECORDING,'segment':'BODY','index':pointer,
                                         'decode_status':'PASS','fields':obs,'clock':deepcopy(c)}))

    def result(self, calibrations=(RECORDING,)):
        return evaluate([self.fixture],self.events,{RECORDING:self.registry},{RECORDING:self.source},
                        MATRIX,set(calibrations))

    def add_failed_auxiliary(self):
        self.events.append(dict(recording_id=RECORDING,segment='BODY',decode_status='CDR_DECODE_FAILED',
                                index={'part_id':'aux','physical_message':4,'topic':'/dex/collection/status'}))

    def test_positive_bounded_decode_cannot_admit_training(self):
        result=self.result()
        self.assertEqual(result['technical_hmb_gate'],'PASS')
        self.assertFalse(result['fixtures'][0]['formal_training_eligible'])

    def test_manus_cdr_failure_is_failure(self):
        self.raw['/manus_glove_left']['decode_status']='CDR_DECODE_FAILED'
        self.assertEqual(self.result()['technical_hmb_gate'],'FAIL')

    def test_tracker_cdr_failure_is_failure(self):
        self.raw['/vive/tracker/right_wrist']['decode_status']='CDR_DECODE_FAILED'
        self.assertEqual(self.result()['technical_hmb_gate'],'FAIL')

    def test_safety_field_missing_cannot_pass(self):
        self.raw['/vive/tracker/safety_status']['fields'].pop('safe')
        self.assertEqual(self.result()['technical_hmb_gate'],'FAIL')

    def test_sync_fields_or_observation_missing_cannot_pass(self):
        self.raw['/manus_glove_right']['clock'].pop('collector_receive')
        self.assertEqual(self.result()['technical_hmb_gate'],'BLOCKED')

    def test_missing_calibration_blocks(self):
        self.assertEqual(self.result(())['technical_hmb_gate'],'BLOCKED')

    def test_source_unknown_is_isolated_despite_decoded_payloads(self):
        self.source['raw_human_source_verified']=False
        result=self.result()
        self.assertEqual(result['technical_hmb_gate'],'BLOCKED')
        self.assertEqual(result['fixtures'][0]['status'],'ISOLATED_SOURCE')

    def test_independent_provenance_keeps_auxiliary_failure_separate(self):
        self.add_failed_auxiliary()
        self.assertEqual(self.result()['technical_hmb_gate'],'PASS')

    def test_auxiliary_cited_as_required_proof_blocks(self):
        self.add_failed_auxiliary()
        self.fixture['required_proof_topics']=['/dex/collection/status']
        self.assertEqual(self.result()['technical_hmb_gate'],'BLOCKED')
