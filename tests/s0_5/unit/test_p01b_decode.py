"""Tests for P01-B. Real bag checks use the separate integration suite."""
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from mcap.records import Schema
from mcap_ros2.writer import Writer

from dexgym.tasks.direct.wholebody_s05.data.adapters.dex_native_v2 import PairValidator
from dexgym.tasks.direct.wholebody_s05.data.adapters.ros2_merged_mcap import decode_part
from dexgym.tasks.direct.wholebody_s05.data.decode_inputs import native_manifest,relevant
from dexgym.tasks.direct.wholebody_s05.data.decode_audit import schema_coverage
from dexgym.tasks.direct.wholebody_s05.data.raw_events import clocks, manus_geometry, message_fields
from dexgym.tasks.direct.wholebody_s05.data.registry_gate import evaluate_registry, merged_signature, signature_conflicts
from dexgym.tasks.direct.wholebody_s05.data.schema_registry import EmbeddedDecoders, DecodeIssue
from dexgym.tasks.direct.wholebody_s05.data.telemetry import eligibility, source_eligibility


def record(index, topic, data=None, log=100):
    return {'index': {'physical_message': index, 'part_id': 'part', 'topic': topic},
            'fields': data or {}, 'clock': clocks({}, log, 99), 'decode_status': 'PASS'}


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.asset = {'id':'a','record_type':'asset','source_id':'r','part_order':0,'file_sha256':'a'*64,
                      'status':'ADMITTED_CANDIDATE','checksum_state':'VERIFIED','role':'episode'}
        self.row = {'id':'r','record_type':'recording','source_asset_ids':['a'],'part_count':1,
                    'producer_kind':'LIVE_TELEOP','status':'ADMITTED_CANDIDATE',
                    'root_episode_id':'root','calibration_refs':[{'sha256':'f'*64}],
                    'flags':[],'capabilities':{'human_topics': {'left':True,'right':True}}}
        self.lineage = [{'episode_id':'root','parent_ids':[],'root_episode_ids':['root'],'unresolved_refs':[]},
                        {'episode_id':'r','parent_ids':['root'],'root_episode_ids':['root'],'unresolved_refs':[]}]
        self.observed = {'gate_A1':'PASS','counts':{'logical_recording_views':1,'mcap_instances':1}}

    def test_gate_is_not_a_successful_process_exit(self):
        self.assertEqual(evaluate_registry([self.asset,self.row],self.lineage,self.observed)['gate'], 'PASS')
        self.row.update(status='QUARANTINED', producer_kind='UNKNOWN')
        self.assertEqual(evaluate_registry([self.asset,self.row],self.lineage,self.observed)['gate'], 'BLOCKED')

    def test_missing_field_invalid_source_and_conservation_are_failure(self):
        for key,value in (('producer_kind','SIM_REPLAY'),('part_count',2),('status',None),('root_episode_id','missing')):
            invalid = dict(self.row)
            if value is None:
                invalid.pop(key)
            else:
                invalid[key]=value
            self.assertEqual(evaluate_registry([self.asset,invalid],self.lineage,self.observed)['gate'], 'FAIL')
        wrong=dict(self.observed,counts={'mcap_instances':9})
        self.assertEqual(evaluate_registry([self.asset,self.row],self.lineage,wrong)['gate'], 'FAIL')

    def test_incomplete_and_ambiguous_merged_group_has_no_signature(self):
        info = {'storage_identifier':'mcap','relative_file_paths':['one_0.mcap','one_1.mcap'],
                'files':[{'path':'one_0.mcap','message_count':18},
                         {'path':'one_1.mcap','message_count':22}]}
        group = {'parts':['one_0.mcap','one_1.mcap'],'missing':[],'order_source':'sidecar'}
        a = dict(self.asset,verified_message_count=18)
        b = dict(self.asset,file_sha256='b'*64,part_order=1,verified_message_count=22)
        self.assertEqual(merged_signature(info,group,[a,b]),('rosbag2',('a'*64,'b'*64)))
        self.assertIsNone(merged_signature(info,dict(group,missing=['one_1.mcap']),[a]))
        self.assertIsNone(merged_signature(info,dict(group,order_source='filename'),[a,b]))
        self.assertIsNone(merged_signature(info,group,[a,dict(b,verified_message_count=21)]))

    def test_conflicting_declarations_are_visible(self):
        one = {'id':'one','signature':('native','abc'),'producer_kind':'LIVE_TELEOP',
               'evidence':[{'sha256':'f'*64}],'calibration_refs':[{'sha256':'1'*64}]}
        two = dict(one,id='two',producer_kind='SIM_REPLAY',calibration_refs=[{'sha256':'2'*64}])
        conflict = signature_conflicts([one,two])
        self.assertEqual(conflict[0]['record_ids'], ['one','two'])
        self.assertIn('producer_kind',conflict[0]['differences'])
        self.assertIn('calibration_sha256',conflict[0]['differences'])


class DecodingTest(unittest.TestCase):
    def test_failed_attempt_is_never_labelled_unrun(self):
        selected = [{'id':'r','format':'ros2-merged-mcap','source_asset_ids':['part']}]
        registry = {'schemas':{'v':{'name':'demo/msg/Status','sha256':'1'*64,'decoder':'test'}},
                    'part_map':{'part':'payload'},
                    'payloads':{'payload':{'channels':[{'topic':'/dex/collection/status','schema_key':'v',
                                                        'declared_count':1}]}}}
        attempts = [{'recording_id':'r','decode_status':'CDR_DECODE_FAILED','index':{'schema_key':'v'}}]
        self.assertEqual(schema_coverage(registry,selected,attempts)[0]['status'],'DECODE_FAILED')
        self.assertEqual(schema_coverage(registry,selected,[])[0]['status'],'NOT_RUN')

    def test_camera_catalogue_does_not_select_camera_payload_or_observation(self):
        self.assertFalse(relevant('/head/right/image_raw'))
        self.assertFalse(relevant('/dex/collection/observations/head_right_color'))
        self.assertTrue(relevant('/dex/collection/observations/tracker_left_raw'))

    def test_unattested_native_manifest_can_only_be_decoded_diagnostically(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = b'format: dex-native-v2\nepisode_id: original\n'
            (Path(tmp)/'manifest.yaml').write_bytes(blob)
            row = {'canonical_uri':tmp,'episode_ref':'original','status':'QUARANTINED',
                   'evidence':[{'sha256':sha256(blob).hexdigest()},{}]}
            manifest, trusted = native_manifest(row,{})
            self.assertEqual(manifest['episode_id'],'original')
            self.assertFalse(trusted)
            with self.assertRaises(OSError):
                native_manifest(dict(row,status='ADMITTED_CANDIDATE'),{})

    def test_same_type_and_schema_id_in_different_files(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = EmbeddedDecoders()
            results = []
            for i, definition in enumerate(('float64[] position\n', 'float64[] position\nfloat64[] velocity\n')):
                path = Path(directory)/f'{i}.mcap'
                with path.open('wb') as handle, Writer(handle) as writer:
                    schema = writer.register_msgdef('manus_wuji_interfaces/msg/HandState', definition)
                    writer.write_message('/left_hand/state',schema,
                                         {'position':[1.5]*20,'velocity':[2.0]*20},log_time=111, publish_time=44)
                asset = {'id':str(i),'canonical_uri':str(path),'file_sha256':str(i)*64,'part_order':i}
                results.append(list(decode_part(asset,bank,{'/left_hand/state'},8,24))[0])
            self.assertNotEqual(results[0]['index']['schema_sha256'],results[1]['index']['schema_sha256'])
            self.assertNotIn('velocity', results[0]['fields'])
            self.assertEqual(len(results[1]['fields']['velocity']),20)
            self.assertEqual(results[1]['index']['physical_message'],0)
            self.assertEqual(results[1]['clock']['mcap_publish']['value'],44)

    def test_corrupt_cdr_and_unavailable_schema(self):
        bank = EmbeddedDecoders()
        schema=Schema(1, b'uint64 value\n', 'ros2msg', 'demo/msg/Sample')
        with self.assertRaises(DecodeIssue) as error:
            bank.decode('cdr',schema,b'\x00\x01\x00\x00')
        self.assertEqual(error.exception.code,'CDR_DECODE_FAILED')
        with self.assertRaises(DecodeIssue) as error:
            bank.decode('cdr',Schema(1,b'{}','ros2idl','demo/msg/Sample'),b'1234')
        self.assertEqual(error.exception.code,'UNSUPPORTED_SCHEMA')

    def test_manus_topology_variable_and_side_mismatch(self):
        nodes = [dict(node_id=i,parent_node_id=i-1,joint_type='TIP',chain_type='THUMB',
                      pose={'position':{'x':i,'y':0,'z':0},
                            'orientation':{'x':0,'y':0,'z':0,'w':1}}) for i in range(25)]
        fields = {'raw_nodes':nodes,'raw_node_count':25,'side':'left'}
        output = manus_geometry(fields,'left')
        self.assertEqual(output['original_node_order'],list(range(25)))
        self.assertEqual(output['nodes'][24]['parent_node_id'],23)
        self.assertEqual(output['nodes'][24]['joint_type'],'TIP')
        short = manus_geometry(dict(fields,raw_nodes=nodes[:12],raw_node_count=12),'left')
        self.assertNotEqual(output['topology_sha256'],short['topology_sha256'])
        self.assertIn('SIDE_MISMATCH',manus_geometry(fields,'right')['issues'])
        self.assertIn('RAW_NODE_COUNT_MISMATCH', manus_geometry(dict(fields,raw_node_count=7),'left')['issues'])

    def test_no_fabricated_controller_or_device_time(self):
        time = clocks({'source_timestamp_us':8,'receive_timestamp_us':19,
                       'header':{'stamp':{'sec':7,'nanosec':9}}},123,234,manus=True)
        self.assertEqual(time['sdk_source']['value'],8)
        self.assertIsNone(time['sdk_source']['unit'])
        self.assertEqual(time['ros_header']['value'],7_000_000_009)
        self.assertEqual(time['controller_receive']['status'],'UNKNOWN')
        self.assertIsNone(time['source_capture']['value'])

    def test_nonfinite_field_keeps_an_explicit_reason(self):
        class MockMessage:
            __slots__ = ('position',)
        mock = MockMessage()
        mock.position = [1.0,float('nan'),float('inf')]
        fields, issues = message_fields(mock)
        self.assertEqual(fields['position'], [1.0,None,None])
        self.assertEqual([x['path'] for x in issues], ['/position/1','/position/2'])

    def test_observation_is_checked_beyond_identical_log_time(self):
        validator = PairValidator('manus_left','dev123','collector1','/payload','/observation','BODY',True)
        raw = record(0,'/payload')
        meta = {'stream_id':'manus_left','source_instance_id':'dev123','source_sequence_id':41,
                'episode_sample_index':0,'host_realtime_ns':100,'host_observed_monotonic_ns':88,
                'receive_ros_timestamp_ns':90}
        def obs(i, **kwargs):
            return record(i,'/observation',dict(meta,**kwargs))
        self.assertEqual(validator.pair(raw,obs(1))['status'],'VERIFIED_WRITER_PAIR')
        self.assertEqual(validator.pair(record(2,'/payload'),obs(3,episode_sample_index=1))['status'],
                         'UNVERIFIED')
        self.assertEqual(validator.pair(record(2,'/payload'),obs(4))['status'],'UNVERIFIED')
        wrong_side = PairValidator('manus_right','dev123','collector1','/payload','/observation','BODY',True)
        self.assertEqual(wrong_side.pair(raw,obs(1))['status'],'UNVERIFIED')

    def test_cached_initial_state_never_becomes_body_pair(self):
        validator = PairValidator('tracker_safety','dev123','collector1','/payload','/obs','INITIAL_STATE',True)
        self.assertEqual(validator.pair(record(0,'/payload'),record(1,'/obs'))['status'],'INITIAL_STATE_ONLY')

    def test_telemetry_cannot_be_certified_from_array_presence(self):
        self.assertEqual(eligibility(False,{},False,False)['level'],'NO_HARDWARE_PROVENANCE')
        self.assertEqual(eligibility(True,{},False,False)['level'],'UNVERIFIED_PAIR')
        for origin in ('SIM_REPLAY','EGOPRO','DERIVED_ONLY','UNKNOWN'):
            out=source_eligibility({'producer_kind':origin,'status':'EXCLUDED'},False)
            self.assertFalse(out['raw_human_source_verified'])
            self.assertFalse(out['formal_training_eligible'])


if __name__ == '__main__':
    unittest.main()
