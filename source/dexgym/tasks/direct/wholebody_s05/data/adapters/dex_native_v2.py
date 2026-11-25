"""Register native stream parts and verify recorder-authored observation pairs."""
from pathlib import Path

from ..raw_events import CLOCKS
from .ros2_merged_mcap import decode_part

COLLECTOR_CLOCKS = ('collector_receive','collector_monotonic','collector_realtime')


class PairValidator:
    def __init__(self, stream_id, source_instance, collector_instance, raw_topic, observation_topic,
                 segment, writer_verified):
        self.stream_id = stream_id
        self.source_instance = source_instance
        self.collector_instance = collector_instance
        self.raw_topic = raw_topic
        self.observation_topic = observation_topic
        self.segment = segment
        self.writer_verified = writer_verified
        self.next_index = 0
        self.sequence = None

    def pair(self, payload, observation):
        if self.segment=='INITIAL_STATE':
            return {'status':'INITIAL_STATE_ONLY','reasons':['PRE_EPISODE_CACHE']}
        errors = []
        if not self.writer_verified or not self.collector_instance or not self.source_instance:
            errors.append('WRITER_OR_INSTANCE_UNATTESTED')
        if payload is None or observation is None:
            return {'status':'UNVERIFIED','reasons':errors+['PAIR_MEMBER_MISSING']}
        lhs,rhs = payload['index'],observation['index']
        obs = observation.get('fields') or {}
        if (lhs['part_id']!=rhs['part_id'] or lhs['physical_message']+1!=rhs['physical_message'] or
                lhs['topic']!=self.raw_topic or rhs['topic']!=self.observation_topic):
            errors.append('WRITER_ORDER_MISMATCH')
        if (obs.get('stream_id')!=self.stream_id or obs.get('source_instance_id')!=self.source_instance):
            errors.append('OBSERVATION_STREAM_OR_INSTANCE_MISMATCH')
        seq,episode_index = obs.get('source_sequence_id'),obs.get('episode_sample_index')
        if (not isinstance(seq,int) or seq<1 or self.sequence is not None and seq<=self.sequence or
                episode_index!=self.next_index):
            errors.append('INDEX_OR_SEQUENCE_MISMATCH')
        log = payload['clock']['mcap_log']['value']
        if log!=observation['clock']['mcap_log']['value'] or log!=obs.get('host_realtime_ns'):
            errors.append('RECORDER_TIME_MISMATCH')
        if payload['decode_status']!='PASS' or observation['decode_status']!='PASS':
            errors.append('PAIR_DECODE_UNAVAILABLE')
        if errors:
            return {'status':'UNVERIFIED','reasons':errors}
        self.next_index += 1
        self.sequence = seq
        return dict(status='VERIFIED_WRITER_PAIR',reasons=[],observation_index=rhs,
                    episode_sample_index=episode_index,source_sequence_id=seq,
                    source_instance_id=self.source_instance,collector_instance_id=self.collector_instance)


def stream_parts(manifest, assets, recording):
    """Group P01-A manifest parts without losing segment or writer boundary."""
    result = {}
    for asset in assets:
        pieces = Path(asset['role']).parts
        if len(pieces)!=2 or pieces[0] not in {'streams','initial_state'}:
            raise ValueError(f"Unrecognized native part role: {asset['role']}")
        phase,stream = pieces
        entry = manifest.get('streams',{}).get(stream,{})
        if not entry or not entry.get('topic') or not entry.get('observation_topic'):
            raise ValueError(f'No recorded writer definition: {stream}')
        if phase=='streams' and not entry.get('recorded'):
            raise ValueError(f'Unexpected unrecorded stream: {stream}')
        order = Path(asset['uri_rel']).stem.rsplit('_',1)[-1]
        if not order.isdecimal():
            raise ValueError(f"Unknown writer-part order: {asset['uri_rel']}")
        key = (phase,stream)
        if key not in result:
            result[key] = {'stream':stream,'segment':'BODY' if phase=='streams' else 'INITIAL_STATE',
                           'entry':entry,'assets':[], 'recording_id':recording['id']}
        result[key]['assets'].append((int(order),asset))
    for group in result.values():
        ordered = sorted(group['assets'],key=lambda a:a[0])
        if [i for i,_ in ordered]!=list(range(len(ordered))):
            raise ValueError(f"Missing or duplicate writer parts: {group['stream']}")
        group['assets'] = [a for _,a in ordered]
    present = set(result)
    for stream, entry in manifest.get('streams',{}).items():
        if entry.get('recorded') and entry.get('written',0)>0 and ('streams',stream) not in present:
            raise ValueError(f'Manifest body stream without parts: {stream}')
        if entry.get('initial_state_path') and ('initial_state',stream) not in present:
            raise ValueError(f'Manifest initial state without parts: {stream}')
    return result


def decode_stream(group, manifest, writer_verified, bank, per_channel, record_limit):
    entry = group['entry']
    topics = {entry['topic'],entry['observation_topic']}
    validator = PairValidator(group['stream'],entry.get('source_instance_id'),manifest.get('collector_instance_id'),
                              entry['topic'],entry['observation_topic'],group['segment'],writer_verified)
    for part in group['assets']:
        pending = None
        for item in decode_part(part,bank,topics,per_channel,record_limit,native=writer_verified):
            item.update(segment=group['segment'],stream=group['stream'],recording_id=group['recording_id'])
            if item['index']['topic']==entry['topic']:
                if pending is not None:
                    pending['pairing'] = validator.pair(pending,None)
                    yield pending
                pending = item
                continue
            association = validator.pair(pending,item)
            item['pairing'] = association
            if pending is not None:
                pending['pairing'] = association
                if association['status']=='VERIFIED_WRITER_PAIR':
                    for key in COLLECTOR_CLOCKS:
                        assert key in CLOCKS
                        pending['clock'][key] = item['clock'][key]
                yield pending
                pending = None
            yield item
        if pending is not None:
            pending['pairing'] = validator.pair(pending,None)
            yield pending
