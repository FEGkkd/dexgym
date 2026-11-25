"""Reproduce the legacy CollectionStatus failure with independent decoders."""
from hashlib import sha256
import json
from pathlib import Path
import sys

from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from dexgym.tasks.direct.wholebody_s05.data.registry_gate import digest


def attempt(decoder, data):
    try:
        value=decoder(data)
        return {'status':'PASS','fields':list(value.__slots__)}
    except Exception as error:
        return {'status':'FAIL','detail':f'{type(error).__name__}: {error}'}


p = Path(sys.argv[1])
fixtures = json.loads((p/'fixtures_manifest.json').read_bytes())['recordings']
fixture = next(r for r in fixtures if r['locator'].endswith('/teleop_no_camera_20261016_133930'))
part = next(a for a in fixture['parts'] if any(
    c['topic']=='/dex/collection/status' for c in a['expected_channels']))
source = Path(part['path'])
if digest(source)!=part['sha256']:
    raise ValueError('MCAP fixture changed')
with source.open('rb') as stream:
    bag = NonSeekingReader(stream,validate_crcs=True)
    for i,(schema,ch,msg) in enumerate(bag.iter_messages(log_time_order=False)):
        if ch.topic!='/dex/collection/status':
            continue
        typ=get_message(schema.name)
        result=dict(recording_id=fixture['recording_id'],part_id=part['part_id'],part_sha256=part['sha256'],
                    physical_message=i,schema_sha256=sha256(schema.data).hexdigest(),
                    cdr_sha256=sha256(msg.data).hexdigest(),topic=ch.topic,
                    ros_topic_type_hash=ch.metadata.get('topic_type_hash'),
                    embedded_ros2msg=attempt(DecoderFactory().decoder_for(ch.message_encoding,schema),msg.data),
                    installed_fast_cdr_diagnostic=attempt(lambda raw:deserialize_message(raw,typ),msg.data))
        print(json.dumps(result,indent=2,sort_keys=True))
        break
    else:
        raise ValueError('Catalogued CollectionStatus not found')
