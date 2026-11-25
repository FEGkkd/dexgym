"""Read ROS2 MCAP in physical message order with embedded type definitions."""
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

from mcap.reader import NonSeekingReader, make_reader

from ..raw_events import clocks, manus_geometry, message_fields
from ..schema_registry import DecodeIssue, identify, schema_record


@contextmanager
def open_asset(asset):
    uri = asset['canonical_uri']
    if asset.get('storage')=='ZIP_MEMBER':
        container, member = uri.split('!',1)
        with ZipFile(container) as archive, archive.open(member) as stream:
            yield stream
    else:
        with Path(uri).open('rb') as stream:
            yield stream


def catalogue_asset(asset):
    """Use MCAP summary/footer. A missing summary remains explicit."""
    with open_asset(asset) as stream:
        summary = make_reader(stream,validate_crcs=True).get_summary()
    if summary is None:
        return [],'SUMMARY_UNAVAILABLE'
    counts = summary.statistics.channel_message_counts if summary.statistics else {}
    output = []
    for channel in summary.channels.values():
        schema = summary.schemas.get(channel.schema_id)
        identity = identify(channel.message_encoding,schema)
        count = counts.get(channel.id,0) if summary.statistics is not None else None
        output.append(dict(channel_id=channel.id,topic=channel.topic,declared_count=count,
                           schema=schema_record(identity,schema) if identity else None))
    return sorted(output,key=lambda c:(c['topic'],c['channel_id'])),'CATALOGUED'


def decode_part(asset, bank, topics, per_channel, record_limit, *, native=False):
    if not topics or per_channel<1 or record_limit<1:
        raise ValueError('A bounded audit needs declared channels and positive limits')
    seen = Counter()
    with open_asset(asset) as stream:
        summary = make_reader(stream,validate_crcs=True).get_summary()
        counts = summary.statistics.channel_message_counts if summary and summary.statistics else {}
        required = {ch.id:min(per_channel,counts.get(ch.id,0) if summary.statistics is not None else per_channel)
                    for ch in summary.channels.values() if ch.topic in topics} if summary else {}
        # Indexed readers can skip unchunked messages when a file mixes record layouts.
        stream.seek(0)
        reader = NonSeekingReader(stream,validate_crcs=True,record_size_limit=64*2**20)
        for i,(schema,channel,message) in enumerate(islice(reader.iter_messages(log_time_order=False),record_limit)):
            if channel.topic not in topics or seen[channel.id]>=per_channel:
                continue
            seen[channel.id] += 1
            identity = identify(channel.message_encoding,schema)
            observation = channel.topic.startswith('/dex/collection/observations/')
            side = channel.topic.removeprefix('/manus_glove_') if channel.topic in {
                '/manus_glove_left','/manus_glove_right'} else None
            index = dict(part_id=asset['id'],part_order=asset['part_order'],asset_sha256=asset['file_sha256'],
                         physical_message=i,topic=channel.topic,channel_id=channel.id,
                         mcap_sequence=message.sequence,
                         schema_sha256=identity.definition_sha256 if identity else None,
                         schema_key=identity.key if identity else None)
            row = dict(index=index,cdr_sha256=sha256(message.data).hexdigest(),fields=None,decode_status='PASS')
            fields = {}
            try:
                decoded = bank.decode(channel.message_encoding,schema,message.data)
                fields,invalid = message_fields(decoded)
                row.update(fields=fields,nonfinite=invalid)
                if side is not None:
                    row['manus'] = manus_geometry(fields,side)
            except DecodeIssue as err:
                row.update(decode_status=err.code,detail=str(err))
            except (TypeError,ValueError,AttributeError) as err:
                row.update(decode_status='RAW_FIELD_CONTRACT_FAILED',detail=str(err))
            row['clock'] = clocks(fields,message.log_time,message.publish_time,manus=bool(side),
                                  observation=observation,native=native)
            yield row
            if required and all(seen[ch_id]>=limit for ch_id,limit in required.items()):
                break
