"""Read bounded native windows with actual writer and physical-record provenance."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import subprocess
from time import perf_counter

from mcap.reader import NonSeekingReader
import yaml

from .hmb_gate import dependency_index
from .raw_events import clocks,manus_geometry,message_fields
from .registry_gate import digest
from .schema_registry import EmbeddedDecoders


def frozen_windows(start,stop,plan,unsafe_ns=None):
    w=plan['width_ns']
    if stop-start<4*w:
        raise ValueError('Recording too short for disjoint head/mid/tail windows')
    intervals=[
        ('head',start+plan['head_offset_ns']),
        ('middle',(start+stop-w)//2),
        ('tail',stop-plan['tail_margin_ns']-w),
    ]
    if unsafe_ns is not None and not any(lo<=unsafe_ns<lo+w for _,lo in intervals):
        intervals.append(('unsafe_event',max(start,min(stop-w,unsafe_ns-w//2))))
    output={}
    for label,left in intervals:
        if left<start or left+w>stop:
            raise ValueError('Frozen window outside recorder envelope')
        output[label]=(left,left+w)
    return output


def clock_observations(rows,label):
    """Same-observation host pairs; realtime remains a raw index, not a tick domain."""
    return [(row['clock']['collector_realtime']['value'],
             row['clock']['collector_monotonic']['value'],row['index'])
            for row in rows if label in row['windows']]


def checked_configs(snapshot):
    result={}
    for u in snapshot['units']:
        name=u['unit']
        if name not in {'manus','tracker','teleop'}:
            continue
        if u.get('config_changed_since_start'):
            raise ValueError(f'{name} changed during capture')
        refs=[]
        for ref in u.get('startedConfigurations',[]):
            p=Path(ref['snapshot'])
            if not p.is_file() or digest(p)!=ref['sha256']:
                raise ValueError(f'Capture config SHA mismatch: {p}')
            refs.append(dict(path=str(p),original_path=ref['path'],sha256=ref['sha256']))
        if not refs:
            raise ValueError(f'No capture configuration for {name}')
        result[name]={'revision':u['revision'],'refs':refs}
    if set(result)!={'manus','tracker','teleop'}:
        raise ValueError('Writer/sensor configuration set incomplete')
    return result


def recorded_code(snapshot, writer_conf):
    repos={r['root']:r['commit'] for r in snapshot['sources']}
    teleop='/home/ubuntu/teleop_ws'
    collector=writer_conf['repo']
    paths={
        teleop:['src/manus_ros2/src/ManusDataPublisher.cpp','src/manus_ros2/src/ManusDataPublisher.hpp',
                'src/ManusSDK/include/ManusSDK.h','src/ManusSDK/include/ManusSDKTypes.h',
                'src/vive_tracker_ros/src/vive_tracker_node.cpp','src/vive_tracker_ros/src/tracker_pose_transformer_node.cpp',
                'src/vive_tracker_ros/src/tracker_calibration_node.cpp','src/vive_tracker_ros/src/world_adapter_node.cpp'],
        collector:[writer_conf['path']],
    }
    out={}
    for repo,relatives in paths.items():
        for relative in relatives:
            raw=subprocess.check_output(['git','-C',repo,'show',f'{repos[repo]}:{relative}'])
            content=raw.decode()
            out[relative]={'repo':repo,'commit':repos[repo],'sha256':sha256(raw).hexdigest()}
            if repo==collector and (repos[repo]!=writer_conf['commit'] or out[relative]['sha256']!=writer_conf['sha256']):
                raise ValueError('Unreviewed native writer implementation')
            if relative.endswith('ManusDataPublisher.hpp') and not (
                    'm_WorldSpace = true' in content and 'Side::Side_Right, 1.0f' in content):
                raise ValueError('MANUS global, metre, handedness convention unavailable')
    return out


def verify_source(row,assets,prior,matrix,writer,*,probe_source=False):
    """Validate native metadata. P01-D probes must still earn bounded data eligibility."""
    if (row['status']!='ADMITTED_CANDIDATE' or row['producer_kind']!='LIVE_TELEOP' or
            not row['root_episode_id'] or (not probe_source and (prior or {}).get('status')!='PASS')):
        raise ValueError('P01-C selected an unverified development source')
    folder=Path(row['canonical_uri'])
    manifest,snapshot=folder/'manifest.yaml',folder/'run_snapshot.json'
    for path in (manifest,snapshot):
        if not any(str(path)==x.get('path') and digest(path)==x.get('sha256') for x in row['evidence']):
            raise ValueError(f'Declared source SHA missing: {path}')
    m=yaml.safe_load(manifest.read_bytes())
    s=json.loads(snapshot.read_bytes())
    if (m['state']!='COMPLETED' or m['episode_id']!=row['episode_ref'] or
            m['collector_instance_id']!=row['collector_instance'] or s.get('profile')!='real_no_head'):
        raise ValueError('Recording collector, source or active profile unresolved')
    configs=checked_configs(s)
    code=recorded_code(s,writer)
    pinned=dependency_index(matrix)
    parts=defaultdict(list)
    for part in assets:
        role=part['role']
        if not role.startswith('streams/'):
            continue
        stream=role.split('/')[1]
        decl=m['streams'][stream]
        topic=decl['topic']
        rule=pinned.get(topic,{})
        if rule.get('class')!='HMB_REQUIRED':
            continue
        slot=rule['topics'].index(topic)
        observed=rule['observation']['topics'][slot]
        if (decl['observation_topic']!=observed or decl['type']!=rule['type'] or
                not decl['recorded'] or not decl['source_instance_id'] or not decl['publishers_at_start'] or
                not Path(part['canonical_uri']).resolve().is_relative_to(folder.resolve()) or
                digest(Path(part['canonical_uri']))!=part['file_sha256']):
            raise ValueError(f'Unverified required stream or file: {stream}')
        parts[stream].append((part,decl,rule))
    if {d['topic'] for p in parts.values() for _,d,_ in p}!={t for t,v in pinned.items() if v['class']=='HMB_REQUIRED'}:
        raise ValueError('At least one required human stream is absent')
    if not any(z['sha256'] in {x['sha256'] for x in row['calibration_refs']}
               and z['original_path'].endswith('/tracker_calibration.yaml') for z in configs['tracker']['refs']):
        raise ValueError('Calibration snapshot not linked to A1 evidence')
    details=dict(recording_id=row['id'],root_episode_id=row['root_episode_id'],capture_date=row['capture_date'],
                 task=row['task'],profile=s['profile'],snapshot_sha256=digest(snapshot),manifest_sha256=digest(manifest),
                 configs=configs,code=code,recorded_interval_ns=row['recorded_interval_ns'],
                 collector_instance_id=row['collector_instance'],
                 conditional_dependency_evidence=['raw manifest/snapshot','all active sensor configs',
                                                  'clock-bound writer observation','tracker validity and safety'],
                 collection_status_used=False,clock_clip_ns=[m['start_monotonic_ns'],m['stop_monotonic_ns']])
    return m,parts,details


def scan_stream(stream,parts,windows,context_ns,max_messages, *, find_unsafe=False,measurement=None):
    started=perf_counter()
    bank=EmbeddedDecoders()
    decode_seconds=0.
    decoded_messages=0
    indexed_messages=0
    count=0
    previous_sequence=-1
    kept=[]
    for asset,decl,rule in sorted(parts,key=lambda p:p[0]['part_order']):
        path=Path(asset['canonical_uri'])
        pending=None
        with path.open('rb') as f:
            reader=NonSeekingReader(f,validate_crcs=True,record_size_limit=32*2**20)
            for i,(schema,channel,msg) in enumerate(reader.iter_messages(log_time_order=False)):
                indexed_messages+=1
                if i>=max_messages:
                    raise ValueError(f'Physical index budget reached: {path}')
                topic=channel.topic
                is_raw=topic==decl['topic']
                expected=rule['schema_sha256'] if is_raw else rule['observation']['schema_sha256']
                if not is_raw and topic!=decl['observation_topic']:
                    raise ValueError(f'Unknown data channel in required writer file: {topic}')
                if sha256(schema.data).hexdigest()!=expected or channel.message_encoding!='cdr':
                    raise ValueError(f'Schema drift in required source: {path}')
                if is_raw:
                    if pending is not None:
                        raise ValueError(f'Required payload lacks its adjacent observation: {path}:{i}')
                    count += 1
                    selected=[label for label,(lo,hi) in windows.items() if lo-context_ns<=msg.log_time<hi+context_ns]
                    pending=(i,msg,schema,channel,selected) if selected or find_unsafe else None
                    continue
                if pending is None:
                    continue
                idx,raw,raw_schema,raw_channel,selected=pending
                pending=None
                t=perf_counter() if measurement is not None else 0.
                observed,_=message_fields(bank.decode(channel.message_encoding,schema,msg.data))
                if measurement is not None:
                    decode_seconds+=perf_counter()-t
                if (idx+1!=i or raw.log_time!=msg.log_time or raw.log_time!=observed['host_realtime_ns'] or
                        observed['stream_id']!=stream or observed['source_instance_id']!=decl['source_instance_id'] or
                        observed['episode_sample_index']!=count-1 or observed['source_sequence_id']<=previous_sequence):
                    raise ValueError(f'Writer or sample identity mismatch: {path}:{idx}')
                previous_sequence=observed['source_sequence_id']
                t=perf_counter() if measurement is not None else 0.
                fields,nonfinite=message_fields(bank.decode(raw_channel.message_encoding,raw_schema,raw.data))
                if measurement is not None:
                    decode_seconds+=perf_counter()-t
                    decoded_messages+=2
                if find_unsafe:
                    if fields.get('safe') is False:
                        return raw.log_time
                    continue
                timing=clocks(fields,raw.log_time,raw.publish_time,manus=stream.startswith('manus'),native=True)
                for target,key,domain in (('collector_receive','receive_ros_timestamp_ns','collector_ros_clock'),
                                          ('collector_monotonic','host_observed_monotonic_ns','collector_host_monotonic'),
                                          ('collector_realtime','host_realtime_ns','collector_host_realtime')):
                    timing[target]={'value':observed[key],'domain':domain,'unit':'ns','status':'RECORDED',
                                    'field':f'SampleObservation.{key}'}
                timing['control_tick']={'value':None,'domain':None,'unit':None,'status':'UNKNOWN','field':None}
                ref=dict(part_id=asset['id'],asset_sha256=asset['file_sha256'],schema_sha256=rule['schema_sha256'],
                         cdr_sha256=sha256(raw.data).hexdigest(),physical_message=idx,observation_message=i,
                         observation_schema_sha256=expected,observation_cdr_sha256=sha256(msg.data).hexdigest(),
                         source_instance=observed['source_instance_id'],writer_sequence=observed['source_sequence_id'],
                         topic=decl['topic'])
                row=dict(stream=stream,index=ref,fields=fields,clock=timing,nonfinite=nonfinite,windows=selected)
                if stream.startswith('manus'):
                    row['manus']=manus_geometry(fields,stream.split('_')[1])
                kept.append(row)
        if pending is not None:
            raise ValueError(f'Missing adjacent SampleObservation: {path}')
    if find_unsafe:
        return None
    if count!=decl['written']:
        raise ValueError(f'Manifest writer count changed for {stream}: {count}/{decl["written"]}')
    if measurement is not None:
        measurement.update(wall_seconds=perf_counter()-started,cdr_and_extract_seconds=decode_seconds,
                           cdr_messages=decoded_messages,indexed_messages=indexed_messages,
                           asset_bytes=sum(asset['size_bytes'] for asset,_,_ in parts))
    return kept,count
