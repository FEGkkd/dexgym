"""Re-decode only the recorded CollectionStatus failures with hash-matched ROS interfaces."""
import argparse
import base64
from collections import Counter,defaultdict
from hashlib import file_digest,sha256
import inspect
import json
from pathlib import Path
import subprocess

from dex_collection_interfaces.msg import CollectionStatus,StreamStatus
from mcap.reader import NonSeekingReader
from mcap.records import Schema
from mcap_ros2.decoder import DecoderFactory
from rclpy.serialization import deserialize_message
from ament_index_python.packages import get_package_share_directory
from rosidl_generator_type_description import calculate_type_hash

from dexgym.tasks.direct.wholebody_s05.data.raw_events import message_fields


TOPIC='/dex/collection/status'
TYPE='dex_collection_interfaces/msg/CollectionStatus'
NESTED='dex_collection_interfaces/msg/StreamStatus'
EMBEDDED_SHA='b8ded4b4b2291f9db6db32388e48e1d7f692fbaa1bf4bca1c7c47d32f64fab5b'
EXPECTED_TYPE_HASH='RIHS01_ce543f80bf868a5ccd4933e5279f7649b3afc53908f2c67630c48ae1a3d2838c'


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')


def jsonrow(stream,value):
    stream.write(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')


def snapshot(prefix):
    folder=prefix/'share/dex_collection_interfaces/msg'
    result={}
    for name in ('CollectionStatus','StreamStatus'):
        path=folder/f'{name}.json'
        desc=read(path)
        graph=desc['type_description_msg']
        type_name=f'dex_collection_interfaces/msg/{name}'
        recorded=next(d['hash_string'] for d in desc['type_hashes'] if d['type_name']==type_name)
        recomputed=calculate_type_hash(graph)
        if recorded!=recomputed:
            raise ValueError(f'ROS type hash mismatch: {path}')
        text=folder/f'{name}.msg'
        result[name]={'description_path':str(path),'description_sha256':digest(path),
                      'ros_type_hash':recomputed,'msg_path':str(text),'msg_sha256':digest(text),
                      'fields':[f['name'] for f in graph['type_description']['fields']]}
    return result


def portable_ros_message(value):
    if hasattr(value,'get_fields_and_field_types'):
        return {name:portable_ros_message(getattr(value,name)) for name in value.get_fields_and_field_types()}
    if isinstance(value,(list,tuple)):
        return [portable_ros_message(x) for x in value]
    return value


def git_witness(repos, historical):
    result=[]
    for root in repos:
        paths=[path for path in ('src/dex_collection_interfaces/msg/CollectionStatus.msg',
                                'src/dex_collection_interfaces/msg/StreamStatus.msg',
                                'ros2_ws/src/dex_collection_interfaces/msg/CollectionStatus.msg',
                                'ros2_ws/src/dex_collection_interfaces/msg/StreamStatus.msg')
               if (root/path).is_file()]
        found=subprocess.run(['git','-C',str(root),'log','--all','--max-count=24',
                              '--format=%H %as %s','--',*paths],capture_output=True,text=True,check=True)
        revisions=[]
        for line in found.stdout.splitlines():
            sha=line.split()[0]
            for path in paths:
                item=subprocess.run(['git','-C',str(root),'show',f'{sha}:{path}'],capture_output=True)
                if item.returncode==0:
                    name=Path(path).stem
                    observed=sha256(item.stdout).hexdigest()
                    revisions.append({'log':line,'path':path,'sha256':observed,
                                      'matches_historical_install':observed==historical[name]['msg_sha256']})
        result.append({'repo':str(root),'scope':'up to 24 commits touching available two definitions',
                       'results':revisions})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('original','historical-prefix','current-prefix','output','teleop-repo','collector-repo'):
        p.add_argument('--'+key,required=True,type=Path)
    args=p.parse_args()
    target=args.output
    if target.exists() or not target.parent.name.startswith('P01-B-REVIEW-'):
        p.error('new output folder in the isolated review Run ID required')
    target.mkdir(mode=0o700)
    hist=snapshot(args.historical_prefix)
    current=snapshot(args.current_prefix)
    resolved=Path(get_package_share_directory('dex_collection_interfaces')).resolve()
    if resolved!= (args.historical_prefix/'share/dex_collection_interfaces').resolve():
        raise ValueError(f'Historical overlay isolation failed: {resolved}')
    if hist['CollectionStatus']['ros_type_hash']!=EXPECTED_TYPE_HASH:
        raise ValueError('Historical CollectionStatus does not match recorded RIHS')
    compiled={}
    for cls in (CollectionStatus,StreamStatus):
        module=Path(inspect.getfile(cls)).resolve()
        if not module.is_relative_to(args.historical_prefix.resolve()):
            raise ValueError(f'Historical import escaped overlay: {module}')
        if list(cls.get_fields_and_field_types())!=hist[cls.__name__]['fields']:
            raise ValueError(f'Historical generated Python fields disagree: {module}')
        compiled[cls.__name__]={'module':str(module),'sha256':digest(module)}
    if not any((args.historical_prefix/'lib').glob('libdex_collection_interfaces__rosidl_typesupport_fastrtps_c.so')):
        raise ValueError('Historical C typesupport is absent')
    fast=args.historical_prefix/'lib/libdex_collection_interfaces__rosidl_typesupport_fastrtps_c.so'
    compiled['fastrtps_c']={'path':str(fast),'sha256':digest(fast)}
    baseline=read(args.original/'fixtures_manifest.json')
    locations={q['part_id']:(j,q) for j in baseline['recordings'] for q in j['parts']}
    wanted=defaultdict(dict)
    with (args.original/'raw_events.jsonl').open(encoding='utf-8') as raw:
        for line in raw:
            row=json.loads(line)
            index=row['index']
            if index['topic']==TOPIC and row['decode_status']!='PASS':
                if index['schema_sha256']!=EMBEDDED_SHA or index['physical_message'] in wanted[index['part_id']]:
                    raise ValueError('Unexpected historical failure identity')
                wanted[index['part_id']][index['physical_message']]=row
    outcomes, embedded_field_diff, group_rows = Counter(),None,[]
    folder=args.historical_prefix/'share/dex_collection_interfaces/msg'
    top=(folder/'CollectionStatus.msg').read_text(encoding='utf-8').strip()
    nested=(folder/'StreamStatus.msg').read_text(encoding='utf-8').strip()
    # Validate the embedded dependency mismatch and decode using isolated compiled ROS,
    # independent from the archived MCAP embedded payload schema.
    with (target/'legacy_raw_cdr.jsonl').open('x',encoding='utf-8') as originals, \
         (target/'legacy_redecoded_status.jsonl').open('x',encoding='utf-8') as recovered:
        for part_id,indices in sorted(wanted.items()):
            fixture,part=locations[part_id]
            path=Path(part['path'])
            if digest(path)!=part['sha256']:
                raise ValueError(f'MCAP identity mismatch: {path}')
            seen=set()
            local=Counter()
            with path.open('rb') as f:
                for i,(schema,ch,msg) in enumerate(NonSeekingReader(f,validate_crcs=True).iter_messages(log_time_order=False)):
                    if i>max(indices):
                        break
                    if i not in indices:
                        continue
                    old=indices[i]
                    idx=old['index']
                    if (ch.topic!=TOPIC or schema.name!=TYPE or ch.message_encoding!='cdr' or
                            sha256(schema.data).hexdigest()!=idx['schema_sha256'] or
                            sha256(msg.data).hexdigest()!=old['cdr_sha256']):
                        raise ValueError(f'Historical record identity mismatch: {part_id}:{i}')
                    seen.add(i)
                    type_hash=ch.metadata.get('topic_type_hash')
                    reference={'recording_id':old['recording_id'],'part_id':part_id,'asset_sha256':part['sha256'],
                               'physical_message':i,'topic':ch.topic,'embedded_schema_sha256':idx['schema_sha256'],
                               'recorded_ros_type_hash':type_hash,'cdr_sha256':old['cdr_sha256'],
                               'archived_outcome':old['decode_status'],'archived_detail':old.get('detail')}
                    jsonrow(originals,dict(**reference,cdr_base64=base64.b64encode(msg.data).decode('ascii')))
                    try:
                        if type_hash!=hist['CollectionStatus']['ros_type_hash']:
                            raise ValueError(f'No exact historical decoder for ROS type hash {type_hash}')
                        val=portable_ros_message(deserialize_message(msg.data,CollectionStatus))
                        result=dict(**reference,historical_outcome='PASS',fields=val)
                    except Exception as exc:
                        result=dict(**reference,historical_outcome='FAIL',error=f'{type(exc).__name__}: {exc}')
                    if embedded_field_diff is None:
                        definition=schema.data.decode()
                        pre=definition.split('================================================================================',1)[0]
                        segment=definition.split('MSG: dex_collection_interfaces/StreamStatus\n',1)[1]
                        embedded_field_diff={
                            'top_level_wire_fields_match':uncomment(pre)==uncomment(top),
                            'embedded_nested_fields':uncomment(segment),
                            'historical_nested_fields':uncomment(nested),
                            'missing_nested_fields':sorted(set(uncomment(nested))-set(uncomment(segment))),
                        }
                        time=definition.split('MSG: builtin_interfaces/Time\n',1)[1].split('================================================================================',1)[0]
                        separator='\n'+'='*80+'\n'
                        experiment=(top+separator+'MSG: builtin_interfaces/Time\n'+time.strip()+
                                    separator+'MSG: dex_collection_interfaces/StreamStatus\n'+nested+'\n').encode()
                        try:
                            corrected=Schema(schema.id,experiment,'ros2msg',TYPE)
                            decoded=DecoderFactory().decoder_for('cdr',corrected)(msg.data)
                            dynamic,nonfinite=message_fields(decoded)
                            embedded_field_diff['secondary_decoder_first_record']=(
                                'MATCH' if dynamic==val and not nonfinite else 'DIFFERENT')
                        except Exception as exc:
                            embedded_field_diff['secondary_decoder_first_record']=f'{type(exc).__name__}: {exc}'
                        embedded_field_diff['secondary_definition_sha256']=sha256(experiment).hexdigest()
                    jsonrow(recovered,result)
                    local[result['historical_outcome']]+=1
                    outcomes[result['historical_outcome']]+=1
            if seen!=indices.keys():
                raise ValueError(f'Failed-index list incomplete: {part_id}')
            group_rows.append({'recording_id':fixture['recording_id'],'source_status':fixture['a_status'],
                               'source_origin':fixture['a_origin'],'path':str(path),'part_id':part_id,
                               'asset_sha256':part['sha256'],'sample_index_min':min(seen),'sample_index_max':max(seen),
                               'records':len(seen),'outcomes':dict(local)})
    summary={'old_gate_B1':'FAIL','original_failed_events':sum(len(g) for g in wanted.values()),
             'auxiliary_review_outcomes':dict(outcomes),'affected_recordings':group_rows,
             'historical_overlay':hist,'current_overlay':current,'historical_typesupport':compiled,
             'embedded_schema_comparison':embedded_field_diff,
             'git_provenance':git_witness((args.teleop_repo,args.collector_repo),hist),
             'scope':'Re-decode original failures only. Provenance and qualification remain unchanged.',
             'files':{f.name:digest(f) for f in sorted(target.glob('*.jsonl'))}}
    write(target/'collection_status_review.json',summary)
    print(json.dumps({'historical_auxiliary_recheck':dict(outcomes),'original_B1':'FAIL'}))


def uncomment(value):
    return [line.split('#',1)[0].strip() for line in value.splitlines() if line.split('#',1)[0].strip()]


if __name__=='__main__':
    main()
