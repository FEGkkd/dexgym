"""Versioned HMB window serializer shared by P01-D pilots and the gated builder."""
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from .geometry import interpolate
from .hmb_validation import validate_semantics
from .registry_gate import digest
from .synchronize import CORE_STREAMS,REASONS,SAFETY,STREAMS


SCHEMA='s05.hmb.simulation.v1'
USE='HISTORICAL_SIMULATION_DEVELOPMENT_ONLY'
QUALITY=('UNUSABLE','SIMULATED','PROXY','OFFLINE_PROXY')
REASONS_V1=(*REASONS,'VELOCITY_HISTORY','VELOCITY_TIME_UNKNOWN','RECOVERY_FRESHNESS')
REASONS_V1=tuple(dict.fromkeys(REASONS_V1))
BASE='eb00f254dc71ac7f6ceef9e2c76a661ac3e912bd'
SYNC_HASH='13550058a3114e062c6f7ba2e92b8149659dc5870ae9a97ec256679b7ed571b5'
SYNC_PATH=Path(__file__).with_name('synchronize.py')


class InjectedPublicationError(RuntimeError):
    pass


COMMON_FIELDS={
    'tick_ns':('int64','T','ns','collector monotonic relative to episode start'),
    'pose':('float64','T,S,N,7','m,XYZW','palm-local for MANUS, recorded frame for raw/registered wrist'),
    'sdk_common_pose':('float64','T,S,N,7','m,XYZW','original SDK common frame; MANUS only'),
    'node_mask':('bool','T,S,N','1','populated geometric nodes'),
    'pose_valid':('bool','T,S','1','independent pose validity'),
    'velocity_m_s':('float64','T,S,N,3','m/s','same frame as pose'),
    'velocity_valid':('bool','T,S','1','independent velocity validity'),
    'velocity_age_ns':('int64','T,S','ns','-1 unknown or offline'),
    'velocity_quality':('uint8','T,S','index','quality enum; 0 for invalid'),
    'safety_permission':('bool','T','1','qualified safety at tick'),
    'safety_source_index':('int64','T','index','-1 if unavailable; see source table'),
    'joint_valid':('bool','T','1','four core streams; raw wrists diagnostic'),
    'clock_quality':('uint8','T,S','index','quality enum; never VERIFIED in this format'),
    'clock_generation':('int32','T,S','index','-1 unknown; dictionary in metadata'),
    'calibration_id':('int32','T,S','index','-1 unknown; dictionary in metadata'),
    'topology_id':('int32','T,S','index','-1 unknown; dictionary in metadata'),
    'reason_code':('uint8','T,S','index','pose reason registry'),
    'velocity_reason_code':('uint8','T,S','index','independent velocity reason registry'),
    'source_index':('int64','T,S,2','index','-1 missing; ordered endpoints table'),
    'segment_id':('int32','T','index','-1 unassigned, no stitching across blocks'),
}
MODE_FIELDS={
    'causal': {
        'cutoff_ns':('int64','T','ns','information cutoff'),
        'capture_bounds_ns':('int64','T,S,2','ns','capture lower and upper; simulated'),
        'arrival_bounds_ns':('int64','T,S,2','ns','arrival lower and upper; simulated'),
        'age_ns':('int64','T,S','ns','tick minus capture lower, -1 unknown'),
    },
    'offline': {
        'bracket_ns':('int64','T,S,2','ns','same recorder proxy domain source interval'),
        'bracket_span_ns':('int64','T,S','ns','bracket upper minus lower, -1 missing'),
        'interpolation_weight':('float64','T,S','1','weight of right endpoint'),
        'bracket_safe':('bool','T,S','1','whole bracket has qualified safety'),
    },
}


def schema():
    return dict(schema=SCHEMA,allowed_use=USE,streams=list(STREAMS),core_streams=list(CORE_STREAMS),
                normal_forms={'XYZW':'quaternion','nodes':'preserve physical raw order; N varies by partition'},
                vocabularies={'quality':list(QUALITY),'reason':list(REASONS_V1)},
                arrays={m:COMMON_FIELDS|extra for m,extra in MODE_FIELDS.items()},
                integer_padding=-1,float_padding='NaN',bool_padding=False,
                files=['causal.npz','offline.npz','metadata.json','sources.jsonl','manifest.json'],
                sync_baseline=BASE,sync_source_sha256=SYNC_HASH)


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()


def lock_sync():
    if digest(SYNC_PATH)!=SYNC_HASH:
        raise ValueError('HMB sync source differs from the accepted P01-C final patch')


def _bounds_crossed(align,lo,hi):
    return any(barrier.intersects(lo,hi) for key in (*CORE_STREAMS,SAFETY)
               for barrier in align.barriers[key])


def segment_ranges(rows,alignment,minimum):
    """Keep causal outputs unchanged; use barriers only to partition future dataset access."""
    result=[]
    start=None
    prev_key=None
    for i,row in enumerate(rows):
        usable=row['joint_valid'] and row.get('safety_permission',True)
        streams=row['streams']
        versions=tuple((streams[s]['version'],streams[s].get('clock_generation')) for s in CORE_STREAMS)
        key=(versions,row.get('safety_release_ns'))
        cut=i>0 and _bounds_crossed(alignment,rows[i-1]['tick_ns'],row['tick_ns'])
        if start is not None and (not usable or key!=prev_key or cut):
            if i-start>=minimum:
                result.append((start,i))
            start=None
        if usable and start is None:
            start=i
        prev_key=key
    if start is not None and len(rows)-start>=minimum:
        result.append((start,len(rows)))
    return result


def _metadata_indices(target,value):
    if value is None:
        return -1
    if value not in target:
        target.append(value)
    return target.index(value)


def _reference_table(prepared):
    refs={}
    data=[]
    transforms={}
    ordered=sorted(prepared,key=lambda item:(item[0].stream,item[0].identity.sequence,item[0].known_ns))
    for event,row in ordered:
        token=event.ref['raw_index_token']
        if token in refs:
            raise ValueError('Raw physical token duplicated')
        refs[token]=len(data)
        transforms[token]=row['raw_pose']
        data.append(dict(record=event.ref,identity=row['scalar_identity'],raw_clock=row['clock'],
                         recorded_frame=(row['fields'].get('header') or {}).get('frame_id'),
                         stream=event.stream,capture_bounds_ns=event.capture,arrival_bounds_ns=event.arrival,
                         calibration_id=event.version[0],topology_id=event.version[1],clock_generation=event.clock_generation,
                         source_eligible=event.authenticated,fresh=event.fresh,raw_valid=event.valid,safe=event.safe))
    return refs,data,transforms


def build_arrays(mode,rows,alignment,prepared,segments,maximum_nodes,dictionaries):
    table,data,raw_values=prepared
    dims={'T':len(rows),'S':len(STREAMS),'N':maximum_nodes,'2':2,'3':3,'7':7}
    fields=COMMON_FIELDS|MODE_FIELDS[mode]
    arrays={}
    for name,(typ,expression,_,_) in fields.items():
        shape=tuple(dims[x] for x in expression.split(','))
        fill=np.nan if typ=='float64' else -1 if typ.startswith('int') else 0
        arrays[name]=np.full(shape,fill,dtype=np.dtype(typ).newbyteorder('<'))
    for segment,(lo,hi) in enumerate(segments):
        arrays['segment_id'][lo:hi]=segment
    causal=mode=='causal'
    for i,tick in enumerate(rows):
        t=tick['tick_ns']
        arrays['tick_ns'][i]=t
        arrays['joint_valid'][i]=tick['joint_valid']
        if causal:
            arrays['cutoff_ns'][i]=tick['cutoff_ns']
            arrays['safety_permission'][i]=tick['safety_permission']
            ref=tick['safety_ref']
        else:
            arrays['safety_permission'][i]=alignment._safe(t,t) is None
            ref=None
            for event,_,_ in reversed(alignment.events[SAFETY]):
                if event.capture[1]<=t:
                    ref=event.ref
                    break
        if ref is not None:
            arrays['safety_source_index'][i]=table[ref['raw_index_token']]
        for j,stream in enumerate(STREAMS):
            row=tick['streams'][stream]
            ix=(i,j)
            refs=[row['ref']] if causal and row['ref'] else row.get('refs') or []
            tokens=[r['raw_index_token'] for r in refs]
            arrays['source_index'][i,j,:len(tokens)]=[table[token] for token in tokens]
            clock=row.get('clock_generation') if causal else data[table[tokens[0]]]['clock_generation'] if tokens else None
            arrays['clock_generation'][ix]=_metadata_indices(dictionaries['clock_generation'],clock)
            if row['version'] is not None:
                for tag,val in zip(('calibration_id','topology_id'),row['version'],strict=True):
                    arrays[tag][ix]=_metadata_indices(dictionaries[tag],val)
            arrays['reason_code'][ix]=REASONS_V1.index(row['reason'])
            arrays['pose_valid'][ix]=row['valid']
            arrays['clock_quality'][ix]=QUALITY.index(tick['quality']) if row['valid'] else 0
            if causal:
                for field,key in (('capture_bounds_ns','capture'),('arrival_bounds_ns','arrival'),('age_ns','age_ns')):
                    if row[key] is not None:
                        arrays[field][ix]=row[key]
            if not row['valid']:
                arrays['velocity_reason_code'][ix]=REASONS_V1.index(row['reason'])
                continue
            pose=row['pose']
            n=len(pose)
            arrays['node_mask'][i,j,:n]=True
            arrays['pose'][i,j,:n]=pose
            if not causal:
                lo,hi=row['time_interval']
                arrays['bracket_ns'][ix]=(lo,hi)
                arrays['bracket_span_ns'][ix]=hi-lo
                arrays['bracket_safe'][ix]=True
                arrays['interpolation_weight'][ix]=row['weight']
            if stream.startswith('manus_'):
                full=[raw_values[token] for token in tokens]
                arrays['sdk_common_pose'][i,j,:n]=full[0] if causal or tokens[0]==tokens[-1] else (
                    interpolate(full[0],full[1],row['weight']))
            if row['velocity'] is not None:
                arrays['velocity_m_s'][i,j,:n]=row['velocity']
                arrays['velocity_valid'][ix]=True
                arrays['velocity_quality'][ix]=QUALITY.index(row['velocity_quality'] if causal else 'OFFLINE_PROXY')
                if causal:
                    arrays['velocity_age_ns'][ix]=row['velocity_age_ns']
            arrays['velocity_reason_code'][ix]=REASONS_V1.index(
                row['velocity_reason'] if causal else 'VALID' if row['velocity'] is not None else 'VELOCITY_HISTORY')
    return arrays


def _write_fsynced(path,data):
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inspect_partition(path):
    manifest=json.loads((path/'manifest.json').read_bytes())
    meta=json.loads((path/'metadata.json').read_bytes())
    expected={'causal.npz','offline.npz','metadata.json','sources.jsonl'}
    if (manifest['schema']!=SCHEMA or set(manifest['files'])!=expected or meta['allowed_use']!=USE or
            meta['sync_source_sha256']!=SYNC_HASH or meta['sync_baseline']!=BASE or
            sha256(json_bytes(meta['identity'])).hexdigest()!=manifest['identity_sha256'] or
            any(digest(path/name)!=value for name,value in manifest['files'].items())):
        raise ValueError('Unrecognized or modified HMB partition')
    return manifest


def read_partition(path,mode):
    """Load only the explicit production schema, with its raw-record contract."""
    inspect_partition(path)
    meta=json.loads((path/'metadata.json').read_bytes())
    t,s,n=meta['ticks'],len(STREAMS),meta['N']
    dims={'T':t,'S':s,'N':n,'2':2,'3':3,'7':7}
    fields=COMMON_FIELDS|MODE_FIELDS[mode]
    with np.load(path/f'{mode}.npz',allow_pickle=False) as payload:
        if set(payload.files)!=set(fields):
            raise ValueError('Unexpected HMB fields')
        data={key:payload[key] for key in fields}
    sources=[json.loads(line) for line in (path/'sources.jsonl').read_bytes().splitlines()]
    for key,(dtype,shape,_,_) in fields.items():
        if (data[key].dtype!=np.dtype(dtype).newbyteorder('<') or
                data[key].shape!=tuple(dims[x] for x in shape.split(','))):
            raise ValueError(f'Wrong HMB dtype/shape: {key}')
    pose,vel=data['pose_valid'],data['velocity_valid']
    pointers=data['source_index']
    if (np.any(vel&~pose) or np.any(pose&~data['safety_permission'][:,None]) or
            np.any(data['joint_valid']&~pose[:,:4].all(axis=1)) or
            np.any(pointers<-1) or np.any(pointers>=len(sources)) or
            np.any(pointers[...,0][pose]<0) or
            any(not sources[i]['source_eligible'] for i in set(pointers[pose].flatten()) if i>=0) or
            np.any(data['clock_quality']>=len(QUALITY)) or np.any(data['velocity_quality']>=len(QUALITY)) or
            np.any(data['clock_quality'][pose]==0) or np.any((data['velocity_quality']!=0)!=vel) or
            np.any(data['reason_code']>=len(REASONS_V1)) or np.any(data['velocity_reason_code']>=len(REASONS_V1))):
        raise ValueError('Unsafe or inconsistent HMB mask, provenance, or time quality')
    if mode=='causal':
        if (np.any(data['capture_bounds_ns'][...,1][pose]>np.broadcast_to(data['tick_ns'][:,None],pose.shape)[pose]) or
                np.any(data['arrival_bounds_ns'][...,1][pose]>np.broadcast_to(data['cutoff_ns'][:,None],pose.shape)[pose]) or
                np.any(pointers[...,1]>=0) or np.any(data['velocity_age_ns'][vel]<0)):
            raise ValueError('Causal partition violates source availability')
    elif (np.any(pointers[...,1][pose]<0) or np.any(data['bracket_span_ns'][pose]<0) or
          np.any(~data['bracket_safe'][pose]) or not np.isfinite(data['interpolation_weight'][pose]).all() or
          np.any(data['interpolation_weight'][pose]<0) or
          np.any(data['interpolation_weight'][pose]>1)):
        raise ValueError('Offline partition lacks safe auditable brackets')
    validate_semantics(data,meta,sources,mode,QUALITY,REASONS_V1)
    return data,meta,sources


def validate_partition_pair(path):
    causal,_,_=read_partition(path,'causal')
    offline,_,_=read_partition(path,'offline')
    if not np.array_equal(causal['tick_ns'],offline['tick_ns']):
        raise ValueError('Causal and offline grids disagree')
    return inspect_partition(path)


def write_partition(destination,*,identity,prepared,topologies,causal,offline,alignment,min_segment,
                    fail_before_publish=False,max_partition_bytes=None):
    lock_sync()
    if destination.parent.is_symlink() or destination.is_symlink():
        raise ValueError('HMB partition destination must not be a symlink')
    destination.parent.mkdir(parents=True,exist_ok=True)
    identity_hash=sha256(json_bytes(identity)).hexdigest()
    with (destination.parent/f'.{destination.name}.lock').open('ab') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if destination.exists():
            manifest=validate_partition_pair(destination)
            if manifest['identity_sha256']!=identity_hash:
                raise ValueError('Existing partition has a different identity')
            return manifest,0
        stage=destination.parent/f'.{destination.name}.{uuid4().hex}.partial'
        stage.mkdir(mode=0o700)
        references,sources,vals=_reference_table(prepared)
        n=max((len(x) for x in vals.values() if x is not None),default=1)
        dictionaries={key:[] for key in ('clock_generation','calibration_id','topology_id')}
        segment_index={}
        for mode,rows in (('causal',causal),('offline',offline)):
            segments=segment_ranges(rows,alignment,min_segment)
            segment_index[mode]=segments
            arrays=build_arrays(mode,rows,alignment,(references,sources,vals),segments,n,dictionaries)
            with (stage/f'{mode}.npz').open('xb') as f:
                np.savez_compressed(f,**arrays)
                f.flush()
                os.fsync(f.fileno())
        _write_fsynced(stage/'sources.jsonl',b''.join(json_bytes(row) for row in sources))
        cuts=[i for i in range(1,len(causal))
              if _bounds_crossed(alignment,causal[i-1]['tick_ns'],causal[i]['tick_ns'])]
        _write_fsynced(stage/'metadata.json',json_bytes(dict(schema=SCHEMA,allowed_use=USE,
            sync_source_sha256=SYNC_HASH,sync_baseline=BASE,identity=identity,topologies=topologies,
            dictionary=dictionaries,segments=segment_index,ticks=len(causal),N=n,
            diagnostic_non_actor={'boundary_cut_indices':cuts})))
        manifest=dict(schema=SCHEMA,identity_sha256=identity_hash,files={file.name:digest(file) for file in stage.iterdir()})
        _write_fsynced(stage/'manifest.json',json_bytes(manifest))
        footprint=sum(file.stat().st_size for file in stage.iterdir())
        if max_partition_bytes is not None and footprint>max_partition_bytes:
            raise ValueError('Staged shard exceeds the approved partition budget')
        validate_partition_pair(stage)
        _fsync_dir(stage)
        if fail_before_publish:
            raise InjectedPublicationError('Injected stop after fsync before atomic publication')
        stage.rename(destination)
        _fsync_dir(destination.parent)
        return manifest,footprint
