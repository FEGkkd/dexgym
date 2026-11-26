"""Bounded P01-C diagnostics and serialization, separate from the online core."""
from collections import Counter,defaultdict
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import calendar
import ctypes
import datetime
import json
import numpy as np

from .clock_domains import Anchor,DOMAIN_FIELDS,PrefixClock
from .geometry import hand_geometry,pose,rotation,summary
from .registry_gate import digest
from .sample_identity import Identity,IdentityHistory
from .synchronize import CausalSynchronizer,Event,OfflineAlignment,REASONS,STREAMS,batch_causal


SDK_SHA='b4e98f04144585cf51428c41e6af202dc664a599fb70a61adbcfe6ff0958825b'


class SDKTimestamp(ctypes.Structure):
    _fields_=[('time',ctypes.c_uint64)]


class SDKComponents(ctypes.Structure):
    _fields_=[('fraction',ctypes.c_uint16),('second',ctypes.c_uint8),('minute',ctypes.c_uint8),
              ('hour',ctypes.c_uint8),('day',ctypes.c_uint8),('month',ctypes.c_uint8),
              ('year',ctypes.c_uint32),('timecode',ctypes.c_bool)]


class CalendarDecoder:
    def __init__(self,library):
        if digest(Path(library))!=SDK_SHA:
            raise ValueError('Unexpected MANUS SDK binary')
        self.sdk=ctypes.CDLL(str(library))
        self.decode=self.sdk.CoreSdk_GetTimestampInfo
        self.decode.argtypes=(SDKTimestamp,ctypes.POINTER(SDKComponents))
        self.decode.restype=ctypes.c_int

    def components(self,stamp):
        if type(stamp) is not int or not stamp:
            return None,None
        out=SDKComponents()
        if self.decode(SDKTimestamp(stamp),ctypes.byref(out))!=0:
            return None,None
        fields={name:getattr(out,name) for name,_ in out._fields_}
        if out.timecode:
            return fields,None
        try:
            d=datetime.datetime(out.year,out.month,out.day,out.hour,out.minute,out.second,out.fraction*1000)
        except ValueError:
            return fields,None
        return fields,calendar.timegm(d.timetuple())*1000+out.fraction


def scalar(row,key):
    return row['clock'][key]['value']


def jsonline(f,x):
    f.write(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')


def eventize(rows,offset,params,cal_version,calendar_decoder):
    prepared=[]
    topology={}
    invalid=[]
    geom=defaultdict(lambda:{'bone':[],'norm_error':[]})
    per_edge=defaultdict(list)
    for row in rows:
        stream,idx=row['stream'],row['index']
        mono=scalar(row,'collector_monotonic')-offset
        pub=scalar(row,'ros_header')
        if pub is None:
            pub=-1
        record=dict(idx,stream=stream,collector_mono_ns=scalar(row,'collector_monotonic'))
        token=f'{idx["part_id"]}:{idx["physical_message"]}'
        dataset,pose_arr,valid,topo_version=row['fields'],None,not bool(row['nonfinite']),None
        sdk_source=scalar(row,'sdk_source')
        sdk_receive=scalar(row,'sdk_receive')
        component,canonical_component=calendar_decoder.components(sdk_source)
        try:
            if 'manus' in row:
                side=stream.split('_')[1]
                natural=row['manus']
                z,palm,lengths,edges=hand_geometry(natural['nodes'],side,natural['raw_side'])
                if natural['issues']:
                    raise ValueError(str(natural['issues']))
                topo_version=natural['topology_sha256']
                topology.setdefault((stream,topo_version),dict(
                    stream=stream,version=topo_version,frame_id=natural['frame_id'],
                    joints=[{k:n[k] for k in ('node_id','parent_node_id','original_order','chain_type','joint_type')}
                            for n in natural['nodes']],edges=edges))
                pose_arr=palm
                geom[stream]['bone'].extend(lengths.tolist())
                geom[stream]['norm_error'].extend(np.abs(np.linalg.norm(z[:,3:],axis=1)-1).tolist())
                for (i,j),length in zip(edges,lengths):
                    ni,nj=(natural['nodes'][k]['node_id'] for k in (i,j))
                    per_edge[(stream,topo_version,ni,nj)].append(length)
                fingerprint=sha256(z.tobytes()+topo_version.encode()).hexdigest()
                if canonical_component is None or sdk_receive is None:
                    valid=False
            elif stream!='tracker_safety':
                pose_arr=pose(dataset['pose'])[None,:]
                side='left_wrist' if 'left' in stream else 'right_wrist'
                if not stream.endswith('_raw'):
                    valid=valid and dataset.get('valid') is True and dataset.get('role')==side
                fingerprint=sha256(pose_arr.tobytes()).hexdigest()
            else:
                fingerprint=idx['cdr_sha256']
        except (KeyError,ValueError,TypeError) as e:
            invalid.append(dict(index=record,reason='GEOMETRY_INVALID',detail=str(e)))
            valid=False
            fingerprint=idx['cdr_sha256']
        if not valid and not any(x['index']==record for x in invalid[-1:]):
            invalid.append(dict(index=record,reason='SOURCE_INVALID'))
        ident=Identity(idx['source_instance'],idx['writer_sequence'],
                       sdk_source if sdk_source is not None else pub,
                       sdk_receive if sdk_receive is not None else scalar(row,'collector_monotonic'),
                       fingerprint,canonical_component)
        # Deliberately diagnostic: no physical capture or controller-bound timestamp is inferred.
        modeled=mono+params['arrival_delay_ns']
        reference=dict(record,raw_index_token=token)
        e=Event(stream,ident,reference,modeled,(mono,mono),(modeled,modeled),'SIMULATED',
                pose_arr,valid,dataset.get('safe') if stream=='tracker_safety' else None,
                (cal_version,topo_version),authenticated=True,clock_generation='collector-monotonic-v1')
        raw_pose = z if ('manus' in row and pose_arr is not None) else pose_arr
        prepared.append((e,dict(row,scalar_identity={'instance':ident.instance,'writer_sequence':ident.sequence,
                               'sdk_timestamp_raw':sdk_source,'sdk_receive_raw':sdk_receive,
                               'calendar':component,'ros_header':pub},
                               raw_pose=raw_pose,processed_pose=pose_arr)))
    geom={k:{metric:summary(values) for metric,values in v.items()} for k,v in geom.items()}
    for (stream,version,node,parent),values in sorted(per_edge.items()):
        x=np.asarray(values)
        center=float(np.median(x))
        absdev=np.abs(x-center)
        geom[stream].setdefault('edges',[]).append(
            dict(topology=version,node=node,parent=parent,length_m=summary(x),absolute_deviation_m=summary(absdev),
                 samples_outside_1mm=int(sum(absdev>.001)),samples_outside_2mm=int(sum(absdev>.002))))
    return prepared,list(topology.values()),invalid,geom


def stream_statistics(prepared):
    groups=defaultdict(list)
    for event,row in prepared:
        groups[event.stream].append((event,row))
    reports={}
    for k,group in groups.items():
        group.sort(key=lambda p:p[0].known_ns)
        counts=Counter()
        h=IdentityHistory()
        for e,r in group:
            counts[h.observe(e.identity)]+=1
        fields={}
        for scope in DOMAIN_FIELDS:
            vals=[r['clock'][scope]['value'] for e,r in group]
            if scope=='sdk_source':
                fields[scope]={'provided':sum(x is not None for x in vals),
                               'missing':sum(x is None for x in vals),
                               'equal_neighbors':sum(a is not None and a==b for a,b in zip(vals,vals[1:])),
                               'packed_no_interval_arithmetic':True}
                continue
            deltas=[b-a for a,b in zip(vals,vals[1:]) if a is not None and b is not None]
            fields[scope]={'provided':sum(x is not None for x in vals),'missing':sum(x is None for x in vals),
                            'backwards':sum(t<0 for t in deltas),'stable_step':summary([t for t in deltas if t>0]),
                            'equal_neighbors':sum(t==0 for t in deltas)}
        component=[e.identity.component_ms for e,_ in group]
        diffs=[b-a for a,b in zip(component,component[1:]) if a is not None and b is not None]
        fields['sdk_calendar_components']={'available':sum(x is not None for x in component),
             'backwards':sum(x<0 for x in diffs),'step_ms':summary([x for x in diffs if x>0]),
             'scope':'Decoded parts. Provides no physical cross-device bound.'}
        n=len(group)
        sec=(group[-1][0].capture[1]-group[0][0].capture[1])*1e-9 if n else 0
        unique=counts['NEW']+counts['SOURCE_RESTART']
        reports[k]=dict(messages=n,duration_s=sec,
                        collector_event_rate_hz=(n-1)/sec if sec>0 else None,
                        independent_source_rate_candidate_hz=(unique-1)/sec if sec>0 else None,
                        sdk_duplicate_fraction=counts['DUPLICATE_SOURCE']/n if n and k.startswith('manus') else None,
                        clock_fields=fields,identity_events=dict(counts))
    return reports


def audit_freshness(prepared):
    """A window cannot establish MANUS freshness from its first cache value."""
    previous={}
    result=[]
    for event,row in sorted(prepared,key=lambda x:(x[0].known_ns,x[0].identity.sequence,x[0].stream)):
        if event.stream.startswith('manus_'):
            identity=event.identity
            prior=previous.get(event.stream)
            consecutive=(prior is not None and prior.instance==identity.instance and
                         identity.sequence==prior.sequence+1)
            fresh=(consecutive and (identity.source_stamp,identity.receive_stamp)!=
                   (prior.source_stamp,prior.receive_stamp))
            result.append((replace(event,fresh=fresh),row))
            previous[event.stream]=identity
        else:
            result.append((event,row))
    return result


def json_result(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,dict):
        return {k:json_result(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_result(v) for v in value]
    return value


def compare_prefix(events,ticks,params):
    baseline=batch_causal(events,ticks,params,'SIMULATED')
    k=len(ticks)//2
    allowed=ticks[k][1]
    initial=events[:]
    future=lambda e: e.known_ns>allowed
    removed=[e for e in initial if not future(e)]
    rng=np.random.default_rng(20261126)
    changed=[]
    for e in initial:
        if not future(e):
            changed.append(e)
            continue
        shift=int(rng.integers(1,1_000_001))
        changed.append(replace(e,pose=e.pose+rng.normal(size=e.pose.shape) if e.pose is not None else None,
                               valid=not e.valid,safe=not e.safe if e.safe is not None else True,
                               capture=tuple(v-shift for v in e.capture),
                               arrival=tuple(v+shift for v in e.arrival)))
    prefix=json_result(baseline[:k+1])
    no_future=json_result(batch_causal(removed,ticks[:k+1],params,'SIMULATED'))
    changed_future=json_result(batch_causal(changed,ticks[:k+1],params,'SIMULATED'))
    streamed=CausalSynchronizer(params,'SIMULATED')
    sequence=iter(sorted(events,key=lambda e:e.known_ns))
    pending=next(sequence,None)
    online=[]
    for t,c in ticks:
        while pending is not None and pending.known_ns<=c:
            streamed.offer(pending)
            pending=next(sequence,None)
        online.append(streamed.step(t,c))
    oracle=PrefixClock(evidence='synthetic anchor QA only')
    oracle.add(Anchor(2,80,90,0,'reference fixture'))
    earlier=oracle.bounds(10,20)
    oracle.add(Anchor(10,100,105,21,'future unmodified fixture'))
    unmodified=oracle.bounds(10,20)
    oracle.anchors.pop()
    oracle.add(Anchor(10,1000,1010,21,'future mutation fixture'))
    ortho=earlier==unmodified==oracle.bounds(10,20)
    result=dict(past_output_rows=len(prefix),future_event_vulnerability_controls=len(initial)-len(removed),
                future_deleted_equal=prefix==no_future,future_pose_clock_valid_safety_changed_equal=prefix==changed_future,
                streaming_equals_batch=json_result(baseline)==json_result(online),
                future_anchor_equal=ortho)
    return baseline,result


def export_alignment(directory,stem,results,causal,special_events):
    times=np.array([d['tick_ns'] for d in results],dtype=np.int64)
    n=max((len(e.pose) for e in special_events if e.pose is not None),default=1)
    shape=(len(times),len(STREAMS),n)
    poses=np.full((*shape,7),np.nan)
    velocities=np.full((*shape,3),np.nan)
    node_mask=np.zeros(shape,dtype=bool)
    validity=np.zeros(shape[:2],dtype=bool)
    reasons=np.full(shape[:2],REASONS.index('DROPOUT'),dtype=np.uint8)
    age=np.full(shape[:2],-1,dtype=np.int64)
    for i,record in enumerate(results):
        for j,stream in enumerate(STREAMS):
            row=record['streams'][stream]
            reasons[i,j]=REASONS.index(row['reason'])
            validity[i,j]=row['valid']
            if row['pose'] is not None:
                z=len(row['pose'])
                poses[i,j,:z]=row['pose']
                node_mask[i,j,:z]=True
            if row['velocity'] is not None:
                velocities[i,j,:len(row['velocity'])]=row['velocity']
            if causal and row['age_ns'] is not None:
                age[i,j]=row['age_ns']
    path=directory/f'{stem}.npz'
    np.savez_compressed(path,tick_ns=times,pose_m_xyzw=poses,velocity_m_s=velocities,mask=validity,node_mask=node_mask,
                        reason=reasons,age_ns=age,joint_valid=np.array([r['joint_valid'] for r in results]))
    return {'file':path.name,'sha256':digest(path),'pose_shape':list(poses.shape),'clock_quality':
            'SIMULATED' if causal else 'OFFLINE_PROXY','tick_count':len(times),'streams':list(STREAMS)}


def export_raw(directory,stem,prepared,trace_stream,recording,window):
    v=[(e,o) for e,o in prepared if window in o['windows']]
    n=max((len(r['raw_pose']) for e,r in v if r['raw_pose'] is not None),default=1)
    raw=np.full((len(v),n,7),np.nan)
    times=np.full((len(v),len(DOMAIN_FIELDS)),-1,dtype=np.int64)
    for i,(e,r) in enumerate(v):
        if r['raw_pose'] is not None:
            raw[i,:len(r['raw_pose'])]=r['raw_pose']
        for k,name in enumerate(DOMAIN_FIELDS):
            value=r['clock'][name]['value']
            if value is not None:
                times[i,k]=value
        jsonline(trace_stream,dict(kind='raw',recording_id=recording,window=window,row=i,stream=e.stream,
                                   index=e.ref,identity=r['scalar_identity'],frame=(r['fields'].get('header') or {}).get('frame_id'),
                                   tracker_valid=r['fields'].get('valid'),safety=r['fields'].get('safe'),version=e.version))
    target=directory/f'{stem}-raw.npz'
    np.savez_compressed(target,pose_m_xyzw=raw,time_values=times,time_available=times!=-1)
    return {'file':target.name,'sha256':digest(target),'raw_shape':list(raw.shape),'clock_columns':list(DOMAIN_FIELDS)}


def measure(causal,offline):
    ages,skews,dist,angular,velocity,static,dynamic,proxy_v=[],[],[],[],[],[],[],[]
    ccover,ocover=Counter(),Counter()
    by_stream=defaultdict(lambda:defaultdict(list))
    by_pair=defaultdict(list)
    for a,b in zip(causal,offline):
        skews.extend(x for x in a['skew_ns'].values() if x is not None)
        for pair,x in a['skew_ns'].items():
            if x is not None:
                by_pair[pair].append(x*1e-6)
        for s,ar in a['streams'].items():
            br=b['streams'][s]
            ccover[s]+=int(ar['valid']); ocover[s]+=int(br['valid'])
            if ar['valid']:
                ages.append(ar['age_ns']*1e-6)
                by_stream[s]['age_ms'].append(ar['age_ns']*1e-6)
            if not (ar['valid'] and br['valid']) or ar['pose'].shape!=br['pose'].shape:
                continue
            residual=np.linalg.norm(ar['pose'][:,:3]-br['pose'][:,:3],axis=1)*1000
            dist.extend(residual.tolist())
            angle=(rotation(ar['pose'][:,3:]).inv()*rotation(br['pose'][:,3:])).magnitude()
            angular.extend(angle.tolist())
            by_stream[s]['position_delta_mm'].extend(residual.tolist())
            by_stream[s]['rotation_delta_rad'].extend(angle.tolist())
            av,bv=ar['velocity'],br['velocity']
            if av is not None and bv is not None:
                speed=np.linalg.norm(bv,axis=1)
                vd=np.linalg.norm(av-bv,axis=1)
                velocity.extend(vd.tolist())
                by_stream[s]['velocity_delta_m_s'].extend(vd.tolist())
                proxy_v.extend(speed.tolist())
                static.extend(residual[speed<=0.02].tolist())
                dynamic.extend(residual[speed>0.02].tolist())
    vel=summary(proxy_v)
    return {'tick_count':len(causal),'causal_ticks_valid_count':dict(ccover),'offline_ticks_valid_count':dict(ocover),
            'causal_joint_valid_count':sum(x['joint_valid'] for x in causal),
            'offline_joint_valid_count':sum(x['joint_valid'] for x in offline),
            'proxy_age_ms':summary(ages),'proxy_skew_ms':summary(np.asarray(skews)*1e-6),
            'by_stream':{s:{k:summary(v) for k,v in vals.items()} for s,vals in by_stream.items()},
            'proxy_skew_by_role_ms':{k:summary(v) for k,v in by_pair.items()},
            'position_delta_mm':summary(dist),'angular_delta_rad':summary(angular),'velocity_delta_m_s':summary(velocity),
            'static_delta_mm_at_candidate_0_02m_s':summary(static),'dynamic_delta_mm_at_candidate_0_02m_s':summary(dynamic),
            'joint_valid_seconds':{'causal': sum(x['joint_valid'] for x in causal)*step_seconds(causal),
                                   'offline': sum(x['joint_valid'] for x in offline)*step_seconds(causal)},
            'proxy_speed_m_s':vel,'v95_times_hypothetical_ms_delta_mm':{
                str(dt):vel['p95']*dt if vel['p95'] is not None else None for dt in (2,5,10,20)}}


def step_seconds(series):
    return (series[1]['tick_ns']-series[0]['tick_ns'])*1e-9 if len(series)>1 else 0.
