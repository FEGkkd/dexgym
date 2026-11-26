"""Bounded qualification, stratified development split, and independent QC."""
from collections import Counter,defaultdict
from dataclasses import dataclass,replace
from hashlib import sha256
from math import ceil,log10
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from .calibration import extrinsics,verify_pairs
from .capture_windows import clock_observations,scan_stream,verify_source
from .clock_domains import proxy_grid
from .geometry import summary
from .hmb_build import segment_ranges
from .hmb_gate import dependency_index,validates_shape
from .review_metrics import spatial_metrics
from .sync_evidence import audit_freshness,compare_prefix,eventize,stream_statistics
from .synchronize import CORE_STREAMS,OfflineAlignment,STREAMS,batch_causal


@dataclass
class Window:
    recording_id: str
    label: str
    raw_interval: tuple
    ticks: list
    prepared: list
    topologies: list
    clock: dict

    def bind_quaternion_qc(self,tolerance):
        rejected=0
        revised=[]
        for event,row in self.prepared:
            raw=row['raw_pose']
            if raw is not None and np.max(np.abs(np.linalg.norm(raw[:,3:],axis=1)-1))>tolerance:
                event=replace(event,valid=False)
                rejected+=1
            revised.append((event,row))
        self.prepared=revised
        return rejected

    def replay(self,params,controls=False):
        events=[e for e,_ in self.prepared]
        if controls:
            causal,prefix=compare_prefix(events,self.ticks,params)
        else:
            causal=batch_causal(events,self.ticks,params,'SIMULATED')
            prefix=None
        offline=OfflineAlignment(events,params)
        return causal,[offline.step(t) for t,_ in self.ticks],offline,prefix


def planned_windows(record,plan):
    start,stop=record['recorded_interval_ns']
    width,offset=plan['window_ns'],plan['offset_ns']
    if stop-start<plan['min_capture_duration_ns']:
        raise ValueError('Recording too short for preregistered probe')
    if stop-start<3*width+2*offset:
        return {'middle':((start+stop-width)//2,(start+stop-width)//2+width)}
    anchors={'head':start+offset,'middle':(start+stop-width)//2,'tail':stop-width-offset}
    return {k:(v,v+width) for k,v in anchors.items()}


def read_recording(record,assets,matrix,writer,sdk,plan,windows,*,production=False):
    started=perf_counter()
    metadata,parts,provenance=verify_source(record,assets,None,matrix,writer,probe_source=True)
    elapsed=perf_counter()-started
    limits=plan['budgets']
    groups={}
    scan={}
    audit=dependency_index(matrix)
    for stream,shards in sorted(parts.items()):
        m={}
        groups[stream],_=scan_stream(stream,shards,windows,plan['context_ns'],limits['physical_per_part'],measurement=m)
        scan[stream]=m
        if not groups[stream]:
            raise ValueError(f'{stream} has no required bounded proof')
        rule=audit[groups[stream][0]['index']['topic']]
        invalid=sum(not validates_shape(row,rule['role']) or bool(row['nonfinite']) for row in groups[stream])
        m.update(source_records=len(groups[stream]),invalid_schema_rows=invalid,
                 schema_sha256=rule['schema_sha256'],
                 writer_source_ids=sorted({row['index']['source_instance'] for row in groups[stream]}),
                 asset_parts=[{'id':a['id'],'sha256':a['file_sha256'],'bytes':a['size_bytes']} for a,_,_ in shards],
                 mcap_index_read_seconds=max(m['wall_seconds']-m['cdr_and_extract_seconds'],1e-9))
    refs=provenance['configs']['tracker']['refs']
    config=next(r for r in refs if r['original_path'].endswith('/tracker_calibration.yaml'))
    target_file=next(r for r in refs if r['original_path'].endswith('/transformer.yaml'))
    target=yaml.safe_load(Path(target_file['path']).read_bytes())['tracker_pose_transformer']['ros__parameters']['target_frame']
    ext=extrinsics(Path(config['path']))
    pair=[]
    for side in ('left','right'):
        main='tracker_'+side
        result=verify_pairs(groups[main+'_raw'],groups[main],ext[side]['transform'],target)
        pair.append(dict(side=side,**result))
    sources=[r for rows in groups.values() for r in rows]
    origin=metadata['start_monotonic_ns']
    observed,topologies,invalid,geom=eventize(sources,origin,plan['diag'],config['sha256'],sdk)
    selected=[]
    for name,interval in windows.items():
        anchors=clock_observations(sources,name)
        if production:
            anchors.append((metadata['start_realtime_ns'],origin,{'manifest_sha256':provenance['manifest_sha256']}))
        ticks,clock=proxy_grid(*interval,anchors,origin,plan['tick_ns'])
        context=audit_freshness([(e,r) for e,r in observed if name in r['windows']])
        selected.append(Window(record['id'],name,interval,ticks,context,topologies,clock))
    pair_ok=lambda a: (a['matched']>=plan['min_registered_pairs_per_side'] and
        a['translation_m']['max']<=plan['calibration_error_limits']['m'] and
        a['rotation_rad']['max']<=plan['calibration_error_limits']['rad'] and
        all(r['reason']=='REGISTERED_INVALID' for r in a['failures']))
    issues=[]
    for stream,m in scan.items():
        if m['invalid_schema_rows']:
            issues.append(f"{stream}: {m['invalid_schema_rows']} invalid schema/required-field rows")
    if not all(pair_ok(a) for a in pair):
        issues.append('Missing minimum independent software registration evidence')
    if any(w.clock['max_abs_recorder_residual_ns']>plan['archived_clock_pair_limit_ns'] for w in selected):
        issues.append('Host recorder pair consistency outside audited 1 ms cap')
    if any(r['reason']=='GEOMETRY_INVALID' for r in invalid):
        issues.append('Invalid raw geometry topology or units')
    if not topologies or set(t['stream'] for t in topologies)!={'manus_left','manus_right'}:
        issues.append('Both observed MANUS topologies required')
    evidence=dict(**provenance,status='BLOCKED' if issues else 'QUALIFIED',issues=issues,episode_id=metadata['episode_id'],
                  capture_qualification='SIMULATED',calibration_sha256=config['sha256'],target_frame=target,
                  calibrated_side_pairs=pair,geometry=geom,topologies=topologies,
                  assets={k:v['asset_parts'] for k,v in scan.items()},source_read=scan,
                  source_event_count=len(sources),manifest_check_seconds=elapsed,
                  source_calibration_scope='recorded software registration only',invalid_record_count=len(invalid),
                  invalid_record_examples=invalid[:12],
                  run_clocks=[dict(window=w.label,**w.clock) for w in selected])
    return selected,evidence


def development_split(qualified,usage,seed):
    by_parent={r['root_episode_id']:r for r in qualified}
    if len(by_parent)!=len(qualified) or len(by_parent)<2:
        raise ValueError('Need at least two unique, qualified development parent roots')
    labels=lambda row: (('task',row['task']),('date',row['capture_date']),
                       ('config',row['configs']['teleop']['revision']))
    left=Counter(l for row in qualified for l in labels(row))
    seen=Counter()
    available=set(by_parent)
    chosen=[]
    rank=lambda key: sha256(f'{seed}:{key}'.encode()).hexdigest()
    for _ in range(max(1,(len(available)+2)//4)):
        choices=[k for k in available if all(left[v]>1 for v in labels(by_parent[k]))]
        if not choices:
            if chosen:
                break
            choices=sorted(available)
        key=min(choices,key=lambda k:(-sum(not seen[v] for v in labels(by_parent[k])),rank(k)))
        available.remove(key)
        chosen.append(key)
        for l in labels(by_parent[key]):
            left[l]-=1
            seen[l]+=1
    def row(key):
        source=by_parent[key]
        return dict(root_episode_id=key,recording_id=source['recording_id'],task=source['task'],
                    capture_date=source['capture_date'],teleop_revision=source['configs']['teleop']['revision'],
                    calibration_sha256=source['calibration_sha256'],
                    historical_roles=sorted(usage.get(key) or {'USAGE_UNKNOWN'}),
                    session_identity='UNKNOWN',sample_rank=rank(key))
    return dict(schema='s05.hmb.development_split.v1',seed=seed,Train=[row(k) for k in sorted(available)],
                Validation=[row(k) for k in sorted(chosen)],Final_Test={'status':'BLOCKED','roots':[]},
                rule='SHA order after max new task/date/config breadth, preserve Train labels when possible; inherit root for derivatives')


def motion_labels(offline,threshold,stop_dwell):
    def moving(tick,stream):
        row=tick['streams'][stream]
        v=row['velocity']
        return row['valid'] and v is not None and np.max(np.linalg.norm(v,axis=1))>threshold
    motion=np.array([[moving(tick,s) for s in CORE_STREAMS] for tick in offline],dtype=bool)
    valid=np.array([all(t['streams'][s]['velocity'] is not None for s in CORE_STREAMS) for t in offline],dtype=bool)
    any_motion=motion.any(axis=1)
    stops=sum(np.all(any_motion[j-stop_dwell:j]&valid[j-stop_dwell:j]) and
              np.all(~any_motion[j:j+stop_dwell]&valid[j:j+stop_dwell])
              for j in range(stop_dwell,len(offline)-stop_dwell+1))
    return any_motion,dict(per_core_moving=dict(zip(CORE_STREAMS,motion.sum(axis=0).tolist())),
        simultaneous_hand_ticks=int((motion[:,0]&motion[:,1]).sum()),
        simultaneous_wrist_ticks=int((motion[:,2]&motion[:,3]).sum()),
        simultaneous_bimanual_ticks=int(((motion[:,0]|motion[:,2])&(motion[:,1]|motion[:,3])).sum()),
        static_ticks=int((~any_motion&valid).sum()),dynamic_ticks=int((any_motion&valid).sum()),
        unknown_motion_ticks=int((~valid).sum()),stop_transitions=int(stops))


def replay_metrics(window,params,plan,reference,full=False,check_prefix=False):
    started=perf_counter()
    c,o,offline,prefix=window.replay(params,check_prefix)
    elapsed=perf_counter()-started
    dynamic,coverage=reference
    row=dict(recording_id=window.recording_id,window=window.label,ticks=len(c),
             causal_joint=sum(t['joint_valid'] for t in c),offline_joint=sum(t['joint_valid'] for t in o),
             causal_dynamic_joint=sum(t['joint_valid'] and bool(d) for t,d in zip(c,dynamic,strict=True)),
             offline_dynamic_joint=sum(t['joint_valid'] and bool(d) for t,d in zip(o,dynamic,strict=True)),
             motion=coverage,sync_seconds=elapsed)
    if not full:
        return row,None
    row['segments']={mode:{str(n):[hi-lo for lo,hi in segment_ranges(rows,offline,n)]
                     for n in plan['min_segment_probes']} for mode,rows in (('causal',c),('offline',o))}
    row['per_stream']={}
    for s in STREAMS:
        online=[tick['streams'][s] for tick in c]
        off=[tick['streams'][s] for tick in o]
        row['per_stream'][s]=dict(causal_valid=sum(v['valid'] for v in online),offline_valid=sum(v['valid'] for v in off),
            velocity_valid=sum(v['velocity_valid'] for v in online),pose_age_ns=summary([v['age_ns'] for v in online if v['valid']]),
            observed_proxy_speed_m_s=summary([float(speed) for v in off if v['velocity'] is not None
                                            for speed in np.linalg.norm(v['velocity'],axis=1)]),
            faults=dict(Counter(v['reason'] for v in online if not v['valid'])),
            offline_faults=dict(Counter(v['reason'] for v in off if not v['valid'])))
    row['recorded_streams']=stream_statistics(window.prepared)
    row['spatial']=spatial_metrics(c,o,window.prepared,window.topologies)
    moving_c=[r for r,m in zip(c,dynamic,strict=True) if m]
    moving_o=[r for r,m in zip(o,dynamic,strict=True) if m]
    row['dynamic_spatial']=spatial_metrics(moving_c,moving_o,window.prepared,window.topologies)
    row['prefix']=prefix
    row['unsafe_ticks']=sum(t['safety_reason']=='UNSAFE' for t in c)
    return row,(c,o,offline)


def coverage_retains(current,reference,tasks,rule,mode):
    def enough(k,n,fraction):
        d=sum(row[k] for row in reference if row['recording_id'] in n)
        return d==0 or sum(row[k] for row in current if row['recording_id'] in n)>=fraction*d
    roots=set().union(*tasks.values())
    return (enough(mode+'_joint',roots,rule['overall']) and enough(mode+'_dynamic_joint',roots,rule['dynamic']) and
            all(enough(mode+'_joint',use,rule['task']) for use in tasks.values()))


def narrower_offline_gap(tick,limit):
    return tick['joint_valid'] and all(
        tick['streams'][s]['time_interval'][1]-tick['streams'][s]['time_interval'][0]<=limit
        for s in CORE_STREAMS)


def candidate_qc(windows,qualified,split,plan):
    baseline=dict(plan['diag'],left_right_skew_ns=plan['diag']['skew_ns'],wrist_hand_skew_ns=plan['diag']['skew_ns'])
    active={x['recording_id'] for x in split['Train']}
    tasks=defaultdict(set)
    for q in qualified:
        if q['recording_id'] in active:
            tasks[q['task']].add(q['recording_id'])
    reference={}
    default=[]
    for w in windows:
        c,o,_,_=w.replay(baseline)
        tag=(w.recording_id,w.label)
        reference[tag]=motion_labels(o,plan['dynamic_threshold_m_s'],plan['stop_dwell_ticks'])
        labels,coverage=reference[tag]
        default.append(dict(recording_id=w.recording_id,window=w.label,ticks=len(c),motion=coverage,
            causal_joint=sum(t['joint_valid'] for t in c),offline_joint=sum(t['joint_valid'] for t in o),
            causal_dynamic_joint=sum(t['joint_valid'] and bool(d) for t,d in zip(c,labels,strict=True)),
            offline_dynamic_joint=sum(t['joint_valid'] and bool(d) for t,d in zip(o,labels,strict=True))))
    selections={}
    results=[]
    rule=plan['retention']
    for name,values in plan['grid_ms'].items():
        modes='offline' if name=='gap_ns' else 'causal'
        outputs={}
        offline_cache={}
        if modes=='offline':
            cap=dict(baseline,gap_ns=int(max(values)*1e6))
            for w in windows:
                if w.recording_id in active:
                    align=OfflineAlignment([e for e,_ in w.prepared],cap)
                    offline_cache[w.recording_id,w.label]=[align.step(t) for t,_ in w.ticks]
        for ms in values:
            params=dict(baseline,**{name:int(ms*1e6)})
            measurements=[]
            for w in windows:
                if w.recording_id not in active:
                    continue
                events=[e for e,_ in w.prepared]
                if modes=='offline':
                    wide=offline_cache[w.recording_id,w.label]
                    eligible=[narrower_offline_gap(t,params['gap_ns']) for t in wide]
                else:
                    eligible=[t['joint_valid'] for t in batch_causal(events,w.ticks,params,'SIMULATED')]
                dyn,_=reference[w.recording_id,w.label]
                measurements.append(dict(recording_id=w.recording_id,window=w.label,ticks=len(w.ticks),
                    **{modes+'_joint':sum(eligible),
                       modes+'_dynamic_joint':sum(t and bool(x) for t,x in zip(eligible,dyn,strict=True))}))
            outputs[ms]=measurements
            results.append(dict(axis=name,value_ms=ms,mode=modes,windows=measurements))
        alternatives=[ms for ms in values if ms*1e6<=baseline[name] and
                     coverage_retains(outputs[ms],outputs[max(values)],tasks,rule,modes)]
        selections[name]=int(alternatives[0]*1e6) if alternatives else baseline[name]
    params=dict(baseline,**selections)
    combined=[replay_metrics(w,params,plan,reference[(w.recording_id,w.label)])[0] for w in windows
              if w.recording_id in active]
    retention=all(coverage_retains(combined,default,tasks,rule,m) for m in ('causal','offline'))
    if not retention:
        params=baseline
    return params,dict(axis_results=results,diagnostic=default,combined_probe=combined,combined_retention=retention,
                       axis_recommendations=selections),reference


def characterize_thresholds(qualified,split,windows,plan):
    active={r['recording_id'] for r in split['Train']}
    captures=[r for r in qualified if r['recording_id'] in active]
    hands=[g for r in captures for g in r['geometry'].values()]
    wrist=[p['input_quat_unit_error']['max'] for r in captures for p in r['calibrated_side_pairs']]
    worst=max(wrist+[g['norm_error']['max'] for g in hands])
    tol=max(1e-6,10**ceil(log10(worst))) if worst>0 else 1e-6
    if tol>1e-4:
        raise ValueError('Raw quaternion precision exceeds audited tolerance range')
    bones=[e['absolute_deviation_m']['p99'] for h in hands for e in h.get('edges',[])]
    speed=[r['observed_proxy_speed_m_s']['p99'] for w,row in windows if w.recording_id in active
           for r in row['per_stream'].values() if r['observed_proxy_speed_m_s']['count']]
    return dict(quaternion_abs_unit_norm_max=worst,quaternion_tolerance=tol,
        quaternion_hard_limit=1e-4,bone_residual_p99_by_edge_m=summary(bones),
        bone_review_m=max(0.001,max(bones,default=0.)),
        source_clock={'negative_step_or_new_generation':'CUT', 'host_clock_pair_residual_ns':plan['archived_clock_pair_limit_ns'],
                      'host_observed_train_ns':summary([w.clock['max_abs_recorder_residual_ns'] for w,_ in windows
                                                        if w.recording_id in active])},
        speed_review={'measured_window_stream_p99_m_s':summary(speed),'mark_review_above_m_s':max(speed,default=0.),
                      'rule':'diagnostic flag only, no high-speed exclusion'},
        modifiers_are_hard={'unauthenticated':True,'cross_version':True,'unknown_time':True,'unsafe':True},
        minimum_segment_ticks=plan['chosen_segment_design_ticks'])
