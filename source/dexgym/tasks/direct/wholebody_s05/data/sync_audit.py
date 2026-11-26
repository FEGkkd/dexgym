"""Real bounded P01-C execution. It never creates a production Motion Bank."""
import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import yaml

from .calibration import extrinsics,verify_pairs
from .capture_windows import frozen_windows,scan_stream,verify_source
from .geometry import summary
from .registry_gate import digest,load_rows
from .synchronize import OfflineAlignment,REASONS,batch_causal
from .sync_evidence import (CalendarDecoder,DOMAIN_FIELDS,SDK_SHA,compare_prefix,eventize,
                            export_alignment,export_raw,json_result,jsonline,measure,stream_statistics)


DOCS=Path('docs/s0-5')


def read(path):
    return json.loads(path.read_bytes())


def emit(path,object):
    with path.open('x',encoding='utf-8') as f:
        json.dump(json_result(object),f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def baseline(args):
    a,b,br=args.old_a,args.old_b,args.old_review
    check=runpy.run_path(str(DOCS/'evidence/P01_B_review/run_hmb_review.py'))
    counts,modules,previous_logs,_=check['verify_provenance'](a,b,args.resource)
    review=read(DOCS/'evidence/P01_B_review/review_receipt.json')
    counts['B_review']=check['verify_outputs'](br,review)
    for f,h in read(DOCS/'evidence/P01_B_review/process_sha256.json').items():
        if digest(br/f)!=h:
            raise ValueError(f'B review process witness drift: {f}')
    return counts,modules,previous_logs,review


def coordinate_frame_witness(capture,calibration,target):
    return {
        'recording_id':capture['recording_id'],'ros_registered_target':target,
        'manus_initialization': 'world-coordinates true; XFromViewer/PositiveZ/Right; unitScale=1.0 metre',
        'manus_frame':'unregistered SDK common coordinates; empty sensor frame_id preserved as UNKNOWN',
        'palm_frame':'per-source root inverse; no world/body alignment',
        'tracker_pose':'OpenVR mDeviceToAbsoluteTracking; ROS x,y,z,w in right handed frame',
        'registered_wrist':'T_target_parent * T_parent_raw * T_raw_wrist, from ROS tracker transformer',
        'openvr_to_teleop_world':'recorded world adapter configuration and fixed rotation; no base calibration implied',
        'robot_base_physical':'BLOCKED_NO_INDEPENDENT_ROBOT_REGISTRATION',
        'tracker_extrinsics':calibration,
        'capture_unit_revisions':{k:v['revision'] for k,v in capture['configs'].items()},
    }


def script_topology_plot(path,prepared,topo):
    """Use actual two-dimensional projections of the source and derived palm bones."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(8,8),layout='constrained')
    for i,side in enumerate(('left','right')):
        row=next((r for e,r in prepared if e.stream=='manus_'+side and e.valid),None)
        if row is None:
            continue
        t=next(x for x in topo if x['stream']=='manus_'+side and x['version']==row['manus']['topology_sha256'])
        for j,(key,x) in enumerate((('raw_pose','SDK common'),('processed_pose','Derived palm'))):
            ax=axes[i,j]; z=row[key]
            for q,p in t['edges']:
                ax.plot(z[[q,p],0],z[[q,p],1],'-o',markersize=3,color='#126d84')
            ax.set(title=f'{side.capitalize()} • {x}',xlabel='X (m)',ylabel='Y (m)',aspect='equal')
            ax.grid(alpha=.25)
    fig.savefig(path,dpi=150)
    plt.close(fig)


def candidate_sensitivity(events,ticks,params):
    results=[]
    for overrides in ({'age_ns':30_000_000},{'age_ns':90_000_000},{'left_right_skew_ns':20_000_000},
                      {'left_right_skew_ns':50_000_000},{'wrist_hand_skew_ns':20_000_000},
                      {'wrist_hand_skew_ns':50_000_000},{'safety_age_ns':60_000_000},{'safety_age_ns':150_000_000},
                      {'gap_ns':30_000_000},{'gap_ns':90_000_000}):
        merged=dict(params,**overrides)
        if 'gap_ns' in overrides:
            off=OfflineAlignment(events,merged)
            covered=sum(off.step(tick)['joint_valid'] for tick,_ in ticks)
            results.append(dict(change=overrides,offline_joint_valid=covered,total_ticks=len(ticks)))
        else:
            covered=sum(x['joint_valid'] for x in batch_causal(events,ticks,merged,'SIMULATED'))
            results.append(dict(change=overrides,causal_joint_valid=covered,total_ticks=len(ticks)))
    return results


def process(args):
    plan=read(args.plan)
    counts,modules,previous_logs,review=baseline(args)
    reg={r['id']:r for r in load_rows(args.old_a/'registry.jsonl')}
    held={r['recording_id']:r for r in review['hmb_real_replay']['fixtures']}
    matrix=read(DOCS/'evidence/P01_B_review/topic_dependency_matrix.json')
    writer=read(DOCS/'evidence/P01_B/sample_plan.json')['native_writer']
    sdk=CalendarDecoder('/home/ubuntu/teleop_ws/src/ManusSDK/lib/amd64/libManusSDK-amd64.so')
    pending,calibs,frames,topology,hosts,fixtures,offline_manifest,causal_manifest,raw_manifest=[],[],[],[],[],[],[],[],[]
    clock_stats,stability,sensitivity,measures=[],[],[],[]
    mask_counts=Counter()
    unsafe_count,records_read,actual_windows,bytes_read=0,0,0,0
    nwin=0
    plots=[]
    out=args.output
    with (out/'source_trace.jsonl').open('x',encoding='utf-8') as trace,(out/'invalid_rows.jsonl').open('x',encoding='utf-8') as bad:
        for rid in plan['selected']:
            print('bounded capture',rid,file=sys.stderr,flush=True)
            try:
                row=reg[rid]
                assets=[reg[i] for i in row['source_asset_ids']]
                m,parts,proof=verify_source(row,assets,held[rid],matrix,writer)
                input_bytes=sum(x[0]['size_bytes'] for p in parts.values() for x in p)
                bytes_read+=input_bytes
                if input_bytes>plan['episode_stream_bytes_limit'] or bytes_read>plan['total_stream_bytes_limit']:
                    raise ValueError('Predeclared read budget exceeded')
                ext=next(z for z in proof['configs']['tracker']['refs'] if z['original_path'].endswith('/tracker_calibration.yaml'))
                target_file=next(z for z in proof['configs']['tracker']['refs'] if z['original_path'].endswith('/transformer.yaml'))
                target=yaml.safe_load(Path(target_file['path']).read_bytes())['tracker_pose_transformer']['ros__parameters']['target_frame']
                cal=extrinsics(Path(ext['path']))
                start,stop=row['recorded_interval_ns']
                selected_unsafe=scan_stream('tracker_safety',parts['tracker_safety'],{},0,
                                            plan['physical_message_limit_per_part'],find_unsafe=True)
                windows=frozen_windows(start,stop,plan,selected_unsafe)
                sources=[]
                stream_counts={}
                for stream,stream_parts in sorted(parts.items()):
                    v,count=scan_stream(stream,stream_parts,windows,plan['context_ns'],plan['physical_message_limit_per_part'])
                    stream_counts[stream]={'manifest_physical_payloads':count,'window_payloads':len(v),
                                           'source_parts':[x[0]['id'] for x in stream_parts]}
                    sources.extend(v)
                if records_read+len(sources)>plan['event_limit'] or actual_windows+len(windows)>plan['max_windows']:
                    raise ValueError('Frozen event or window budget exceeded')
            except (OSError,ValueError,KeyError,RuntimeError) as exc:
                pending.append(dict(recording_id=rid,reason='SOURCE_OR_DECODE_BLOCKED',detail=str(exc)))
                continue
            records_read+=len(sources)
            actual_windows+=len(windows)
            unsafe_count+=sum(r['fields'].get('safe') is False for r in sources if r['stream']=='tracker_safety')
            proof.update(windows_realtime_ns=windows,stream_coverage=stream_counts,
                         safety_first_unsafe_log_ns=selected_unsafe,actual_bytes=sum(
                           x[0]['size_bytes'] for p in parts.values() for x in p))
            fixtures.append(proof)
            frames.append(coordinate_frame_witness(proof,cal,target))
            by_stream={}
            for stream in sorted(parts):
                by_stream[stream]=[r for r in sources if r['stream']==stream]
            for side in ('left','right'):
                test=verify_pairs(by_stream['tracker_'+side+'_raw'],by_stream['tracker_'+side],cal[side]['transform'],target)
                test.update(recording_id=rid,side=side,calibration_sha256=ext['sha256'],target_frame=target)
                calibs.append(test)
            mono_origin=m['start_monotonic_ns']
            observed,topo,invalid,stats=eventize(sources,mono_origin,plan['diagnostic'],ext['sha256'],sdk)
            topology.extend(dict(**r,recording_id=rid) for r in topo)
            hosts.append(dict(recording_id=rid,geometry_statistics=stats))
            for e in invalid:
                jsonline(bad,dict(**e,recording_id=rid))
            if not plots or (len(plots)==1 and row['task']=='bi_hand'):
                graphic=out/f'manus-{len(plots)+1:02}.png'
                script_topology_plot(graphic,observed,topo)
                plots.append(graphic.name)
            for window,(lo,hi) in windows.items():
                epoch=m['start_realtime_ns']
                ticks=[(t,t) for t in range(lo-epoch,hi-epoch,plan['diagnostic']['tick_ns'])]
                tagged=[(e,r) for e,r in observed if window in r['windows']]
                events=[e for e,r in tagged]
                base,negatives=compare_prefix(events,ticks,plan['diagnostic'])
                offline=OfflineAlignment(events,plan['diagnostic'])
                off=[offline.step(t) for t,_ in ticks]
                met=measure(base,off)
                meta={'recording_id':rid,'window':window,'task':row['task'],'capture_date':row['capture_date']}
                for a,b in zip(base,off):
                    for s,state in a['streams'].items():
                        mask_counts[state['reason']]+=1
                        if not state['valid']:
                            jsonline(bad,dict(kind='causal_mask',stream=s,reason=state['reason'],
                                               tick_ns=a['tick_ns'],retained_ref=state['ref'],**meta))
                        c={k:v for k,v in state.items() if k not in {'pose','velocity'}}
                        d={k:v for k,v in b['streams'][s].items() if k not in {'pose','velocity'}}
                        jsonline(trace,dict(kind='aligned',stream=s,tick_ns=a['tick_ns'],cutoff_ns=a['cutoff_ns'],
                                             causal=c,offline=d,joint_valid=a['joint_valid'],
                                             safety_ref=a['safety_ref'],clock_quality='SIMULATED',**meta))
                stem=f'{nwin:02}'
                nwin+=1
                raw_manifest.append(dict(**meta,**export_raw(out,stem,tagged,trace,rid,window)))
                for dest,series,key,flag in ((offline_manifest,off,'offline',False),(causal_manifest,base,'causal',True)):
                    dest.append(dict(**meta,**export_alignment(out,f'{stem}-{key}',series,flag,events)))
                measures.append(dict(**meta,**met))
                clock_stats.append(dict(**meta,stream_stats=stream_statistics(tagged)))
                stability.append(dict(**meta,**negatives))
                sensitivity.append(dict(**meta,changes=candidate_sensitivity(events,ticks,plan['diagnostic'])))
                if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>plan['derived_bytes_limit']:
                    raise ValueError('Frozen derived output budget exceeded')
    topology_size=Counter(len(t['joints']) for t in topology)
    topo_groups={t['recording_id'] for t in topology}
    has_bad_geometry=any('GEOMETRY_INVALID' in line for line in (out/'invalid_rows.jsonl').read_text().splitlines())
    bad_calibration=any(c['failures'] or c['translation_m']['count']==0 or
                        c['translation_m']['max']>1e-5 or c['rotation_rad']['max']>1e-5 for c in calibs)
    c1=('FAIL' if has_bad_geometry or bad_calibration else
        'BLOCKED' if pending or len(topo_groups)!=len(plan['selected']) else 'PASS')
    c2_algo=('FAIL' if any(not all(r[k] for k in ('future_deleted_equal','future_pose_clock_valid_safety_changed_equal',
                                                  'streaming_equals_batch','future_anchor_equal')) for r in stability)
             else 'BLOCKED' if not stability or pending else 'PASS')
    total_ticks=sum(x['tick_count'] for x in measures)
    coverage=dict(selected=len(plan['selected']),clean_recordings=len(fixtures),sample_windows=actual_windows,
                  episode_input_bytes=bytes_read,raw_window_events=records_read,ticks=total_ticks,
                  window_seconds=actual_windows*plan['width_ns']*1e-9,
                  naive_independent_events_not_claimed=True,real_unsafe_rows=unsafe_count,
                  causal_joint_valid=sum(x['causal_joint_valid_count'] for x in measures),
                  offline_joint_valid=sum(x['offline_joint_valid_count'] for x in measures),
                  real_verified_joint_valid=0,scoring_scope='selected, bounded, mixed motion development diagnostic',
                  unresolved_sources=pending)
    products={
        'geometry_contract.json':dict(schema='s05.p01c.geometry.v1',convention='XYZ metres / XYZW right handed',
             raw_sdk_frame_registration='UNKNOWN beyond configured SDK origin',
             bone_length='difference of common-frame points, palmar variant derived by root inverse',
             raw_axes='per message and original order retained',calibration_rule='T_A_C = T_A_B T_B_C'),
        'manus_topology_report.json':dict(topologies=topology,node_count_occurrences=dict(topology_size),metrics=hosts,plots=plots),
        'coordinate_frame_report.json':dict(recordings=frames),
        'calibration_report.json':dict(rows=calibs,check_C1=c1,reconstruction_numeric_candidate=1e-5,
             ground_truth='independently recorded registered-wrist samples, no external base measurement'),
        'clock_domain_registry.json':dict(domains=DOMAIN_FIELDS,normalized_tick_unit='ns',
             unknown_raw_array_sentinel=-1,absolute_cross_device_mapping=False),
        'clock_quality.json':dict(windows=clock_stats,real_availability='UNUSABLE',collector_diagnostic='PROXY',
             tested_input='SIMULATED',reason='No independent physical capture to controller arrival upper bound'),
        'clock_mapping_report.json':dict(mapped_live=False,anchors_verified=0,physical_capture='UNUSABLE',
             controller_arrival='UNUSABLE',clock_eligibility='BLOCKED',
             simulated_mapping='raw collector monotonic minus manifest start; arrival = proxy capture + preset delay',
             simulated_delay_ns=plan['diagnostic']['arrival_delay_ns'],causal_anchor_intersection_tested=True),
        'sample_identity_report.json':dict(windows=clock_stats,definition='instance plus packed source / local receive / joint digest',
             source_capture_sequence_verified=False,physical_indices_retained=True,SDK_sha256=SDK_SHA),
        'duplicate_source_report.json':dict(rows=[dict(recording_id=x['recording_id'],window=x['window'],
             stream_identity={k:v['identity_events'] for k,v in x['stream_stats'].items()}) for x in clock_stats],
             age_refresh_on_duplicate=False),
        'aligned_offline_sample_manifest.json':dict(rows=offline_manifest,raw_windows=raw_manifest,
             timestamp_semantics='OFFLINE_PROXY',fields_order=['x','y','z','qx','qy','qz','qw'],publication='bounded audit only'),
        'aligned_causal_sample_manifest.json':dict(rows=causal_manifest,raw_windows=raw_manifest,
             timestamp_semantics='SIMULATED',source_proof='source_trace.jsonl',training_admission=False,
             fields_order=['x','y','z','qx','qy','qz','qw']),
        'prefix_mutation_results.json':dict(rows=stability,gate=c2_algo),
        'validity_report.json':dict(reasons=dict(mask_counts),reason_codes={j:i for i,j in enumerate(REASONS)},
             real_unsafe_rows=unsafe_count,initial_state_policy='excluded from BODY sampling; independent bounds needed in deployment'),
        'sync_error_budget.json':dict(windows=measures,physical_accuracy_status='UNMEASURED',
             budget_rule='proxy finite-difference v95 × hypothesized error; unknown physical age bound',
             static_split_candidate='offline proxy 0.02 m/s',offline_is_external_truth=False),
        'qc_threshold_candidates.json':dict(status='NOT_FROZEN',frozen_experiment_only=plan['diagnostic'],
             variations=sensitivity,assessments=threshold_guidance(measures,hosts,calibs,clock_stats),
             stratified_inputs='clock, validity, geometry, error budget reports'),
        'dataset_coverage.json':coverage,
        'fixtures_manifest.json':dict(recordings=fixtures,exclusions=pending),
        'technical_gates.json':{'C1':c1,'C2_algorithm':c2_algo,'C2_real_clock':'BLOCKED',
                                'P01-C':'FAILED' if 'FAIL' in {c1,c2_algo} else 'BLOCKED',
                                'history_sha_reverified':counts,'B_module_hashes':modules,'B_diagnostic_hashes':previous_logs,
                                'eligible_training_grants':0,'final_test':'BLOCKED'},
    }
    for name,doc in products.items():
        emit(out/name,doc)
    print(json.dumps({'coverage':coverage,'gates':products['technical_gates.json']},default=str))


def threshold_guidance(measures,geometry,calibration,clocks):
    def above(rows,key,cut):
        values=[row[key]['max'] for row in rows if row[key]['count']]
        return {'groups_tested':len(values),'groups_exceeding':sum(x>cut for x in values)}

    geometry_rows=[r for capture in geometry for r in capture['geometry_statistics'].values()]
    speeds=[w['proxy_speed_m_s']['p99'] for w in measures if w['proxy_speed_m_s']['count']]
    jump_samples=[r['clock_fields']['sdk_calendar_components']['step_ms']['p99']
                  for w in clocks for stream,r in w['stream_stats'].items()
                  if stream.startswith('manus_') and r['clock_fields']['sdk_calendar_components']['step_ms']['count']]
    scales=(1e-6,1e-5,1e-4)
    return {
        'age_skew_safety_stale_gap':{'basis':'observed proxy intervals / safety writer, numerical curves are diagnostic',
           'age_ms':[30,60,90],'left_right_skew_ms':[20,35,50],'wrist_hand_skew_ms':[20,35,50],
           'safety_age_ms':[60,100,150],'offline_gap_ms':[30,60,90],
           'risk':'proxy coverage does not set real controller deadlines'},
        'quaternion_unit_error':{'candidates':[
            {'threshold':t,'hand':above(geometry_rows,'norm_error',t),
             'wrist':above(calibration,'input_quat_unit_error',t)} for t in scales],
           'basis':'float32/64 and original norm distributions; report captures exceeding, no frame removal',
           'risk':'normalization cannot repair wrong axes'},
        'calibration_residual_in_m_or_rad':{'candidates':[
            {'threshold':t,'translation':above(calibration,'translation_m',t),
             'rotation':above(calibration,'rotation_rad',t)} for t in scales],
           'basis':'compare matched original and emitted wrists; numerical reproduction only',
           'risk':'processing agreement cannot certify independent physical base calibration'},
        'bone_length':{'candidates_mm_absolute_deviation':[1,2],
           'basis':'per-side per-node median and deviations; per-edge flagged counts in topology report',
           'risk':'fixed raw-length cutoff could remove recording-specific hands and motions'},
        'motion_speed':{'basis':'per-window causal/offline delta and static/dynamic proxy speed distributions',
           'candidates_m_s':[{'cut':x,'windows_with_p99_above':sum(s>x for s in speeds)}
                             for x in (float(np.median(speeds)),max(speeds))] if speeds else [],
           'risk':'percentile-derived screening probes, evaluate tail sources without discarding fast motions'},
        'clock_jump':{'basis':'order, domain, publisher source identity and independently anchored bounds',
           'negative_component_step_ms':0,
           'long_positive_step_candidates_ms':[{'cut':x,'hand_windows_with_p99_above':sum(v>x for v in jump_samples)}
                for x in (float(np.median(jump_samples)),max(jump_samples))] if jump_samples else [],
           'risk':'long positive gaps may be dropout or capture cadence; cross-device bounds remain unknown'},
        'joint_coverage':{'basis':'per-window denominator and scenario bins, no threshold selected',
           'candidates':[{'fraction':x,'windows_below':sum(m['causal_joint_valid_count']/m['tick_count']<x for m in measures)}
                         for x in (.90,.95,.99)],
           'risk':'single episode high coverage must not dominate'},
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','old-a','old-b','old-review','resource','output'):
        p.add_argument('--'+key,required=True,type=Path)
    args=p.parse_args()
    base=Path('/home/ubuntu/dexgym_data/s0-5/p01/runs')
    output=args.output.resolve()
    if (not output.is_relative_to(base) or
            not output.relative_to(base).parts[0].startswith('P01-C-')):
        p.error('Output must be isolated in the current P01-C run')
    args.output.mkdir(mode=0o700)
    process(args)


if __name__=='__main__':
    main()
