"""Repeat the original P01-C raw window audit with the reviewed replay semantics."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .calibration import extrinsics,verify_pairs
from .capture_windows import clock_observations,frozen_windows,scan_stream,verify_source
from .clock_domains import proxy_grid
from .registry_gate import digest,load_rows
from .review_metrics import paired_masks,spatial_metrics
from .sync_audit import DOCS,baseline,emit,read
from .sync_evidence import CalendarDecoder,audit_freshness,compare_prefix,eventize,jsonline,stream_statistics
from .synchronize import OfflineAlignment,STREAMS,batch_causal


def previous_results(legacy):
    audits=read(legacy/'output_sha256.json')
    receipt=read(DOCS/'evidence/P01_C/review_receipt.json')
    if (digest(legacy/'output_sha256.json')!=receipt['products_manifest_sha256'] or
            len(audits)!=92 or any(digest(legacy/file)!=sha for file,sha in audits.items())):
        raise ValueError('The sealed P01-C attempt-03 outputs differ')
    causal=read(legacy/'aligned_causal_sample_manifest.json')['rows']
    offline=read(legacy/'aligned_offline_sample_manifest.json')['rows']
    fixtures={row['recording_id']:row for row in read(legacy/'fixtures_manifest.json')['recordings']}
    return {(r['recording_id'],r['window']):(r,o) for r,o in zip(causal,offline,strict=True)},fixtures


def trace_window(destination,rid,label,causal,offline):
    for a,b in zip(causal,offline,strict=True):
        data={s:{k:v for k,v in row.items() if k not in {'pose','velocity'}}
              for s,row in a['streams'].items()}
        offline_indices={s:{k:row[k] for k in ('refs','time_interval','weight','valid','reason')}
                         for s,row in b['streams'].items()}
        jsonline(destination,dict(recording_id=rid,window=label,clock='collector_host_monotonic',
                                  tick_ns=a['tick_ns'],cutoff_ns=a['cutoff_ns'],joint_valid=a['joint_valid'],
                                  safety_permission=a['safety_permission'],safety_ref=a['safety_ref'],
                                  safety_release_ns=a['safety_release_ns'],safety_reason=a['safety_reason'],
                                  causal=data,offline=offline_indices,skew_ns=a['skew_ns'],
                                  offline_source_span_ns=b['skew_ns']))


def audit_window(rid,label,interval,sources,prepared,topologies,manifest,pair,args,out_trace,params):
    anchors=clock_observations(sources,label)
    ticks,timing=proxy_grid(*interval,anchors,manifest['start_monotonic_ns'],params['tick_ns'])
    context=audit_freshness([(e,row) for e,row in prepared if label in row['windows']])
    events=[e for e,_ in context]
    causal,controls=compare_prefix(events,ticks,params)
    interpolator=OfflineAlignment(events,params)
    offline=[interpolator.step(t) for t,_ in ticks]
    raw_refs={e.ref['raw_index_token'] for e in events}
    for tick in causal:
        for state in tick['streams'].values():
            if state['valid'] and (state['capture'][1]>tick['tick_ns'] or state['arrival'][1]>tick['cutoff_ns'] or
                                   state['ref']['raw_index_token'] not in raw_refs):
                raise ValueError('A causal state lacks bounds or traceable source identity')
    with np.load(args.legacy/pair[0]['file'],allow_pickle=False) as old, np.load(
            args.legacy/pair[1]['file'],allow_pickle=False) as old_off:
        previous_ticks=[(int(t),int(t)) for t in old['tick_ns']]
        if (len(ticks)!=len(previous_ticks) or len(ticks)!=100 or
                any(b[0]-a[0]!=params['tick_ns'] for a,b in zip(previous_ticks,previous_ticks[1:]))):
            raise ValueError('Window grid cardinality/spacing differs from the sealed nominal grid')
        timing['phase_shift_ns_from_original_ordinal']=ticks[0][0]-previous_ticks[0][0]
        timing['old_grid_revised_core_joint_valid']=sum(
            x['joint_valid'] for x in batch_causal(events,previous_ticks,params,'SIMULATED'))
        masks=paired_masks(old,old_off,causal,offline,events)
    trace_window(out_trace,rid,label,causal,offline)
    streams=stream_statistics(context)
    for stream,report in streams.items():
        report['freshness_unknown_events']=sum(e.stream==stream and not e.fresh for e in events)
        report['physical_capture_uncertainty_ns']=None
        report['availability_qualification']='SIMULATED'
    return dict(recording_id=rid,window=label,raw_realtime_interval_ns=interval,clock=timing,
                masks=masks,controls=controls,streams=streams,
                spatial=spatial_metrics(causal,offline,context,topologies))


def process(args):
    plan=read(args.plan)
    counts,modules,previous_logs,review=baseline(args)
    pairs,old_fixtures=previous_results(args.legacy)
    a={x['id']:x for x in load_rows(args.old_a/'registry.jsonl')}
    h={x['recording_id']:x for x in review['hmb_real_replay']['fixtures']}
    matrix=read(DOCS/'evidence/P01_B_review/topic_dependency_matrix.json')
    writer=read(DOCS/'evidence/P01_B/sample_plan.json')['native_writer']
    sdk=CalendarDecoder('/home/ubuntu/teleop_ws/src/ManusSDK/lib/amd64/libManusSDK-amd64.so')
    results=[]
    geometry=[]
    mass=Counter()
    with (args.output/'source_trace.jsonl').open('x',encoding='utf-8') as trace:
        for rid in plan['selected']:
            print(f'bounded native source {rid}',flush=True)
            row=a[rid]
            m,parts,proof=verify_source(row,[a[p] for p in row['source_asset_ids']],h[rid],matrix,writer)
            prior=old_fixtures[rid]
            if any(proof.get(k)!=prior.get(k) for k in ('root_episode_id','snapshot_sha256','manifest_sha256','code','configs')):
                raise ValueError(f'Capture provenance drift: {rid}')
            size=sum(x[0]['size_bytes'] for p in parts.values() for x in p)
            mass['input_file_bytes']+=size
            if size>plan['episode_stream_bytes_limit'] or mass['input_file_bytes']>plan['total_stream_bytes_limit']:
                raise ValueError('Predeclared capture budget exceeded')
            first_unsafe=scan_stream('tracker_safety',parts['tracker_safety'],{},0,
                                     plan['physical_message_limit_per_part'],find_unsafe=True)
            windows=frozen_windows(*row['recorded_interval_ns'],plan,first_unsafe)
            if {k:list(v) for k,v in windows.items()}!=prior['windows_realtime_ns']:
                raise ValueError(f'Frozen windows drift: {rid}')
            groups={}
            for stream,stream_parts in sorted(parts.items()):
                data,n=scan_stream(stream,stream_parts,windows,plan['context_ns'],plan['physical_message_limit_per_part'])
                old=prior['stream_coverage'][stream]
                if n!=old['manifest_physical_payloads'] or len(data)!=old['window_payloads']:
                    raise ValueError('Raw bounded coverage differs')
                groups[stream]=data
            sources=[r for data in groups.values() for r in data]
            mass['raw_window_events']+=len(sources)
            if mass['raw_window_events']>plan['event_limit'] or len(results)+len(windows)>plan['max_windows']:
                raise ValueError('Frozen event or window budget exceeded')
            mass['recordings']+=1
            mass['real_unsafe_events']+=sum(r['fields']['safe'] is False for r in groups['tracker_safety'])
            ext=next(x for x in proof['configs']['tracker']['refs'] if x['original_path'].endswith('/tracker_calibration.yaml'))
            cal=extrinsics(Path(ext['path']))
            archived_cal=read(args.legacy/'calibration_report.json')['rows']
            for side in ('left','right'):
                tracker=f'tracker_{side}'
                same=next(r for r in archived_cal if r['side']==side and r['recording_id']==rid)
                new=verify_pairs(groups[tracker+'_raw'],groups[tracker],cal[side]['transform'],same['target_frame'])
                if any(new[k]!=same[k] for k in ('matched','failures','translation_m','rotation_rad')):
                    raise ValueError('C1 geometric reconstruction differs from baseline')
                mass['registered_pairs']+=new['matched']
                geometry.append(dict(recording_id=rid,side=side,matched=new['matched'],translation_m=new['translation_m'],
                                     rotation_rad=new['rotation_rad'],calibration_sha256=ext['sha256']))
            events,topos,invalid,_=eventize(sources,m['start_monotonic_ns'],plan['diagnostic'],ext['sha256'],sdk)
            if any(r['reason']=='GEOMETRY_INVALID' for r in invalid):
                raise ValueError('Unexpected raw geometry failure')
            for label,interval in windows.items():
                report=audit_window(rid,label,interval,sources,events,topos,m,pairs.pop((rid,label)),args,trace,plan['diagnostic'])
                results.append(report)
    if pairs or len(results)!=23 or mass['raw_window_events']!=47983 or mass['registered_pairs']!=13640:
        raise ValueError('Audit differs from the frozen real corpus')
    if (args.output/'source_trace.jsonl').stat().st_size>plan['derived_bytes_limit']:
        raise ValueError('Output budget exceeded')
    metrics=('ticks','causal_old','causal_new','offline_old','offline_new','joint_lost','joint_gained')
    totals={k:sum(r['masks'][k] for r in results) for k in metrics}
    totals.update(mass)
    span=max(r['clock']['max_abs_recorder_residual_ns'] for r in results)
    if totals['causal_old']!=2248 or totals['ticks']!=2300:
        raise ValueError('Original mask/count reproduction differs')
    if not all(all(r['controls'][k] for k in ('future_deleted_equal','future_pose_clock_valid_safety_changed_equal',
                   'streaming_equals_batch','future_anchor_equal')) for r in results):
        raise ValueError('Prefix or stream equivalence regression')
    emit(args.output/'paired_regression.json',dict(results=results,totals=totals,
         thresholds=plan['diagnostic'],baseline_hashes=counts|{'C_original':92},
         historical_B_module_hashes=modules,historical_B_diagnostic_hashes=previous_logs,
         max_host_clock_pair_residual_ns=span,geometry=geometry,
         note='Pose/velocity/safety are independent. Nominal tick ordinal paired across the proxy clock correction.'))
    gates={'C1-GEO':'PASS','C2-SIM':'PASS' if span<=1_000_000 else 'BLOCKED',
           'C2-RT':'BLOCKED','robot-base':'BLOCKED','P01-T':'BLOCKED','Final_Test':'BLOCKED'}
    emit(args.output/'technical_gates.json',dict(gates=gates,verified_control_ticks=0,
         proxy_replay_recorder_consistency_guard_ns=1_000_000,units='ns',
         qualifies_hardware=False,grants_any_followup_phase=False))
    print(json.dumps(dict(totals=totals,gates=gates,recorder_consistency_ns=span)))


def main():
    parser=ArgumentParser(description=__doc__)
    for key in ('plan','old-a','old-b','old-review','resource','legacy','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    scope=Path('/home/ubuntu/dexgym_data/s0-5/p01/runs')
    resolved=args.output.resolve()
    if not resolved.is_relative_to(scope) or not resolved.relative_to(scope).parts[0].startswith('P01-C-REVIEW-'):
        parser.error('Use a new isolated P01-C Review run')
    args.output.mkdir(mode=0o700)
    process(args)


if __name__=='__main__':
    main()
