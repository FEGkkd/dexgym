"""P01-D bounded qualification and the separately gated P01-E build entry."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from math import ceil
from pathlib import Path
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from time import perf_counter

from .hmb_build import BASE,SCHEMA,SYNC_HASH,USE,json_bytes,lock_sync,read_partition,schema,write_partition
from .p01d_probes import candidate_qc,characterize_thresholds,development_split,planned_windows,read_recording,replay_metrics
from .registry_gate import digest,load_rows
from .sync_audit import DOCS,baseline,emit,read
from .sync_evidence import CalendarDecoder
from .synchronize import STREAMS

RUN_BASE=Path('/home/ubuntu/dexgym_data/s0-5/p01/runs')
SDK='/home/ubuntu/teleop_ws/src/ManusSDK/lib/amd64/libManusSDK-amd64.so'
MODULE_PATH=Path(__file__).parent
REQUIRED=(*STREAMS,'tracker_safety')


def registry(a):
    entries=load_rows(a/'registry.jsonl')
    return {x['id']:x for x in entries},[x for x in entries if x.get('status')=='ADMITTED_CANDIDATE'
               and x.get('producer_kind')=='LIVE_TELEOP' and x.get('root_episode_id')]


def history(a):
    roles=defaultdict(set)
    with (a/'prior_use.jsonl').open() as f:
        for line in f:
            record=json.loads(line)
            for root in record.get('root_episode_ids') or [record.get('root_episode_id')]:
                if root:
                    roles[root].add(record.get('observed_role') or record.get('role') or 'USAGE_UNKNOWN')
    return roles


def verify_c(directory,receipt_rel,revision):
    receipt=read(DOCS/receipt_rel)
    external=directory/Path(receipt_rel).name
    if external.exists() and digest(external)!=digest(DOCS/receipt_rel):
        raise ValueError('Historical receipt mismatch')
    products=directory/'products'
    manifest=read(products/'output_sha256.json')
    if (digest(products/'output_sha256.json')!=receipt['products_manifest_sha256'] or
            any(digest(products/path)!=h for path,h in manifest.items())):
        raise ValueError(f'Sealed historical products differ: {directory}')
    for path,witness in receipt['execution_source_sha256'].items():
        raw=subprocess.check_output(['git','show',f'{revision}:{path}'])
        if sha256(raw).hexdigest()!=witness:
            raise ValueError(f'Historical source hash mismatch: {path}')
    return len(manifest),len(receipt['execution_source_sha256'])


def resources(plan,needed=0):
    cap=plan['budgets']
    available=shutil.disk_usage(RUN_BASE).free
    mem=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
             if line.startswith('MemAvailable:'))
    if available<needed+cap['minimum_free_bytes'] or mem<cap['minimum_ram_available_bytes']:
        raise ValueError('Resource pause: minimum disk reserve or RAM not available')
    return dict(disk_available_bytes=available,ram_available_bytes=mem)


def runtime_provenance():
    import mcap
    import numpy
    import scipy
    import yaml
    import dexgym
    from importlib.metadata import version
    packages={m.__name__:str(Path(m.__file__).resolve()) for m in (mcap,numpy,scipy,yaml,dexgym)}
    if Path(packages['dexgym']).parents[1]!=Path.cwd()/'source':
        raise ValueError('Unexpected DexGym module resolution')
    return dict(python=sys.executable,version=platform.python_version(),sys_path=sys.path,
                modules=packages,package_versions={x:version(x) for x in ('mcap','numpy','scipy','PyYAML')},
                overlays=os.environ.get('AMENT_PREFIX_PATH','').split(os.pathsep))


def hints():
    return (read(DOCS/'evidence/P01_B_review/topic_dependency_matrix.json'),
            read(DOCS/'evidence/P01_B/sample_plan.json')['native_writer'])


def asset_records(reg,row):
    return [reg[k] for k in row['source_asset_ids']]


def apparent_bytes(assets):
    return sum(r['size_bytes'] for r in assets if r['role'].startswith('streams/') and
               r['role'].split('/')[1] in REQUIRED)


def attach_positive_and_negative_controls(row):
    checks=row['prefix']
    return all(checks[k] for k in ('future_deleted_equal','future_pose_clock_valid_safety_changed_equal',
                                   'streaming_equals_batch','future_anchor_equal'))


def publish_args(window,frames,source,params,role,min_segment):
    c,o,align=frames
    identity={'recording_id':source['recording_id'],'root_episode_id':source['root_episode_id'],
              'segment_block':window.label,'verification':'SIMULATED','allowed_use':USE,'development_split':role,
              'episode_id':source['episode_id'],'collector_instance_id':source['collector_instance_id'],
              'raw_interval_realtime_ns':window.raw_interval,
              'tested_host_clock':window.clock,'parameters':params,'source_manifest_sha256':source['manifest_sha256'],
              'calibration_sha256':source['calibration_sha256']}
    return dict(identity=identity,prepared=window.prepared,topologies=window.topologies,
                causal=c,offline=o,alignment=align,min_segment=min_segment)


def pilot_partitions(output,windows,by_id,roles,params,plan):
    started=perf_counter()
    serial=[]
    names=[]
    jobs=[]
    for w,frames in windows:
        name=f"{w.recording_id.split(':')[-1]}-{w.label}"
        names.append(name)
        jobs.append(publish_args(w,frames,by_id[w.recording_id],params,roles[w.recording_id],
                                 plan['chosen_segment_design_ticks']))
        serial.append(write_partition(output/'pilot-serial'/name,**jobs[-1]))
    serial_seconds=perf_counter()-started
    # Inject a fault after real file writes; intentionally keep the incomplete staging object.
    fault=output/'intentional-failure'/names[0]
    try:
        write_partition(fault,**jobs[0],fail_before_publish=True)
    except RuntimeError as exc:
        if str(exc)!='Injected stop after fsync before atomic publication':
            raise
    else:
        raise ValueError('Fault injection failed to stop publication')
    staged=list(fault.parent.glob(f'.{fault.name}.*.partial'))
    hidden=sum(p.stat().st_size for stage in staged for p in stage.iterdir() if p.is_file())
    if fault.exists() or len(staged)!=1:
        raise ValueError('Failure exposed a partial published shard')
    repaired,repair_bytes=write_partition(fault,**jobs[0])
    if repaired!=serial[0][0] or write_partition(fault,**jobs[0])[1]!=0:
        raise ValueError('Recovery did not produce an idempotent shard')
    begin=perf_counter()
    with ThreadPoolExecutor(max_workers=plan['budgets']['active_workers']) as pool:
        pending=[pool.submit(write_partition,output/'pilot-parallel'/name,**job) for name,job in zip(names,jobs)]
        parallel=[task.result() for task in pending]
    parallel_seconds=perf_counter()-begin
    if [v[0] for v in serial]!=[v[0] for v in parallel]:
        raise ValueError('Serial and parallel pilot hashes differ')
    ticks=0
    nmax=0
    cohort=[]
    for name,witness in zip(names,serial,strict=True):
        path=output/'pilot-serial'/name
        a,meta,src=read_partition(path,'causal')
        b,_,_=read_partition(path,'offline')
        if len(a['tick_ns'])!=len(b['tick_ns']) or not src:
            raise ValueError('Empty or mismatched round-trip blocks')
        finite=a['pose_valid'][:,0].any()
        ticks+=int(a['joint_valid'].sum())
        nmax=max(nmax,meta['N'])
        cohort.append(dict(name=name,hashes=witness[0]['files'],bytes=witness[1],ticks=meta['ticks'],
                           physical_refs=len(src),finite_left_pose=bool(finite),N=meta['N']))
    staging_total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if staging_total>plan['budgets']['derived_bytes']:
        raise ValueError('P01-D pilot output budget exceeded')
    return dict(partitions=cohort,serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,
                writes_bytes_total=staging_total,serial_bytes=sum(n for _,n in serial),partial_bytes=hidden,
                recovery_bytes=repair_bytes,archive_peak_count=2,max_nodes=nmax,causal_joint_ticks=ticks,
                parallel_equal_sha=True,failed_target_hidden=True,idempotent_recovery=True)


def resource_report(qualified,study,pilot,plan):
    caps=plan['budgets']
    members=[m for q in qualified for m in q['source_read'].values()]
    read_bytes=sum(v['asset_bytes'] for v in members)
    read_time=sum(v['mcap_index_read_seconds'] for v in members)
    cdr=sum(v['cdr_messages'] for v in members)
    elapsed=sum(v['cdr_and_extract_seconds'] for v in members)
    seconds=sum((q['recorded_interval_ns'][1]-q['recorded_interval_ns'][0])*1e-9 for q in qualified)
    max_nodes=max(pilot['max_nodes'],max(len(t['joints']) for q in qualified for t in q['topologies']))
    ticks_total=ceil(seconds*50)+len(qualified)
    # f8 XYZ/XYZW arrays, fixed integer/mask columns and worst-case uncompressed text for each indexed source.
    upper_array_per_tick=2*(6*max_nodes*(8*(7+7+3)+1)+6*128+128)
    source_density=max(sum(m['source_records'] for m in q['source_read'].values())/
                       (sum((w['anchor_interval_realtime_ns'][1]-w['anchor_interval_realtime_ns'][0])*1e-9
                            for w in q['run_clocks'])) for q in qualified)
    seconds_with_context=seconds*1.15
    text_bound=ceil(seconds_with_context*max(1800,source_density*2)*8192)
    archive=ceil((upper_array_per_tick*ticks_total+text_bound)*1.25)
    peak=2*archive
    scan_total=sum(ceil((q['recorded_interval_ns'][1]-q['recorded_interval_ns'][0])*1e-9/
                  caps['production_partition_seconds'])*sum(m['asset_bytes'] for m in q['source_read'].values())
                   for q in qualified)
    measured=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,1)*1024
    free=resources(plan,peak)
    seconds_pilot=sum(x['ticks'] for x in pilot['partitions'])/50
    return dict(qualified_sources_measured=len(qualified),observed_read_bytes=read_bytes,
        mcap_index_read_seconds=read_time,mcap_index_read_bytes_s=read_bytes/read_time,
        decoded_wire_messages=cdr,cdr_decode_extract_messages_s=cdr/elapsed,
        sync_ticks_s=sum(x['ticks'] for x in study)/sum(x['sync_seconds'] for x in study),
        highwater_process_rss_bytes=measured,ram_budget_bytes=max(measured*4,caps['minimum_ram_available_bytes']),
        pilot=pilot,derived_bytes_per_raw_second=pilot['serial_bytes']/seconds_pilot,
        derived_bytes_per_valid_causal_tick=pilot['serial_bytes']/max(1,pilot['causal_joint_ticks']),
        observed_pilot_write_amplification=pilot['writes_bytes_total']/pilot['serial_bytes'],
        production=dict(max_active_workers=caps['active_workers'],partition_seconds=caps['production_partition_seconds'],
            projected_single_pass_archive_bytes=archive,required_peak_including_staging_bytes=peak,
            conservative_repeated_index_read_bytes=scan_total,
            index_read_time_estimate_seconds=scan_total*2/(read_bytes/read_time),max_topology_nodes=max_nodes,
            disk_hard_reserve_bytes=caps['minimum_free_bytes'],per_chunk_published_cap_bytes=ceil(archive/seconds*12),
            physical_record_pause_per_part=caps['physical_per_part'],minimum_available_ram_bytes=max(measured*4,caps['minimum_ram_available_bytes']),
            pause_on=['disk/RSS reserve','unexpected source hash/schema/calibration','archive or record cap']),
        **free)


def aggregate_coverage(study,by_id,minimum):
    grouped=defaultdict(list)
    totals=[item for _,item in study]
    for _,item in study:
        source=by_id[item['recording_id']]
        grouped[item['recording_id']].append(item)
    per_source=[]
    for key,rows in grouped.items():
        source=by_id[key]
        observed=sum(r['ticks'] for r in rows)
        source_s=(source['recorded_interval_ns'][1]-source['recorded_interval_ns'][0])*1e-9
        lengths=[v for item in rows for v in item['segments']['causal'][str(minimum)]]
        eligible=sum(lengths)
        per_source.append(dict(recording_id=key,parent=source['root_episode_id'],task=source['task'],
            date=source['capture_date'],raw_source_seconds=source_s,sampled_seconds=observed*.02,
            sampled_causal_joint_ticks=sum(x['causal_joint'] for x in rows),sampled_valid_segment_ticks=eligible,
            sampled_invalid_joint_seconds=(observed-sum(x['causal_joint'] for x in rows))*.02,
            sampled_valid_joint_seconds=sum(x['causal_joint'] for x in rows)*.02,
            sampled_valid_segment_seconds=eligible*.02,observed_segment_count=len(lengths),
            observed_segment_ticks=lengths,projection_50hz_valid_ticks=round(source_s*50*eligible/observed)))
    motion=Counter()
    stream_motion=Counter()
    for row in totals:
        motion.update({k:v for k,v in row['motion'].items() if k!='per_core_moving'})
        stream_motion.update(row['motion']['per_core_moving'])
    groups={}
    for task in sorted({r['task'] for r in per_source}):
        selected={r['recording_id'] for r in per_source if r['task']==task}
        taskrows=[r for r in totals if r['recording_id'] in selected]
        counts=Counter()
        for row in taskrows:
            counts.update({k:v for k,v in row['motion'].items() if k!='per_core_moving'})
        groups[task]={'parents':len(selected),'ticks':sum(r['ticks'] for r in taskrows),'motion':dict(counts)}
    all_lengths=[n for r in per_source for n in r['observed_segment_ticks']]
    return dict(scope='bounded measurements and separate extrapolation',parents=len(per_source),
        groups=groups,source_rows=per_source,observed_ticks=sum(x['ticks'] for x in totals),windows=len(totals),
        observed_causal_joint=sum(x['causal_joint'] for x in totals),observed_offline_joint=sum(x['offline_joint'] for x in totals),
        observed_unsafe_ticks=sum(x['unsafe_ticks'] for x in totals),observed_segment_count=len(all_lengths),
        observed_segment_ticks_distribution={str(k):v for k,v in sorted(Counter(all_lengths).items())},
        motion=dict(motion),per_core_moving=dict(stream_motion),
        raw_seconds=sum(r['raw_source_seconds'] for r in per_source),
        extrapolated_50hz_valid_ticks=sum(r['projection_50hz_valid_ticks'] for r in per_source),
        extrapolation='per parent bounded valid-segment fraction × source duration × 50; P01-E replaces with observed',
        verified_control_ticks=0,Final_Test='BLOCKED')


def study(args):
    lock_sync()
    plan=read(args.plan)
    env=runtime_provenance()
    free=resources(plan)
    old_count,old_modules,old_diag,review=baseline(args)
    archived={}
    for path,label,rev,receipt in ((args.original_c,'C_original','0d9d351','evidence/P01_C/review_receipt.json'),
            (args.review_c,'C_review','aecfd2d','evidence/P01_C_review/review_receipt.json'),
            (args.final_c,'C_final',BASE,'evidence/P01_C_final/final_receipt.json')):
        archived[label]=verify_c(path,receipt,rev)
    assert digest(DOCS/'evidence/P01_C_final/final_receipt.json')=='d708acdba29e291043de17706e3dc77e51fff42105fa7fe76c1ed08256cef224'
    reg,candidates=registry(args.old_a)
    if len(candidates)!=32 or len({r['root_episode_id'] for r in candidates})!=32:
        raise ValueError('A1 frozen parent cohort differs')
    used=history(args.old_a)
    old_members={r['recording_id'] for r in review['hmb_real_replay']['fixtures']}
    matrix,writer=hints()
    sdk=CalendarDecoder(SDK)
    probes=[]
    qualified=[]
    all_windows=[]
    bytes_read=0
    count=0
    for row in sorted(candidates,key=lambda r:r['id']):
        rid=row['id']
        assets=asset_records(reg,row)
        input_bytes=apparent_bytes(assets)
        entry=dict(recording_id=rid,root_episode_id=row['root_episode_id'],task=row['task'],capture_date=row['capture_date'],
                   registry_input_file_bytes=input_bytes,prior_C_cohort=rid in old_members,
                   historical_roles=sorted(used.get(row['root_episode_id']) or {'USAGE_UNKNOWN'}),
                   final_test_eligible=False)
        print('bounded qualification',rid,row['task'],row['capture_date'],flush=True)
        try:
            if bytes_read+input_bytes>plan['budgets']['input_bytes']:
                raise ValueError('preregistered cumulative MCAP read cap')
            bytes_read+=input_bytes
            selected,witness=read_recording(row,assets,matrix,writer,sdk,plan,planned_windows(row,plan))
            count+=witness['source_event_count']
            if count>plan['budgets']['event_count']:
                raise ValueError('preregistered total event cap')
            entry.update(witness)
            if witness['status']=='QUALIFIED':
                if not any(any(t['joint_valid'] for t in w.replay(plan['diag'])[0]) for w in selected):
                    entry.update(status='BLOCKED',issues=['No jointly eligible bounded input at frozen diagnostic settings'])
                else:
                    qualified.append(entry)
                    all_windows.extend(selected)
        except (OSError,RuntimeError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            entry.update(status='BLOCKED',issues=[f'{type(exc).__name__}: {exc}'])
        probes.append(entry)
        print(rid,entry['status'],entry['issues'],flush=True)
    out=args.output
    emit(out/'source_qualification.json',dict(rows=probes,A1_unknown_quarantined=89,
        excluded_simulation=1,exclude=['EgoPro','derivations','unknown provenance'],bounded_body_rows=count,
        history={'A_B_B_review_outputs':old_count,'B_runtime_modules':old_modules,'B_diagnostics':old_diag,
                 'C_products_and_source_files':archived},environment=env,initial_resources=free))
    split=development_split(qualified,used,plan['seed'])
    roles={r['recording_id']:role for role in ('Train','Validation') for r in split[role]}
    overlap={r['root_episode_id'] for r in split['Train']}&{r['root_episode_id'] for r in split['Validation']}
    if overlap:
        raise ValueError('Split leakage')
    source_by_id={r['recording_id']:r for r in qualified}
    params,grid,dynamic=candidate_qc(all_windows,qualified,split,plan)
    calibration_qc=characterize_thresholds(qualified,split,[],plan)
    official=[]
    sample=[]
    seen=set()
    for window in all_windows:
        rid=window.recording_id
        quat_flags=window.bind_quaternion_qc(calibration_qc['quaternion_tolerance'])
        row,frames=replay_metrics(window,params,plan,dynamic[rid,window.label],full=True,check_prefix=True)
        if not attach_positive_and_negative_controls(row):
            raise ValueError('Final sync patch prefix regression')
        row['split']=roles[rid]
        row['quaternion_flagged_records']=quat_flags
        official.append((window,row))
        if len(sample)<plan['budgets']['pilot_windows'] and rid not in seen:
            sample.append((window,frames))
            seen.add(rid)
    if len(sample)!=plan['budgets']['pilot_windows']:
        raise ValueError('Insufficient independent physical pilot blocks')
    numerical=characterize_thresholds(qualified,split,official,plan)
    pilot=pilot_partitions(out,sample,source_by_id,roles,params,plan)
    budget=resource_report(qualified,[r for _,r in official],pilot,plan)
    coverage=aggregate_coverage(official,source_by_id,plan['chosen_segment_design_ticks'])
    whitelist=dict(schema='s05.hmb.whitelist.v1',status='PROPOSED_PENDING_HUMAN',allowed_use=USE,
        sources=[{k:q[k] for k in ('recording_id','root_episode_id','manifest_sha256','snapshot_sha256','capture_date',
                     'calibration_sha256','episode_id','collector_instance_id','assets','task','recorded_interval_ns','topologies')}|
                  {'split':roles[q['recording_id']]}
                 for q in qualified],blocked_roots=[q['root_episode_id'] for q in probes if q['status']!='QUALIFIED'])
    report=dict(replay_parameters=params,numerical_qc=numerical,selected_from='Train only',window_results=[r for _,r in official],
                validation_used_for_selection=False,development_scope=USE,geometry_is_ground_truth=False)
    artifacts={'source_whitelist.json':whitelist,'qc_threshold_report.json':report,'qc_sensitivity_results.json':grid,
        'hmb_schema.json':schema(),'reason_code_registry.json':{'schema':SCHEMA,'codes':schema()['vocabularies']['reason']},
        'split.json':split,'split_isolation_report.json':{'status':'PASS','overlap_count':len(overlap),'all_assignments_are_roots':True,
            'whole_root_inheritance_required':True,'session_certified':False,'historical_data_only':True},
        'dataset_coverage.json':coverage,'resource_budget.json':budget,
        'build_protocol.json':dict(schema=SCHEMA,allowed_use=USE,entry='python -m dexgym.tasks.direct.wholebody_s05.data.p01d_freeze build',
            arguments=['--config <approved frozen_config.json>','--old-a <A1 registry directory>',
                       '--output <new approved P01-E run products directory>','--approval-state <phase_state.json>'],
            requires=['P01-D ACCEPTED with frozen_config_sha256','P01-E IN_PROGRESS with separate human authorization',
                      'locked source/code hash and current resource gate'],
            resume='Use the same output directory and verified config after an authorized retry. Published partitions are immutable.',
            interval='fixed 8 second records; per-partition context masked, no stitching; monotonic tick axis')}
    for key,value in artifacts.items():
        emit(out/key,value)
    decision=dict(schema='s05.p01d.freeze_candidate.v1',approval='PROPOSED_PENDING_HUMAN',allowed_use=USE,
        sync_baseline=BASE,sync_source_sha256=SYNC_HASH,parameters=params,numerical_qc=numerical,
        registry_sha256=digest(args.old_a/'registry.jsonl'),plan=plan,
        artifact_sha256={k:digest(out/k) for k in ('source_whitelist.json','hmb_schema.json','split.json',
                                                   'reason_code_registry.json','resource_budget.json','build_protocol.json')},
        implementation_sha256={p.name:digest(p) for p in [MODULE_PATH/k for k in ('capture_windows.py','hmb_build.py',
                               'p01d_probes.py','p01d_freeze.py','synchronize.py')]})
    emit(out/'frozen_config.json',decision)
    print(json.dumps({'qualified':len(qualified),'blocked':32-len(qualified),'Train':len(split['Train']),
          'Validation':len(split['Validation']),'windows':coverage['windows'],'ticks':coverage['observed_ticks'],
          'parameters':params,'pilot':pilot['serial_bytes'],'scope':USE}))


def approved(config_path,phase_path,destination):
    lock_sync()
    config=read(config_path)
    if config['schema']!='s05.p01d.freeze_candidate.v1' or config['allowed_use']!=USE or config['sync_source_sha256']!=SYNC_HASH:
        raise ValueError('Incompatible frozen configuration')
    if phase_path.resolve()!= (DOCS/'phase_state.json').resolve():
        raise ValueError('Production requires the tracked phase ledger')
    state=read(phase_path)
    events=state['p01_subphase_events']
    signed=next((e for e in reversed(events) if e['phase']=='P01-D' and e['to']=='ACCEPTED'),{})
    start=next((e for e in reversed(events) if e['phase']=='P01-E' and e['to']=='IN_PROGRESS'),{})
    if (state['p01_subphases']['P01-D']!='ACCEPTED' or state['p01_subphases']['P01-E']!='IN_PROGRESS' or
            signed.get('approval',{}).get('type')!='human' or signed.get('frozen_config_sha256')!=digest(config_path) or
            not start.get('authorization_ref') or start.get('frozen_config_sha256')!=digest(config_path)):
        raise ValueError('P01-E build has no approved freeze and distinct execution authorization')
    relative=destination.resolve().relative_to(RUN_BASE)
    if not relative.parts[0].startswith('P01-E-') or len(relative.parts)<2:
        raise ValueError('Use an isolated P01-E run output directory')
    for file,witness in config['artifact_sha256'].items():
        if digest(config_path.parent/file)!=witness:
            raise ValueError('Approved configuration sidecar differs')
    for file,witness in config['implementation_sha256'].items():
        if digest(MODULE_PATH/file)!=witness:
            raise ValueError('Approved builder source differs')
    if read(config_path.parent/'hmb_schema.json')!=json.loads(json_bytes(schema())):
        raise ValueError('Approved array schema differs from writer')
    return config


def build(args):
    raise ValueError('The archived D1 builder is retired. Use the separately reviewed P01-E entry.')


def main():
    parser=ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    probe=commands.add_parser('study')
    for key in ('plan','old-a','old-b','old-review','resource','original-c','review-c','final-c','output'):
        probe.add_argument('--'+key,type=Path,required=True)
    production=commands.add_parser('build')
    for key in ('config','old-a','approval-state','output'):
        production.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='build':
        build(args)
    else:
        relative=args.output.resolve().relative_to(RUN_BASE)
        if not relative.parts[0].startswith('P01-D-') or len(relative.parts)<2:
            parser.error('Use an isolated P01-D output directory')
        args.output.mkdir(mode=0o700)
        study(args)


if __name__=='__main__':
    main()
