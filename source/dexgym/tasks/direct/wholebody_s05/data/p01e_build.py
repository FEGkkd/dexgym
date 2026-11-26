"""Authorized E0 production-format probe and separately gated full HMB builder."""
from argparse import ArgumentParser
from dataclasses import asdict
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import shutil
from time import perf_counter

import numpy as np

from .hmb_build import (USE, InjectedPublicationError, _fsync_dir, _write_fsynced,
                         json_bytes, read_partition, validate_partition_pair, write_partition)
from .p01d_freeze import RUN_BASE, SDK, asset_records, hints, publish_args, registry, resources
from .p01d_probes import InsufficientSourceEvidence, read_recording
from .p01e_ledger import Ledger
from .p01e_lock import D_DIR, PLAN, audited_d, authorize, candidate, lock_inputs, verify_lock
from .p01e_qualification import qualify_partition
from .registry_gate import digest
from .sync_audit import read
from .sync_evidence import CalendarDecoder


E0='E0_TRANSIENT_NON_PRODUCTION'
E1='E1_PRODUCTION'


class ResourcePause(RuntimeError):
    pass


def plan_blocks(pool,plan,config_sha,scope):
    result=[]
    width=plan['window_ns']
    for source in pool:
        start,stop=source['recorded_interval_ns']
        assets={row['id']:row['sha256'] for rows in source['assets'].values() for row in rows}
        binding=sha256(json_bytes(dict(source=source['recording_id'],snapshot=source['snapshot_sha256'],
            calibration=source['calibration_sha256'],collector=source['collector_instance_id'],
            topologies=source['topologies'],clocks=plan['allowed_clock_generations'],assets=assets,
            config=config_sha,scope=scope))).hexdigest()
        for i,lo in enumerate(range(start,stop,width)):
            hi=min(lo+width,stop)
            label=f'{i:05}'
            result.append(dict(key=f"{source['recording_id']}/{label}",label=label,index=i,
                recording_id=source['recording_id'],parent=source['root_episode_id'],split=source['split'],
                start_ns=lo,stop_ns=hi,tail=hi-lo<width,scope_digest=binding,
                tail_inheritance_max_ns=plan['tail_inheritance_max_ns'],
                planned_ticks=(hi-lo+19_999_999)//20_000_000))
    return result


def _footprint(part):
    return sum(p.stat().st_size for p in part.rglob('*') if p.is_file())


def _temp_footprint(part):
    return sum(_footprint(p) for p in part.parent.glob(f'.{part.name}.*.partial'))


def _host_io():
    return {n:int(v) for n,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())
            if n in {'rchar','wchar','read_bytes','write_bytes'}}


def _publish_index(output,index):
    target=output/'build_index.json'
    if target.exists():
        if read(target)!=index:
            raise ValueError('Existing terminal index differs from ledger')
        return
    candidate_path=output/f'.build_index.{os.getpid()}.partial'
    _write_fsynced(candidate_path,json_bytes(index))
    candidate_path.rename(target)
    _fsync_dir(output)


def _summarize(part):
    modes={}
    for mode in ('causal','offline'):
        arrays,meta,sources=read_partition(part,mode)
        masks=arrays['joint_valid']
        modes[mode]=dict(ticks=len(masks),joint_valid=int(masks.sum()),joint_invalid=int((~masks).sum()),
                         velocity_valid_by_stream=arrays['velocity_valid'].sum(axis=0).tolist(),
                         segment_lengths_ticks=[hi-lo for lo,hi in meta['segments'][mode]],
                         safety_inactive=int((~arrays['safety_permission']).sum()),
                         source_record_count=len(sources))
    return modes


def verify_existing(part,block,config_sha,scope,entry):
    manifest=validate_partition_pair(part)
    meta=read(part/'metadata.json')
    identity=meta['identity']
    if (digest(part/'manifest.json')!=entry['manifest_sha256'] or
            identity['execution_candidate_sha256']!=config_sha or
            identity['publication_scope']!=scope or identity['root_episode_id']!=block['parent'] or
            identity['development_split']!=block['split'] or identity['segment_block']!=block['label'] or
            identity['raw_interval_realtime_ns']!=[block['start_ns'],block['stop_ns']] or
            manifest['identity_sha256']!=entry['identity_sha256']):
        raise ValueError('Existing partition differs from the authorized durable ledger')
    if _summarize(part)!=entry['counts']:
        raise ValueError('Recovered valid-frame counts differ from previous publication')


def stage_one(source,record,assets,block,config,proposal,scope,plan,budget,matrix,writer,sdk,output,
              inherited=None,inject=False):
    current=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    try:
        capacity=resources(config['plan'],budget['per_chunk_published_cap_bytes'])
    except ValueError as exc:
        raise ResourcePause(str(exc)) from exc
    if (capacity['ram_available_bytes']<budget['minimum_available_ram_bytes'] or
            current>budget['minimum_available_ram_bytes'] or
            _footprint(output)>budget['required_peak_including_staging_bytes']):
        raise ResourcePause('Approved RSS, remaining RAM or staging/reserve condition reached')
    started=perf_counter()
    before=_host_io()
    interval=(block['start_ns'],block['stop_ns'])
    try:
        windows,w=read_recording(record,assets,matrix,writer,sdk,config['plan'],{block['label']:interval},
                                 production=True)
    except InsufficientSourceEvidence as exc:
        return None,dict(classification='INSUFFICIENT',reason='MISSING_STREAM_WITNESS',detail=str(exc))
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        return None,dict(classification='HARD_FAILURE',reason='RAW_SOURCE_CLOCK_OR_DECODE_PROOF_FAILED',detail=str(exc))
    q=qualify_partition(source,w,block,config['plan'],plan['allowed_clock_generations'],inherited)
    details=asdict(q)
    details.pop('status')
    details['classification']=q.status
    details['qualifier_reads']=w['source_read']
    details['calibration_max']=[
        {k:row[k] for k in ('side','matched','translation_m','rotation_rad')}
        for row in w['production_witness']['body_calibration'][block['label']]]
    if not q.eligible:
        return None,details
    item=windows[0]
    if any(len(t['joints'])>budget['max_topology_nodes'] for t in item.topologies):
        return None,dict(classification='HARD_FAILURE',reason='TOPOLOGY_EXCEEDS_APPROVED_NODE_BUDGET')
    quat=item.bind_quaternion_qc(config['numerical_qc']['quaternion_tolerance'])
    t0=perf_counter()
    causal,offline,alignment,_=item.replay(config['parameters'])
    sync_seconds=perf_counter()-t0
    if len(causal)!=block['planned_ticks'] or len(offline)!=block['planned_ticks']:
        raise ValueError('Planned and replayed ticks do not conserve the raw interval')
    args=publish_args(item,(causal,offline,alignment),w,config['parameters'],source['split'],
                      config['plan']['chosen_segment_design_ticks'])
    args['identity'].update(publication_scope=scope,execution_candidate_sha256=proposal,
        publication_validation=dict(asset_sha256_by_part={r['id']:r['sha256'] for v in source['assets'].values() for r in v},
           clock_generations=plan['allowed_clock_generations'],calibration_sha256=source['calibration_sha256'],
           min_segment_ticks=config['plan']['chosen_segment_design_ticks'],
           quaternion_tolerance=config['numerical_qc']['quaternion_tolerance']))
    dest=output/source['split']/source['root_episode_id'].replace(':','_')/block['label']
    result,written=write_partition(dest,**args,fail_before_publish=inject,
                    max_partition_bytes=budget['per_chunk_published_cap_bytes'])
    after=_host_io()
    cdr_seconds=sum(row['cdr_and_extract_seconds'] for row in w['source_read'].values())
    cdr_count=sum(row['cdr_messages'] for row in w['source_read'].values())
    retained=dict(unsafe_ticks=sum(x['safety_reason']=='UNSAFE' for x in causal),
                  held_pose_stream_ticks=sum(r['reason']=='HELD' for x in causal for r in x['streams'].values()),
                  invalid_pose_stream_ticks=sum(not r['valid'] for x in causal for r in x['streams'].values()),
                  stale_pose_stream_ticks=sum(r['reason']=='STALE' for x in causal for r in x['streams'].values()),
                  flagged_quaternions=quat)
    return dest,dict(qualification=details,classification=q.status,recording_id=source['recording_id'],
        scope_digest=block['scope_digest'],full_block=not block['tail'],stop_ns=block['stop_ns'],
        path=str(dest.relative_to(output)),manifest_sha256=digest(dest/'manifest.json'),
        identity_sha256=result['identity_sha256'],counts=_summarize(dest),retained=retained,
        measured={'actual_file_bytes':written,'published_file_bytes':_footprint(dest),
                  'retained_partial_bytes':_temp_footprint(dest),'wall_seconds':perf_counter()-started,
                  'sync_seconds':sync_seconds,'cdr_seconds':cdr_seconds,'cdr_messages':cdr_count,
                  'indexed_asset_bytes':sum(r['asset_bytes'] for r in w['source_read'].values()),
                  'linux_proc_io_delta':{k:after[k]-before[k] for k in before},
                  'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'disk_available_bytes':shutil.disk_usage(output).free})


def execute(args,config,scope,source_pool,plan,session):
    paths,_=registry(args.old_a)
    matrix,writer=hints()
    sdk=CalendarDecoder(SDK)
    sha=digest(args.config)
    blocks=plan_blocks(source_pool,plan,sha,scope)
    budget=read(D_DIR/'resource_budget.json')['production']
    output=args.output
    if not blocks:
        raise ValueError('No authorized source blocks')
    if scope==E1:
        resources(config['plan'],budget['required_peak_including_staging_bytes'])
    output.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (output/'.run.lock').open('ab') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_locked(args,config,scope,source_pool,plan,session,paths,matrix,writer,sdk,sha,blocks,budget)


def run_locked(args,config,scope,source_pool,plan,session,paths,matrix,writer,sdk,sha,blocks,budget):
    output=args.output
    by_source={r['recording_id']:r for r in source_pool}
    ledger=Ledger(output/'build_log.jsonl',scope,blocks)
    verified_again=[]
    full_sources={}
    for block in blocks:
        k=block['key']
        previous=ledger.latest[k]['status']
        source=by_source[block['recording_id']]
        row=paths[source['recording_id']]
        record_assets=asset_records(paths,row)
        dest=output/source['split']/source['root_episode_id'].replace(':','_')/block['label']
        if previous=='PUBLISHED':
            entry=ledger.publications[k]
            try:
                verify_existing(dest,block,sha,scope,entry)
            except (ValueError,OSError) as exc:
                _write_fsynced(output/f'.audit_rejection_{session}_{block["label"]}.json',json_bytes(
                    dict(status='PAUSED',partition=block,reason='EXISTING_PUBLICATION_INVALID',detail=str(exc))))
                raise
            verified_again.append(k)
        elif previous=='QUARANTINED':
            continue
        else:
            if previous!='PLANNED':
                ledger.add(k,'RESUMED',prior=previous,session=session)
            ledger.add(k,'RUNNING',session=session,disk_available_bytes=shutil.disk_usage(output).free)
            try:
                actual,entry=stage_one(source,row,record_assets,block,config,sha,scope,plan,budget,matrix,writer,sdk,output,
                         inherited=full_sources.get(source['recording_id']),
                         inject=scope==E0 and args.inject_index==block['index'])
            except InjectedPublicationError as exc:
                ledger.add(k,'FAILED',reason='INJECTED_AFTER_SEMANTIC_STAGING',detail=str(exc),
                           target_exists=dest.exists(),retained_partial_bytes=_temp_footprint(dest))
                ledger.add(k,'PAUSED',reason='CONTROLLED_E0_INTERRUPTION',session=session)
                raise
            except ResourcePause as exc:
                ledger.add(k,'PAUSED',reason='RESOURCE_LIMIT',detail=str(exc),session=session)
                raise
            except Exception as exc:
                ledger.add(k,'FAILED',reason='UNEXPECTED_OUTPUT_OR_BUILD_ERROR',detail=str(exc),
                           target_exists=dest.exists(),retained_partial_bytes=_temp_footprint(dest))
                raise
            if actual is None:
                ledger.add(k,'QUARANTINED',**entry)
                continue
            ledger.add(k,'PUBLISHED',**entry)
        if not block['tail'] and entry['classification']=='ELIGIBLE':
            full_sources[source['recording_id']]=entry
    end=ledger.completed_index()
    _publish_index(output,dict(scope=scope,allowed_use=USE,execution_candidate_sha256=sha,
                               partitions=end,planned_ticks=sum(p['planned_ticks'] for p in blocks)))
    _write_fsynced(output/f'.session-{session}.json',json_bytes(dict(
        scope=scope,session=session,previous_publications_revalidated=verified_again,
        ledger_entries=ledger.entries,complete=True)))
    print(json.dumps(dict(scope=scope,partitions=len(end),published=sum(r['status']=='PUBLISHED' for r in end),
                          prior_revalidated=len(verified_again),planned_ticks=sum(x['planned_ticks'] for x in blocks))))


def main():
    p=ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    lock=sub.add_parser('prepare')
    lock.add_argument('--old-a',type=Path,required=True)
    lock.add_argument('--output',type=Path,required=True)
    for mode in ('probe','build'):
        run=sub.add_parser(mode)
        for arg in ('old-a','config','lock','output'):
            run.add_argument('--'+arg,type=Path,required=True)
        run.add_argument('--session',required=True,type=int)
        if mode=='probe':
            run.add_argument('--fixture',type=Path,required=True)
            run.add_argument('--inject-index',type=int)
    args=p.parse_args()
    if args.mode=='prepare':
        dest=args.output.resolve()
        if not dest.is_relative_to(RUN_BASE) or not dest.relative_to(RUN_BASE).parts[0].startswith('P01-E-E0-'):
            raise ValueError('Use an isolated E0 candidate directory')
        cfg=audited_d()
        evidence=lock_inputs(args.old_a,cfg)
        dest.mkdir(mode=0o700)
        _write_fsynced(dest/'execution_lock.json',json_bytes(evidence))
        _write_fsynced(dest/'candidate_config.json',json_bytes(candidate(cfg,digest(dest/'execution_lock.json'))))
        return
    scope=E0 if args.mode=='probe' else E1
    authorize(scope,args.output,digest(args.config),digest(args.lock))
    cfg=verify_lock(args.old_a,args.config,args.lock)
    e0=read(PLAN)
    pool=read(D_DIR/'source_whitelist.json')['sources']
    if args.mode=='probe':
        expected_fixture=args.output.resolve().parent.parent/'fixture_plan.json'
        if args.fixture.resolve()!=expected_fixture:
            raise ValueError('E0 requires the recorded preregistered run fixture')
        fixture=read(args.fixture)
        pool=[row for row in pool if row['recording_id']==fixture['recording_id'] and
              row['root_episode_id']==fixture['root_episode_id'] and row['split']==e0['split'] and
              row['recorded_interval_ns']==[fixture['start_ns'],fixture['stop_ns']]]
        if len(pool)!=1:
            raise ValueError('E0 may probe only the authorized short-tail fixture')
    execute(args,cfg,scope,pool,e0,args.session)


if __name__=='__main__':
    main()
