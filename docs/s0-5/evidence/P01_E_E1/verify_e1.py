"""Independent E1 readback, disposition accounting and real MCAP witnesses."""
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np
from mcap.reader import NonSeekingReader

from dexgym.tasks.direct.wholebody_s05.data.geometry import hand_geometry, pose
from dexgym.tasks.direct.wholebody_s05.data.hmb_build import SCHEMA, SYNC_HASH, USE, json_bytes, read_partition
from dexgym.tasks.direct.wholebody_s05.data.p01e_build import E1
from dexgym.tasks.direct.wholebody_s05.data.p01e_ledger import Ledger
from dexgym.tasks.direct.wholebody_s05.data.p01e_lock import D_DIR, verify_lock
from dexgym.tasks.direct.wholebody_s05.data.raw_events import manus_geometry, message_fields
from dexgym.tasks.direct.wholebody_s05.data.registry_gate import digest, load_rows
from dexgym.tasks.direct.wholebody_s05.data.schema_registry import EmbeddedDecoders
from dexgym.tasks.direct.wholebody_s05.data.synchronize import SAFETY, STREAMS


metrics_spec=spec_from_file_location('audit_metrics',Path(__file__).with_name('audit_metrics.py'))
metrics=module_from_spec(metrics_spec)
metrics_spec.loader.exec_module(metrics)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    with path.open('xb') as f:
        f.write(json_bytes(obj))


class Audit:
    def __init__(self):
        self.groups=defaultdict(Counter)
        self.dist=defaultdict(metrics.Distributions)
        self.witnesses={}
        self.partitions=[]
        self.files={}

    def tally(self, tags, values, dist):
        for name in tags:
            self.groups[name].update(values)
            self.dist[name].extend(dist)


def evidence_one(sources,data,indices,seen,scenario,path,block):
    if scenario in seen:
        return
    wanted={}
    for i in indices:
        for j,stream in enumerate(STREAMS):
            key=int(data['source_index'][i,j,0])
            if key<0:
                continue
            row=sources[key]
            expected=None
            if data['pose_valid'][i,j]:
                field='sdk_common_pose' if stream.startswith('manus_') else 'pose'
                expected=data[field][i,j,data['node_mask'][i,j]].tolist()
            wanted.setdefault(key,dict(tick=i,stream=stream,source=row,expected=expected))
        key=int(data['safety_source_index'][i])
        if key>=0:
            wanted.setdefault(key,dict(tick=i,stream=SAFETY,source=sources[key],expected=None))
    streams={r['stream'] for r in wanted.values()}
    if scenario.startswith(('normal','tail','recovery')) and streams!=set((*STREAMS,SAFETY)):
        return
    if scenario=='invalid' and not any(r['source']['safe'] is False or not r['source']['raw_valid']
                                       for r in wanted.values()):
        return
    seen[scenario]=dict(path=path,key=block['key'],rows=list(wanted.values()))


def choose_witnesses(audit, sources, data, block, relative):
    good=np.flatnonzero(data['joint_valid'] & data['pose_valid'].all(axis=1))
    if len(good):
        evidence_one(sources,data,[int(good[len(good)//2])],audit.witnesses,'normal_'+block['split'],relative,block)
        if block['tail']:
            evidence_one(sources,data,[int(good[len(good)//2])],audit.witnesses,'tail',relative,block)
    unsafe=[]
    for i,idx in enumerate(data['safety_source_index']):
        if idx>=0 and (sources[int(idx)]['safe'] is False or not sources[int(idx)]['raw_valid']):
            unsafe.append(i)
    if unsafe:
        i=unsafe[0]
        evidence_one(sources,data,[i],audit.witnesses,'invalid',relative,block)
        later=[int(k) for k in good if k>i]
        if later:
            evidence_one(sources,data,later[:1],audit.witnesses,'recovery',relative,block)
    prior={}
    for record in sources:
        stream=record['stream']
        name='repeated_'+stream
        if not stream.startswith('manus_') or name in audit.witnesses:
            continue
        old=prior.get(stream)
        prior[stream]=record
        if old is None or not old['raw_valid'] or not record['raw_valid']:
            continue
        a,b=old['identity'],record['identity']
        stamp=(b['sdk_timestamp_raw'],b['sdk_receive_raw'])
        if (None not in stamp and stamp==(a['sdk_timestamp_raw'],a['sdk_receive_raw'])
                and a['instance']==b['instance'] and a['writer_sequence']+1==b['writer_sequence']
                and block['start_ns']<=record['raw_clock']['collector_realtime']['value']<block['stop_ns']):
            audit.witnesses[name]=dict(path=relative,key=block['key'],
                rows=[dict(tick=None,stream=stream,source=w,expected=None) for w in (old,record)])


def raw_readback(audit, registry):
    requests=defaultdict(set)
    for case in audit.witnesses.values():
        for item in case['rows']:
            ref=item['source']['record']
            requests[ref['part_id']].update((ref['physical_message'],ref['observation_message']))
    actual={}
    for part, requested in requests.items():
        asset=registry[part]
        path=Path(asset['canonical_uri'])
        check(digest(path)==asset['file_sha256'], f'Raw witness MCAP differs: {part}')
        rows={}
        with path.open('rb') as f:
            reader=NonSeekingReader(f,validate_crcs=True,record_size_limit=32*2**20)
            for index,payload in enumerate(reader.iter_messages(log_time_order=False)):
                if index in requested:
                    rows[index]=payload
                if len(rows)==len(requested):
                    break
        check(set(rows)==requested,f'Raw witness missing physical record: {part}')
        actual[part]=rows
    cases={}
    decoder=EmbeddedDecoders()
    for name,case in sorted(audit.witnesses.items()):
        observations=[]
        repeated=[]
        for witness in case['rows']:
            row, stream=witness['source'],witness['stream']
            ref=row['record']
            part=ref['part_id']
            physical, oi=ref['physical_message'],ref['observation_message']
            sch,ch,msg=actual[part][physical]
            osch,och,obs=actual[part][oi]
            check(digest_bytes(sch.data)==ref['schema_sha256'] and digest_bytes(msg.data)==ref['cdr_sha256']
                  and digest_bytes(osch.data)==ref['observation_schema_sha256']
                  and digest_bytes(obs.data)==ref['observation_cdr_sha256'] and ch.topic==ref['topic'],
                  f'Raw CDR/schema/topic mismatch at {part}:{physical}')
            fields,nonfinite=message_fields(decoder.decode(ch.message_encoding,sch,msg.data))
            o,_=message_fields(decoder.decode(och.message_encoding,osch,obs.data))
            check(oi==physical+1 and o['stream_id']==stream and o['source_instance_id']==ref['source_instance']
                  and o['source_sequence_id']==ref['writer_sequence'] and
                  o['host_realtime_ns']==msg.log_time==obs.log_time==row['raw_clock']['collector_realtime']['value']
                  and o['host_observed_monotonic_ns']==row['raw_clock']['collector_monotonic']['value'],
                  f'Observation identity/time mismatch at {part}:{physical}')
            expected=witness['expected']
            geometry='NOT_USED_FOR_INVALID_OR_SAFETY'
            if expected is not None:
                if stream.startswith('manus_'):
                    side=stream.split('_')[1]
                    raw=manus_geometry(fields,side)
                    check(raw['topology_sha256']==row['topology_id'] and not raw['issues'],'Witness topology mismatch')
                    measured=hand_geometry(raw['nodes'],side,raw['raw_side'])[0]
                else:
                    measured=pose(fields['pose'])[None,:]
                check(not nonfinite and np.array_equal(measured,np.asarray(expected)),
                      f'Published geometry does not match raw payload at {part}:{physical}')
                geometry='RAW_VALUE_EXACT'
            elif name.startswith('repeated_'):
                side=stream.split('_')[1]
                raw=manus_geometry(fields,side)
                check(raw['topology_sha256']==row['topology_id'] and not raw['issues'] and not nonfinite,
                      'Repeat witness topology or pose mismatch')
                measured=hand_geometry(raw['nodes'],side,raw['raw_side'])[0]
                repeated.append(digest_bytes(measured.tobytes()))
                geometry='RAW_REPEAT_PAIR_MEMBER'
            if stream==SAFETY:
                check(fields['safe'] is row['safe'],'Safety witness disagrees with record')
            if stream in ('tracker_left','tracker_right') and row['raw_valid']:
                check(fields['valid'] is True,'Registered tracker witness validity mismatch')
            observations.append(dict(stream=stream,tick_index=witness['tick'],status='PASS',geometry=geometry,
                                     raw_valid=row['raw_valid'],safe=row['safe'],record=ref,
                                     observation_topic=och.topic, sample_index=o['episode_sample_index']))
        cases[name]=dict(partition=case['key'],path=case['path'],matched=observations)
        if repeated:
            cases[name]['raw_geometry_sha256']=repeated
            cases[name]['equal_raw_geometry']=len(set(repeated))==1
    required=('normal_Train','normal_Validation','invalid','recovery','tail')
    return dict(cases=cases,missing_scenarios=sorted(set(required)-set(cases)),
                raw_files=len(requests),decoded_pairs=sum(len(a['matched']) for a in cases.values()))


def digest_bytes(data):
    return sha256(data).hexdigest()


def partition_report(audit, entry, source, products, cfg, by_key, idx, budget):
    block, state, e=entry['partition'],entry['status'],entry['evidence']
    tags=('All',block['split'],'root:'+block['parent'],'task:'+str(source['task']),
          'config:'+str(source['recorder_config_sha256']))
    ticks=block['planned_ticks']
    a=Counter(planned_ticks=ticks,duration_ns=block['stop_ns']-block['start_ns'],partitions=1)
    classification=e['classification']
    a[f'classification.{classification}']=1
    a[f'classified_ticks.{classification}']=ticks
    part=dict(key=block['key'],root=block['parent'],split=block['split'],index=block['index'],
              planned_ticks=ticks,status=state,classification=classification,
              start_ns=block['start_ns'],stop_ns=block['stop_ns'],scope_digest=block['scope_digest'])
    dist=metrics.Distributions()
    if state=='QUARANTINED':
        a['quarantined_ticks']=ticks
        a['quarantined_duration_ns']=a['duration_ns']
        part['evidence']=e
        audit.tally(tags,a,dist)
        audit.partitions.append(part)
        return
    check(state=='PUBLISHED' and classification in {'ELIGIBLE','TAIL_INHERITED'},'Invalid published qualification')
    if classification=='TAIL_INHERITED':
        q=e['qualification']
        candidates=[r for r in by_key[block['parent']] if r['evidence'].get('manifest_sha256')==q['inherited_from']]
        check(len(candidates)==1,'Inherited partition has no unique antecedent')
        antecedent=candidates[0]
        check(antecedent['evidence']['classification']=='ELIGIBLE' and antecedent['evidence']['full_block'] and
              antecedent['partition']['stop_ns']<=block['start_ns'] and
              antecedent['partition']['scope_digest']==block['scope_digest'] and block['tail'],
              'Short-tail inheritance binding failed')
        part['tail_inheritance']=dict(antecedent=antecedent['partition']['key'],qualification=q,
                                       manifest_sha256=q['inherited_from'])
    relative=Path(e['path'])
    target=products/relative
    check(target.resolve().is_relative_to(products.resolve()) and
          relative==Path(block['split'])/block['parent'].replace(':','_')/block['label'],
          'Published partition is outside its planned root')
    manifest=read(target/'manifest.json')
    check(digest(target/'manifest.json')==e['manifest_sha256'] and manifest['identity_sha256']==e['identity_sha256'],
          f'Build index manifest SHA mismatch: {block["key"]}')
    a['published_ticks']=ticks
    causal,meta,sources=read_partition(target,'causal')
    offline,other,_=read_partition(target,'offline')
    identity=meta['identity']
    check(meta==other and meta['schema']==SCHEMA and meta['sync_source_sha256']==SYNC_HASH
          and identity['publication_scope']==E1 and identity['execution_candidate_sha256']==idx['execution_candidate_sha256']
          and identity['allowed_use']==USE and identity['verification']=='SIMULATED'
          and identity['development_split']==block['split'] and identity['root_episode_id']==block['parent']
          and identity['raw_interval_realtime_ns']==[block['start_ns'],block['stop_ns']]
          and identity['parameters']==cfg['parameters'] and len(causal['tick_ns'])==ticks
          and np.array_equal(causal['tick_ns'],offline['tick_ns']),'Wrong scope, interval or data schema')
    for mode,data in (('causal',causal),('offline',offline)):
        values=metrics.mode_metrics(data,meta,mode,dist,cfg['numerical_qc'],cfg['plan'])
        measured=e['counts'][mode]
        check(values['joint_valid']==measured['joint_valid'] and values['joint_invalid']==measured['joint_invalid']
              and sum(measured['segment_lengths_ticks'])==values['segment_ticks']
              and all(x>=cfg['plan']['chosen_segment_design_ticks'] for x in measured['segment_lengths_ticks'])
              and values['joint_valid']+values['joint_invalid']==ticks,'Independent mask accounting mismatch')
        a.update({f'{mode}.{key}':v for key,v in values.items()})
        if mode=='offline':
            check(values['static_ticks']+values['dynamic_ticks']+values['unknown_motion_ticks']==ticks,
                  'Offline diagnostic motion labels do not conserve ticks')
    metrics.pair_metrics(causal,offline,meta,dist)
    a.update(metrics.source_metrics(sources,block,dist))
    a.update({f'retained.{k}':v for k,v in e['retained'].items()})
    choose_witnesses(audit,sources,causal,block,str(relative))
    part.update(counters=a,qc=dist.summary(), manifest_sha256=e['manifest_sha256'],
                max_calibration_residual=e['qualification']['calibration_max'])
    audit.tally(tags,a,dist)
    audit.partitions.append(part)
    audit.files.update({str(relative/name):value for name,value in manifest['files'].items()})
    audit.files[str(relative/'manifest.json')]=e['manifest_sha256']
    check(e['measured']['published_file_bytes']<=budget['per_chunk_published_cap_bytes'],'Partition exceeds frozen bound')


def group_reports(audit, plan):
    result={}
    for name,a in sorted(audit.groups.items()):
        check(a['planned_ticks']==a['published_ticks']+a['quarantined_ticks'],f'Disposition does not conserve ticks: {name}')
        for mode in ('causal','offline'):
            check(a[f'{mode}.joint_valid']+a[f'{mode}.joint_invalid']==a['published_ticks'],
                  f'Product accounting nonconservation: {name}/{mode}')
        sec=a['published_ticks']*0.02
        body_duration=(a['duration_ns']-a['quarantined_duration_ns'])/1e9
        frequency={}
        if body_duration:
            for stream in (*STREAMS,SAFETY):
                frequency[stream]=dict(publications_per_sec=a[f'{stream}.body_publications']/body_duration)
                if stream.startswith('manus_'):
                    frequency[stream]['sdk_pair_candidates_per_sec']=a[f'{stream}.independent_sdk_pair_candidates']/body_duration
        matching=[r for r in plan['sources'] if name=='All' or name==r['split'] or name=='root:'+r['root']
                   or name=='task:'+str(r['task']) or name=='config:'+str(r['recorder_config_sha256'])]
        result[name]=dict(counts=a,source_count=len(matching),scheduled_published_grid_seconds=sec,
                          published_raw_duration_seconds=body_duration,frequency=frequency)
    return result


def review(run, output, old_a, config, lock):
    cfg=verify_lock(old_a,config,lock)
    whitelist={r['recording_id']:r for r in read(D_DIR/'source_whitelist.json')['sources']}
    plan=read(run/'production_plan.json')
    products=run/'products'
    idx=read(products/'build_index.json')
    log=products/'build_log.jsonl'
    check(idx['scope']==E1 and idx['allowed_use']==USE and idx['execution_candidate_sha256']==digest(config)
          and idx['planned_ticks']==plan['total']['planned_ticks'],'Final index scope or planned count differs')
    journal=[json.loads(line) for line in log.open()]
    check({x['key'] for x in journal}=={x['key'] for x in plan['partitions']},'Incomplete ledger coverage')
    states=Ledger(log,E1,plan['partitions'])
    entries=states.completed_index()
    check(entries==idx['partitions'],'Final index differs from durable ledger')
    check([x['partition'] for x in entries]==plan['partitions'],'Partition identity/conservation differs')
    budget=read(D_DIR/'resource_budget.json')['production']
    by_parent=defaultdict(list)
    for row in entries:
        by_parent[row['partition']['parent']].append(row)
    audit=Audit()
    src={r['recording_id']:r for r in plan['sources']}
    for entry in entries:
        partition_report(audit,entry,src[entry['partition']['recording_id']],products,cfg,by_parent,idx,budget)
        if len(audit.partitions)%32==0:
            print(f'Independently checked {len(audit.partitions)}/{len(entries)} terminal partitions',flush=True)
    group=group_reports(audit,plan)
    roots={role:sorted(r['root'] for r in plan['sources'] if r['split']==role) for role in ('Train','Validation')}
    check(not set(roots['Train'])&set(roots['Validation']) and set().union(*map(set,roots.values()))==set(by_parent),
          'Split isolation or candidate dispositions differ')
    scope_manifest_paths={str(p.relative_to(products)) for p in products.glob('*/*/[0-9]*/manifest.json')}
    check(scope_manifest_paths=={f for f in audit.files if f.endswith('/manifest.json')},
          'Unexpected visible manifest in output scope')
    for split,expected in plan['source_totals'].items():
        group_count=group[split]['counts']
        check(all(expected[k]==group_count[k] for k in ('planned_ticks','duration_ns','partitions')),
              f'Split plan counts differ: {split}')
    raw=raw_readback(audit,{r['id']:r for r in load_rows(old_a/'registry.jsonl')})
    hard=[r['key'] for r in audit.partitions if r['classification']=='HARD_FAILURE']
    insufficient=[r['key'] for r in audit.partitions if r['classification']=='INSUFFICIENT']
    gate='FAIL' if hard else 'BLOCKED' if insufficient or raw['missing_scenarios'] else 'PASS'
    measured=[r['evidence']['measured'] for r in entries if r['status']=='PUBLISHED']
    step_stats=Counter(x['status'] for x in journal)
    resources=dict(sessions=step_stats,peak_builder_rss_bytes=max(x['peak_process_rss_bytes'] for x in measured),
                   min_free_disk_bytes=min(x['disk_available_bytes'] for x in measured),
                   logical_read_bytes=sum(x['linux_proc_io_delta']['rchar'] for x in measured),
                   physical_read_bytes_linux=sum(x['linux_proc_io_delta']['read_bytes'] for x in measured),
                   logical_write_bytes=sum(x['linux_proc_io_delta']['wchar'] for x in measured),
                   cdr_count=sum(x['cdr_messages'] for x in measured),cdr_s=sum(x['cdr_seconds'] for x in measured),
                   sync_s=sum(x['sync_seconds'] for x in measured),published_bytes=sum(x['published_file_bytes'] for x in measured),
                   rehash_and_build_wall_s=sum(x['wall_seconds'] for x in measured),
                   max_partition_bytes=max(x['published_file_bytes'] for x in measured),
                   partial_paths=[str(p.relative_to(products)) for p in products.rglob('*.partial')],
                   budget=budget)
    arch={str(p.relative_to(products)):p.stat().st_size for p in products.rglob('*') if p.is_file()}
    resources['total_archive_bytes']=sum(arch.values())
    resources['remaining_partial_bytes']=sum(n for f,n in arch.items() if '.partial/' in f or f.endswith('.partial'))
    diagnostics={k:v.summary() for k,v in audit.dist.items()}
    source_disposition=[]
    for r in plan['sources']:
        members=whitelist[r['recording_id']]['assets']
        raw_by_stream={name:dict(files=len(rows),bytes=sum(x['bytes'] for x in rows))
                       for name,rows in members.items()}
        source_disposition.append(dict(**r,raw_mcap_by_stream=raw_by_stream,
            disposition={x['partition']['key']:x['status'] for x in by_parent[r['root']]},
            counts=group['root:'+r['root']]['counts']))
    audit.outputs=dict(
        source_disposition=source_disposition,
        partition_disposition=audit.partitions,full_coverage=group,
        full_qc=dict(diagnostic_only=True,hard_partitions=hard,insufficient_partitions=insufficient,
                     velocity_flags_drop_samples=False,diagnostics=diagnostics),
        segment_statistics=dict(criterion_min_ticks=cfg['plan']['chosen_segment_design_ticks'],
                   groups={k:{'counts':{n:c for n,c in v['counts'].items() if 'segment' in n},
                              'lengths':{n:d for n,d in diagnostics.get(k,{}).items() if 'segment_length' in n}}
                           for k,v in group.items()}),
        split_isolation=dict(roots=roots,overlap=0,Final_Test='BLOCKED',passed=True),
        runtime_resources=resources,raw_traceability=raw)
    for name,obj in audit.outputs.items():
        write(output/(name+'.json'),obj)
    check(set(audit.files)<=set(arch),'Manifest references absent published data')
    hashes=dict(schema='s05.p01e.e1_output_hashes.v1',root=str(run),files={
        **{f'products/{name}':h for name,h in sorted(audit.files.items())},
        **{f'products/{name}':digest(products/name) for name in ('build_index.json','build_log.jsonl')},
        'production_plan.json':digest(run/'production_plan.json')})
    write(output/'output_sha256.json',hashes)
    manifest=dict(schema=SCHEMA,scope=E1,allowed_use=USE,approval='AWAITING_E1_HUMAN_REVIEW',technical_gate=gate,
                  run_id=run.name,products=str(products),original_plan_sha256=hashes['files']['production_plan.json'],
                  execution_candidate_sha256=digest(config),execution_lock_sha256=digest(lock),sync_sha256=SYNC_HASH,
                  index_sha256=hashes['files']['products/build_index.json'], published_partitions=len(measured),
                  quarantined_partitions=len(hard)+len(insufficient),counts={k:group[k] for k in ('All','Train','Validation')},
                  arrays={'causal':'SIMULATED','offline':'OFFLINE_PROXY'},sealed_final_test='BLOCKED',
                  available_fields={'causal':'see frozen hmb_schema', 'offline':'diagnostic_non_actor'})
    write(output/'hmb_manifest.json',manifest)
    print(json.dumps(dict(gate=gate,planned=plan['total'],published_partitions=len(measured),hard=hard,
                          insufficient=insufficient,raw_witnesses=raw['decoded_pairs'],
                          missing_scenarios=raw['missing_scenarios'], resources=resources),sort_keys=True))
    return gate


if __name__=='__main__':
    if sys.argv[1]=='--smoke':
        run,old_a,config,lock=(Path(arg).resolve() for arg in sys.argv[2:])
        cfg=verify_lock(old_a,config,lock)
        budget=read(D_DIR/'resource_budget.json')['production']
        plan=read(run/'production_plan.json')
        sources={r['recording_id']:r for r in plan['sources']}
        products=run/'products'
        journal=(products/'build_log.jsonl').read_bytes()
        entries=[r for line in journal[:journal.rfind(b'\n')].splitlines()
                 if (r:=json.loads(line))['status']=='PUBLISHED']
        check(bool(entries),'No production partition available for read-only smoke')
        chosen=[entries[0]]
        tail=next((e for e in entries if e['partition']['tail'] and e['evidence']['classification']=='ELIGIBLE'),None)
        if tail:
            chosen.append(tail)
        audit=Audit()
        for entry in chosen:
            recording=entry['partition']['recording_id']
            partition_report(audit,entry,sources[recording],products,cfg,{},
                             {'execution_candidate_sha256':digest(config)},budget)
        raw=raw_readback(audit,{r['id']:r for r in load_rows(old_a/'registry.jsonl')})
        check('normal_'+chosen[0]['partition']['split'] in raw['cases'],'No real geometry readback')
        print(json.dumps(dict(partitions=len(chosen),raw_witnesses=raw['decoded_pairs'],
                              found=list(raw['cases'])),sort_keys=True))
    elif review(*(Path(arg).resolve() for arg in sys.argv[1:]))!='PASS':
        sys.exit(3)
