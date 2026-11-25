"""Recheck the scoped HMB raw-input dependency gate against immutable and real replay evidence."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from dexgym.tasks.direct.wholebody_s05.data.decode_audit import decode_fixture
from dexgym.tasks.direct.wholebody_s05.data.hmb_gate import dependency_index, evaluate
from dexgym.tasks.direct.wholebody_s05.data.registry_gate import digest, load_rows
from dexgym.tasks.direct.wholebody_s05.data.schema_registry import EmbeddedDecoders


ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_bytes())


def put(path,doc):
    with path.open('x',encoding='utf-8') as f:
        json.dump(doc,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def verify_outputs(folder,receipt):
    names=read(folder/'output_sha256.json')
    if names!=receipt['output_sha256'] or any(digest(folder/p)!=key for p,key in names.items()):
        raise ValueError(f'Historic products differ: {folder}')
    return len(names)


def verify_provenance(a, b, resource):
    pre=read(resource)['result']
    cfg=read(ROOT/'evidence/P01_A/host_registry_config.json')
    if not all(pre['imports'][m]['file']['sha256']==h for m,h in cfg['pinned_runtime_sha256'].items()):
        raise ValueError('Runtime module changed')
    if pre['foreign'] or pre['wrong_dexgym'] or not pre['output']['writable']:
        raise ValueError('Isolated runtime and writable output required')
    aa=read(ROOT/'evidence/P01_A/review_receipt.json')
    bb=read(ROOT/'evidence/P01_B/review_receipt.json')
    counts={'A':verify_outputs(a,aa),'B':verify_outputs(b/'products',bb)}
    data=ROOT.parents[1]/'source/dexgym/tasks/direct/wholebody_s05/data'
    for f,h in bb['module_sha256'].items():
        if digest(data/f)!=h:
            raise ValueError(f'Historic runtime module changed: {f}')
    for f,h in bb['local_diagnostic_file_sha256'].items():
        if digest(b.parent/f)!=h:
            raise ValueError(f'Historic diagnostic changed: {f}')
    return counts,len(bb['module_sha256']),len(bb['local_diagnostic_file_sha256']),bb


def verify_calibrations(fixtures, registry):
    proven={}
    for f in fixtures:
        row=registry[f['recording_id']]
        if row['status']!='ADMITTED_CANDIDATE':
            continue
        results=[]
        for ref in row.get('calibration_refs',()):
            p=Path(ref['path'])
            observed=digest(p) if p.is_file() else None
            results.append({'path':str(p),'expected_sha256':ref['sha256'],'observed_sha256':observed,
                            'matches':observed==ref['sha256']})
        if results:
            proven[f['recording_id']]=results
    return proven


def replay_real(a,registry,eligible,fixtures,source,matrix,calibration_ids):
    schema=read(a/'schema_registry.json')
    cfg=read(ROOT/'evidence/P01_B/sample_plan.json')
    cfg.update(native_per_channel=8)
    bank=EmbeddedDecoders()
    chosen=[registry[r['recording_id']] for r in fixtures if r['recording_id'] in eligible]
    assets={k:v for k,v in registry.items() if v['record_type']=='asset'}
    fresh_fixtures,fresh_events,results=[],[],[]
    for row in chosen:
        f,events,attempt=decode_fixture(row,['P01B_REVIEW_REAL_PREFIX'],assets,schema,cfg,bank,{})
        fresh_fixtures.append(f)
        fresh_events.extend(events)
        results.append(attempt)
    result=evaluate(fresh_fixtures,fresh_events,registry,source,matrix,calibration_ids)
    old={
        (e['recording_id'],e['index']['part_id'],e['index']['physical_message']):e
        for e in load_rows(a/'raw_events.jsonl') if e['recording_id'] in eligible
    }
    differences=[]
    for event in fresh_events:
        r=(event['recording_id'],event['index']['part_id'],event['index']['physical_message'])
        other=old.get(r,{})
        if any(event.get(key)!=other.get(key) for key in
               ('index','segment','cdr_sha256','decode_status','fields','clock','manus','pairing')):
            differences.append(r)
    process_issues=[i for r in results for i in r['issues']]
    result.update(event_count=len(fresh_events),lost_or_different_indexes=differences,
                  adapter_issues=process_issues,recordings=len(chosen))
    if differences or process_issues:
        result['technical_hmb_gate']='FAIL'
    return result


def catalogue_summary(catalogue,matrix):
    direct=dependency_index(matrix)
    watchers={t for rule in matrix['streams'] for t in rule.get('observation',{}).get('topics',[])}
    counts=Counter()
    per_channel={}
    for payload in catalogue['payloads'].values():
        for c in payload['channels']:
            t, schema = c['topic'],c.get('schema_key')
            if t in direct:
                label=direct[t]['class']
            elif t in watchers:
                label='HMB_CONDITIONAL'
            else:
                label='OUT_OF_SCOPE'
            # Type identity is kept for all declarations, including unknown versions.
            per_channel[t,schema]=label
            counts[label]+=1
    return {'per_topic_schema':[{'topic':t,'schema_key':s,'classification':k}
                                for (t,s),k in sorted(per_channel.items(),key=lambda item:str(item[0]))],
            'catalogue_channel_occurrences_by_class':dict(counts)}


def inspect_telemetry(p,legacy,source):
    plan=read(ROOT/'evidence/P01_B/sample_plan.json')
    levels=read(p/'telemetry_eligibility.json')['levels']
    rows=load_rows(p/'telemetry_index.jsonl')
    check=Counter()
    for row in rows:
        if row['expected_dimension'] is not None:
            check['indexed_named_axis_rows']+=1
            check['missing_robot_feedback_clock']+=int(row['clock']['robot_feedback']['value'] is None)
            check['missing_command_issue_clock']+=int(row['clock']['command_issue']['value'] is None)
    source_support=Counter(item['scope'] for row in source.values() for item in row['supporting_files'])
    return {'B2':'BLOCKED','P01_T':'BLOCKED','known_historical_attestations':len(plan['attestations']),
            'views_lacking_hardware_provenance':levels.get('NO_HARDWARE_PROVENANCE',0),
            'authenticated_dynamic_candidates':levels.get('DYNAMIC_SYSTEM_ID_CANDIDATE',0),
            'telemetry_levels':levels,'source_support_files':dict(source_support),'clock_audit':dict(check),
            'new_affirmative_provenance':0,'collection_status_recovery_changes_eligibility':False,
            'review_legacy_recordings':[g['recording_id'] for g in legacy['affected_recordings']]}


def main():
    a,b,run=map(Path,sys.argv[1:])
    counts,mods,logs,history=verify_provenance(a,b,run/'resource_preflight.json')
    matrix_path=ROOT/'evidence/P01_B_review/topic_dependency_matrix.json'
    matrix=read(matrix_path)
    p=b/'products'
    registry={r['id']:r for r in load_rows(a/'registry.jsonl')}
    source={r['recording_id']:r for r in load_rows(p/'source_eligibility_updates.jsonl')}
    fixtures=read(p/'fixtures_manifest.json')['recordings']
    # All admitted witnesses in this review use the published independent native source proof.
    # Historical merged fixtures are diagnostic and no proof profile has been authorized for them.
    calibration=verify_calibrations(fixtures,registry)
    proven={k for k,v in calibration.items() if all(r['matches'] for r in v)}
    observed=evaluate(fixtures,load_rows(p/'raw_events.jsonl'),registry,source,matrix,proven)
    ids={f['recording_id'] for f in observed['fixtures'] if f['status']!='ISOLATED_SOURCE'}
    replay=replay_real(p,registry,ids,fixtures,source,matrix,proven)
    old=history['tech_gates']
    legacy=read(run/'legacy/collection_status_review.json')
    if legacy['original_failed_events']!=old['decode_attempt_counts']['CDR_DECODE_FAILED']:
        raise ValueError('Historical diagnostic denominator differs')
    selection={'gate_on_immutable_B':observed,'real_replay':replay,
               'calibration_witnesses':calibration,
               'technical_hmb_gate':('PASS' if observed['technical_hmb_gate']==replay['technical_hmb_gate']=='PASS'
                                     else 'FAIL' if 'FAIL' in (observed['technical_hmb_gate'],replay['technical_hmb_gate'])
                                     else 'BLOCKED'),
               'train_eligibility_grants':0,'collection_status_dependency':{
                   'admitted_fixtures':sorted(ids),
                   'dependencies_used':matrix['sidecar_requirements'],
                   'dependent_on_collection_status':[],
                   'merged_source_provenance':'UNVERIFIED_OR_EXCLUDED; check proof per future admission'}}
    telemetry=inspect_telemetry(p,legacy,source)
    catalogue=catalogue_summary(read(p/'schema_registry.json'),matrix)
    put(run/'hmb_gate_result.json',selection)
    put(run/'telemetry_review.json',telemetry)
    put(run/'topic_catalogue_classification.json',catalogue)
    transient={'output_sha256.json','review_receipt.json','run.log','process_sha256.json',
               'hmb.stdout','hmb.stderr','hmb.exit'}
    hashes={str(f.relative_to(run)):digest(f) for f in sorted(run.rglob('*'))
            if f.is_file() and f.name not in transient}
    put(run/'output_sha256.json',hashes)
    receipt={'schema':'s05.p01b.review-fix.v1','utc_offset_observed':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
             'reviewed_parent':matrix['assessment_baseline'],'run_id':run.name,'absolute_run':str(run),
             'original_archive_sha_checks':counts,'verified_historical_modules':mods,'verified_historical_diagnostics':logs,
             'dependency_matrix_sha256':digest(matrix_path),'output_sha256':hashes,
             'manifest_sha256':digest(run/'output_sha256.json'),'historic_B1':old['B1'],'historic_B2':old['B2'],
             'new_hmb_gate':selection['technical_hmb_gate'],'hmb_real_replay':replay,
             'hmb_fixed_witnesses':observed['fixture_counts'],
             'legacy_collection_status':legacy,'telemetry_review':telemetry,
             'manus_sdk_original_witness':history['sdk_components'],'training_admission_grants':0,
             'final_test':'BLOCKED'}
    put(run/'review_receipt.json',receipt)
    print(json.dumps({'HMB':selection['technical_hmb_gate'],'history':old['B1'],
                      'aux':legacy['auxiliary_review_outcomes'],'B2':'BLOCKED',
                      'fixed_witness_counts':observed['fixture_counts'],'fresh_events':replay['event_count']}))


if __name__=='__main__':
    main()
