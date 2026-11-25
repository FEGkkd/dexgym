"""Publish compact B review evidence, leaving the real samples on Ubuntu."""
from collections import Counter,defaultdict
from datetime import datetime
from hashlib import file_digest
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo


def doc(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def summarize(attempt):
    p = attempt/'products'
    hashes = doc(p/'output_sha256.json')
    if (set(hashes)!={f.name for f in p.iterdir()}-{'output_sha256.json'} or
            any(digest(p/name)!=key for name,key in hashes.items())):
        raise ValueError('Missing or changed P01-B output')
    contract,coverage,attempts = (doc(p/f'{name}.json') for name in
                                 ('adapter_contract','dataset_coverage','decode_attempts'))
    evidence = doc(p/'schema_registry.json')
    fixtures = doc(p/'fixtures_manifest.json')['recordings']
    history = doc(p/'lineage_followup.json')
    clocks = doc(p/'raw_timestamp_report.json')['rows']
    topology = doc(p/'manus_topology_report.json')['rows']
    source = rows(p/'source_eligibility_updates.jsonl')
    test = doc(attempt/'test_results.json')
    missing=doc(p/'missing_fields.json')['telemetry']
    telem = doc(p/'telemetry_eligibility.json')
    sdk = doc(attempt/'sdk_components.json')
    cdr_probe = doc(attempt/'collection_status_probe.json')
    types = Counter((x['format'],x['status']) for x in evidence['real_semantic_version_coverage'])
    by_fixture = {v['recording_id']:v for v in fixtures}
    clock_rows=[]
    for row in clocks:
        dimensions = {name:{k:v for k,v in s.items() if k in (
            'missing','repeat_steps','backward_steps','step_median_raw','large_forward_steps_diagnostic_only',
            'zero_values','unit')} for name,s in row['clock_statistics'].items()
                      if name in {'sdk_source','sdk_receive','ros_header','mcap_log','controller_receive'}}
        clock_rows.append(dict(recording_id=row['recording_id'],topic=row['topic'],
                               task=by_fixture[row['recording_id']]['task'],clocks=dimensions,
                               repeats=row['repeated_source_candidates']))
    cast = Counter(x['node_count'] for row in topology for x in row['known_topologies'].values())
    volumes = {name:(p/name).stat().st_size for name in hashes}
    tags = Counter(v['task'] or 'UNKNOWN' for v in fixtures)
    legacy = [f for f in fixtures if f['locator'].endswith('/teleop_no_camera_20261016_133930')][0]
    telemetry_preview=[]
    for row in rows(p/'telemetry_index.jsonl'):
        if row['recording_id']!=legacy['recording_id'] or row['expected_dimension'] is None:
            continue
        if any(x['role']==row['role'] for x in telemetry_preview):
            continue
        telemetry_preview.append({k:row[k] for k in ('role','index','axes_match_recorded_schema',
                    'available_fields','effort_unit','expected_dimension','measured_or_command')}
                | {'vector_lengths':{k:v['length'] for k,v in row['field_vectors'].items()},
                   'joint_names':row['names']['value']})
    a1=contract['preflight_A1']
    a_input=contract['runtime']['result']
    return {
        'schema':'s05.p01b.review-receipt.v1',
        'observed':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'absolute_attempt':str(attempt),'parent_commit':a_input['repositories'][
            a_input['cwd']]['head']['stdout'].strip(),
        'frozen_p01_spec_sha256':contract['inherited_spec_sha256'],
        'output_sha256':hashes,'output_sha256_manifest':digest(p/'output_sha256.json'),
        'output_bytes':volumes,'run_scope':'B only, bounded diagnostics',
        'module_sha256':contract['modules'],
        'resource':{'python':a_input['python'],'executable':a_input['executable'],
                     'overlay_setup_files':a_input['overlay_setup_files'],
                     'distributions':a_input['distributions'],'free_bytes':a_input['output']['available_bytes']},
        'inherited_a_denominators':coverage['inherited_a_counts'],
        'independent_A1':a1,'tech_gates':contract['gates'],
        'catalogue':{'payloads':len(evidence['payloads']),'parts':len(evidence['part_map']),
                     'schemas_total':len(evidence['schemas']),'unavailable':attempts['catalogue_unavailable'],
                     'unknown_target_channels':attempts['unknown_target_channels'],
                     'semantic_versions_by_format_and_status':[
                          {'format':fmt,'status':st,'count':n} for (fmt,st),n in sorted(types.items())]},
        'typed_versions':evidence['real_semantic_version_coverage'],
        'decode_sample_counts':{'recordings':len(fixtures),'formats':dict(Counter(x['format'] for x in fixtures)),
                                'task_tags':dict(tags),
                                'scanned_parts':sum(len(x['parts']) for x in fixtures)},
        'actual_source_levels':coverage['source_levels'],'fixture_timezone':'Asia/Shanghai',
        'source_supporting_file_roles':dict(Counter(item['scope'] for row in source for item in row['supporting_files'])),
        'bounded_native_overlap_ns':coverage['bounded_native_overlap_ns'],
        'bounded_native_coverage':coverage['native_sample_coverage'],
        'telemetry_levels_by_recording_view':telem['levels'],'telemetry_missing_fields':missing,
        'telemetry_earliest_old_hand_and_arm_examples':telemetry_preview,
        'signature_followup':{
            'weak_count':len(history['weak_merged_signatures']),
            'weak_under_old_aliases':sum(bool(x['old_alias_of']) for x in history['weak_merged_signatures']),
            'complete_conflicts':len(history['complete_payload_conflicts']),
            'observed_payload_conflicts':len(history['equal_observed_payload_declaration_conflicts']),
            'unopenable_or_unbound_sidecars':len(history['inaccessible_sidecars']),
            'unresolved_semantic_matches':len(history['unresolved_semantic_matches']),
            'old_unresolved_prior_use_events':history['old_unresolved_prior_use_events']},
        'topology':{'seen_per_stream_examples':dict(cast),'events':sum(row['samples'] for row in topology),
                    'topology_changes':sum(len(row['topology_changes']) for row in topology),
                    'geometry_issue_counts':dict(Counter(i for row in topology
                                           for i,n in row['geometry_issue_counts'].items() for _ in range(n)))},
        'raw_manus_clocks':clock_rows,
        'legacy_collection_status_probe':cdr_probe,
        'sdk_components':{k:v for k,v in sdk.items() if k!='per_recording_and_side'} |
            {'calendar_count':sum(c['valid_calendar'] for c in sdk['per_recording_and_side']),
             'total_samples':sum(c['total'] for c in sdk['per_recording_and_side']),
             'timecode_true':sum(c['timecode_flag_true'] for c in sdk['per_recording_and_side']),
             'median_ms_values':sorted({c['calendar_step_p50_ms'] for c in sdk['per_recording_and_side']
                                        if c['calendar_step_p50_ms'] is not None})},
        'test_exits':{'unit':test['unit']['exit'],'decode':test['decode_exit'],'integration':test['integration']['exit']},
        'test_counts':{k:int(re.search(r'Ran (\d+) tests?',test[k]['stdout_stderr'])[1]) for k in ('unit','integration')},
        'local_diagnostic_file_sha256':{str(q.relative_to(attempt.parent)):digest(q) for q in (
            attempt/'sdk_components.json',attempt/'test_results.json',attempt/'decode.command',
            attempt/'timing_source.json',attempt/'collection_status_probe.json',
            attempt.parent/'runner01.log',attempt.parent/'runner02.log',attempt.parent/'runner03.log',
            attempt.parent/'runner04.log')},
        'final_test':coverage['final_test'],
    }


if __name__=='__main__':
    print(json.dumps(summarize(Path(sys.argv[1])),indent=2,sort_keys=True))
