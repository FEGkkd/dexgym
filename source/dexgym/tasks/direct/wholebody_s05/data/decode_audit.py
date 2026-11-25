"""P01-B bounded audit. Process success and B1/B2 decisions are independent."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

from .adapters.dex_native_v2 import decode_stream,stream_parts
from .adapters.ros2_merged_mcap import decode_part,open_asset
from .decode_inputs import (MANUS, WRISTS, TARGET_TOPICS, capture_notes, channels_for, native_manifest,
                            pack_catalogue, relevant, select_recordings, signature_followup)
from .decode_metrics import field_report,native_coverage,semantic_candidates
from .registry_gate import digest,inspect_snapshot,load_rows
from .schema_registry import EmbeddedDecoders
from .source_registry import publish,stable_json
from .telemetry import (eligibility,observed_roles,source_eligibility,telemetry_row,
                        validate_bindings,verify_attestation)


def read(path):
    return json.loads(path.read_bytes())


def verify_asset(asset):
    from hashlib import file_digest
    with open_asset(asset) as stream:
        actual = file_digest(stream,'sha256').hexdigest()
    if actual!=asset.get('file_sha256'):
        raise ValueError(f"Selected MCAP SHA mismatch: {asset['id']}")
    return actual


def decode_fixture(row, reasons, assets, registry, config, decoders, hash_cache):
    fixture = dict(recording_id=row['id'],root_episode_id=row['root_episode_id'],locator=row['locator'],
                   format=row['format'],a_status=row['status'],a_origin=row['producer_kind'],selection=reasons,
                   task=row.get('task'),date=row.get('capture_date'),profile=row['evidence'][-1].get('profile'),
                   record_limit_per_part=config['physical_record_limit'],
                   role='DEVELOPMENT_DIAGNOSTIC_ONLY',verified_writer=False,parts=[])
    events, errors = [], []
    parts = [assets[k] for k in row['source_asset_ids']]
    selected = []
    for asset in parts:
        channels = channels_for(registry,asset['id'])
        payloads = {r['topic'] for r in channels if r['topic'] in TARGET_TOPICS and r['declared_count']!=0}
        topics = {r['topic'] for r in channels if relevant(r['topic']) and r['declared_count']!=0}
        if row['format']=='dex-native-v2' and not payloads:
            # Never select a native writer merely because it carries SampleObservation.
            continue
        if not topics:
            continue
        cache_key = (asset['canonical_uri'],asset['file_sha256'])
        try:
            if cache_key not in hash_cache:
                hash_cache[cache_key] = verify_asset(asset)
        except (OSError,ValueError) as error:
            errors.append({'part_id':asset['id'],'status':'SOURCE_CHECKSUM_FAILED','detail':str(error)})
            continue
        fixture['parts'].append(
            {'part_id':asset['id'],'path':asset['canonical_uri'],'sha256':hash_cache[cache_key],
             'expected_channels':[r for r in channels if r['topic'] in topics]})
        selected.append((asset,topics))
    part_ids = {p['id'] for p,_ in selected}
    try:
        if row['format']=='dex-native-v2':
            manifest,writer_verified = native_manifest(row,config)
            fixture['verified_writer'] = writer_verified
            for group in stream_parts(manifest,parts,row).values():
                group['assets'] = [p for p in group['assets'] if p['id'] in part_ids]
                events.extend(decode_stream(group,manifest,writer_verified,decoders,config['native_per_channel'],
                                            config['physical_record_limit']))
        else:
            for asset,topics in selected:
                decoded = decode_part(asset,decoders,topics,config['merged_per_channel'],config['physical_record_limit'])
                events.extend(dict(r,recording_id=row['id'],segment='BODY') for r in decoded)
    except Exception as error:
        errors.append({'status':'ADAPTER_ERROR','detail':f'{type(error).__name__}: {error}'})
    sampled = defaultdict(list)
    for item in events:
        sampled[item['index']['part_id']].append(item['index'])
    for part in fixture['parts']:
        indexes = sampled[part['part_id']]
        part['actual_events'] = len(indexes)
        part['sampled_physical_record_interval_inclusive'] = (
            [indexes[0]['physical_message'],indexes[-1]['physical_message']] if indexes else None)
    indexed = Counter((x['index']['topic'],x['index']['schema_key'],x['decode_status']) for x in events)
    return fixture,events,dict(recording_id=row['id'],issues=errors,event_count=len(events),
                              results=[dict(topic=t,schema_key=k,status=status,count=v)
                                       for (t,k,status),v in sorted(indexed.items(),key=lambda v:str(v[0]))])


def source_updates(recordings, assets, rows_by_id, fixtures, followup, config):
    external = {r['recording_id']:r for r in config['attestations']}
    conflicts = followup['complete_payload_conflicts']+followup['equal_observed_payload_declaration_conflicts']
    excluded = {v for r in conflicts for v in r['record_ids']}
    excluded.update(r['recording_id'] for r in followup['weak_merged_signatures'])
    ids = {f['recording_id'] for f in fixtures}
    results, tech = [], []
    for row in recordings:
        rid = row['id']
        ordered_sha = [assets[k]['file_sha256'] for k in row['source_asset_ids']]
        proof = {} if rid in excluded else verify_attestation(row,external.get(rid),
                                                              config['attestation_roots'],ordered_sha)
        effective = dict(row,status='QUARANTINED') if rid in excluded and row['status']=='ADMITTED_CANDIDATE' else row
        result = source_eligibility(effective,proof)
        result.update(recording_id=rid,old_status=row['status'],diagnostic_sample=rid in ids,
                      source_check='PROOF_VERIFIED' if proof else 'NO_NEW_CONTEMPORANEOUS_PROOF',
                      supporting_files=capture_notes(row),
                      orig_source_qualification_from_a=row['root_episode_id'] is not None)
        results.append(result)
        telemetry = rows_by_id.get(rid,[])
        roles = observed_roles(telemetry)
        pair,clock = validate_bindings(rid,telemetry,proof)
        decision = eligibility(result['hardware_provenance'],roles,pair,clock)
        if result['level']=='EXCLUDED':
            decision['level'] = 'EXCLUDED_SOURCE'
        elif result['level']=='DUPLICATE_ALIAS':
            decision['level'] = 'DUPLICATE_ALIAS'
        elif not result['hardware_provenance'] and row['capabilities']['telemetry_status']=='NO_COMMAND_FEEDBACK_PAIR':
            decision['level'] = 'NO_COMMAND_FEEDBACK_PAIR'
        decision.update(recording_id=rid,a_topic_capability=row['capabilities']['telemetry_status'],
                        diagnostic_sample=rid in ids,source_level=result['level'])
        tech.append(decision)
    return results,tech


def schema_coverage(registry, recordings, events):
    roles = {r['id']:r['format'] for r in recordings}
    candidates = defaultdict(set)
    samples, failures = Counter(), Counter()
    for row in recordings:
        for part in row['source_asset_ids']:
            for channel in channels_for(registry,part):
                if relevant(channel['topic']) and channel['schema_key'] and channel['declared_count']!=0:
                    candidates[row['format'],channel['schema_key']].add(channel['topic'])
    for event in events:
        key = (roles[event['recording_id']],event['index']['schema_key'])
        if event['decode_status']=='PASS':
            samples[key] += 1
        else:
            failures[key] += 1
    out = []
    for (fmt,key),topics in sorted(candidates.items()):
        schema = registry['schemas'][key]
        if not schema['decoder']:
            state = 'UNSUPPORTED_SCHEMA'
        elif failures[fmt,key]:
            state = 'DECODE_FAILED'
        else:
            state = 'VERIFIED_IN_REAL_SAMPLE' if samples[fmt,key] else 'NOT_RUN'
        out.append(dict(format=fmt,schema_key=key,type=schema['name'],sha256=schema['sha256'],
                        topics=sorted(topics),sample_count=samples[fmt,key],
                        failed_count=failures[fmt,key],status=state))
    return out


def gates(issues, events, versions, fixtures, telemetry):
    statuses = Counter(e['decode_status'] for e in events)
    missing = [v for v in versions if v['status']!='VERIFIED_IN_REAL_SAMPLE']
    severe = [i for a in issues for i in a['issues']]
    structural = sum(bool(e.get('manus',{}).get('issues')) for e in events)
    pairs = Counter(e['pairing']['status'] for e in events if e.get('pairing'))
    native = {f['recording_id'] for f in fixtures if f['format']=='dex-native-v2' and f['verified_writer']}
    native_unverified = sum(e['recording_id'] in native and e.get('pairing',{}).get('status')=='UNVERIFIED'
                            for e in events)
    observed = {e['index']['topic'] for e in events if e['recording_id'] in native and
                e.get('pairing',{}).get('status')=='VERIFIED_WRITER_PAIR'}
    legacy_hand = any(v['type'].endswith('/HandState') and v['format']=='ros2-merged-mcap' and
                      v['status']=='VERIFIED_IN_REAL_SAMPLE' for v in versions)
    if severe or statuses['CDR_DECODE_FAILED'] or statuses['RAW_FIELD_CONTRACT_FAILED'] or structural:
        b1 = 'FAIL'
    elif missing or not native or not (MANUS|WRISTS)<=observed or not legacy_hand or native_unverified:
        b1 = 'BLOCKED'
    else:
        b1 = 'PASS'
    b2 = 'PASS' if any(t['dynamic_system_id_candidate'] for t in telemetry) else 'BLOCKED'
    return {'B1':b1,'B2':b2,'decode_attempt_counts':dict(statuses),'pairing_counts_by_event':dict(pairs),
            'missing_schema_versions':len(missing),'manus_structural_failures':structural,
            'unverified_events_in_attested_native':native_unverified,
            'adapter_issues':severe,'old_hand_state_sample_verified':legacy_hand,
            'timer_mapping':'NOT_RUN_P01_C'}


def run(args):
    cfg = read(args.config)
    baseline = read(args.a_receipt)
    old_summary = read(args.a_registry/'source_inventory.json')
    technical_a = inspect_snapshot(args.a_registry,baseline['output_sha256'])
    if technical_a['gate']!='PASS':
        raise ValueError(stable_json({'preflight_A1':technical_a,'B1':'BLOCKED','B2':'BLOCKED'}))
    rows = load_rows(args.a_registry/'registry.jsonl')
    assets = {r['id']:r for r in rows if r['record_type']=='asset'}
    recordings = [r for r in rows if r['record_type']=='recording']
    runtime = read(args.resource)
    host_cfg = read(args.a_config)
    env = runtime['result']
    if env['wrong_dexgym'] or env['foreign'] or not all(
        env['imports'][k]['file']['sha256']==h for k,h in host_cfg['pinned_runtime_sha256'].items()):
        raise ValueError('Runtime differs from the approved P01-A baseline')
    unit = read(args.unit)
    if unit['exit']!=0:
        raise ValueError('Unit tests must complete before bounded raw decode')
    technical_a['historical_gate_unchanged'] = old_summary['gate_A1']
    report = lambda *msg: print(*msg,file=sys.stderr,flush=True)
    registry = pack_catalogue(assets,lambda i,n: report('P01-B MCAP catalogue',i,n))
    followup = signature_followup(recordings,assets,registry,old_summary)
    fixtures, events, attempts = [], [], []
    decoders, cache = EmbeddedDecoders(), {}
    for row,reasons in select_recordings(recordings,registry,cfg):
        report('P01-B bounded decode',row['locator'])
        fixture,decoded,result = decode_fixture(row,reasons,assets,registry,cfg,decoders,cache)
        fixtures.append(fixture)
        events.extend(decoded)
        attempts.append(result)
    followup['unresolved_semantic_matches'] = semantic_candidates(events)
    telemetry = [row for e in events if (row:=telemetry_row(e)) is not None]
    by_record = defaultdict(list)
    for row in telemetry:
        by_record[row['recording_id']].append(row)
    source, policy = source_updates(recordings,assets,by_record,fixtures,followup,cfg)
    versions = schema_coverage(registry,recordings,events)
    field, timestamps, topology, absences = field_report(events,telemetry)
    gate = gates(attempts,events,versions,fixtures,policy)
    source_ids = {r['recording_id'] for r in source if r['raw_human_source_verified']}
    coverage = native_coverage(recordings,events,source_ids)
    old_coverage = read(args.a_registry/'dataset_coverage.json')
    schema_unavailable = sum(p['catalogue_state']!='CATALOGUED' for p in registry['payloads'].values())
    unknown_target_channels = sum(not c['schema_key'] and relevant(c['topic']) and c['declared_count']!=0
                                  for p in registry['payloads'].values() for c in p['channels'])
    if (schema_unavailable or unknown_target_channels) and gate['B1']=='PASS':
        gate['B1']='BLOCKED'
    products = {
        'schema_registry.json':dict(**registry,real_semantic_version_coverage=versions),
        'adapter_contract.json':dict(schema='s05.raw-decode.p01b.v1',config=cfg,
            inherited_spec_sha256=baseline['approved_spec_sha256'],runtime=runtime,
            preflight_A1=technical_a,gates=gate,
            modules={str(p.relative_to(Path(__file__).parent)):digest(p)
                     for p in Path(__file__).parent.rglob('*.py')},
            status_semantics={'process_exit':'audit execution','B1_B2':'technical decision',
                              'approval':'human stage decision'},
            physical_index='Zero-based message number in physical part; only bounded prefixes inspected',
            no_training_admission=True),
        'fixtures_manifest.json':dict(recordings=fixtures,protocol='P01-B bounded samples'),
        'decode_fields.json':dict(rows=field,version_coverage=versions),
        'decode_attempts.json':dict(aggregate=gate,attempts=attempts,catalogue_unavailable=schema_unavailable,
                                    unknown_target_channels=unknown_target_channels),
        'source_eligibility_updates.jsonl':source,
        'telemetry_index.jsonl':telemetry,
        'telemetry_eligibility.json':dict(records=policy,
            levels=dict(Counter(r['level'] for r in policy)),gate_B2=gate['B2']),
        'missing_fields.json':dict(telemetry=absences,clock_by_stream=[
            {k:v for k,v in r.items() if k in {'recording_id','topic','schema_key','segment','clock_missing_counts'}} for r in field]),
        'raw_timestamp_report.json':dict(rows=timestamps,investigation_witness=read(args.timing),
            unknown=['SDK host clock mode for each sample','source capture identity for equal compressed values',
                     'controller receive and response clocks'],mapping_applied=False),
        'manus_topology_report.json':dict(rows=topology,raw_fields_retained=True),
        'lineage_followup.json':followup,
        'dataset_coverage.json':dict(inherited_a_counts=old_summary['counts'],source_levels=dict(Counter(r['level'] for r in source)),
            source_root_count_confirmed=len(source_ids),catalogued_payloads=len(registry['payloads'])-schema_unavailable,
            catalogue_unavailable=schema_unavailable,inherited_native_metadata=old_coverage,
            sampled_recordings=len(fixtures),native_sample_coverage=coverage,
            bounded_native_overlap_ns=sum(r['four_stream_overlap_ns'] or 0 for r in coverage),
            final_test='BLOCKED',no_augmentation_counted=True),
        'test_results.json':unit,
        'raw_events.jsonl':events,
    }
    publish(args.output,products)
    print(stable_json(dict(run=args.output.parent.parent.name,B1=gate['B1'],B2=gate['B2'],
                           sampled_recordings=len(fixtures),real_events=len(events))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','a-config','a-registry','a-receipt','resource','unit','timing','output'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    output=args.output.resolve()
    expected=Path(read(args.config)['output_root']).resolve()
    if (not output.is_relative_to(expected/'runs') or not any(
            p.startswith('P01-B-') for p in output.relative_to(expected/'runs').parts[:-1])
            or output.exists() or output.is_relative_to(args.a_registry.parent.resolve())):
        p.error('Output must be new and isolated under the approved P01-B Run ID')
    run(args)


if __name__=='__main__':
    main()
