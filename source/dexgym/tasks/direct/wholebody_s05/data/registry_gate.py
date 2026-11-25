"""Independent A1 acceptance checks and conservative complete-copy signatures."""
from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path

from .lineage import Lineage

STATUSES = {
    'ADMITTED_CANDIDATE', 'EXCLUDED', 'QUARANTINED', 'DUPLICATE_ALIAS',
    'PENDING_DECODE_VERIFICATION',
}
REQUIRED_OUTPUTS = {
    'registry.jsonl','lineage.jsonl','prior_use.jsonl','rejects.jsonl','pending_verification.jsonl',
    'source_inventory.json','dataset_coverage.json','telemetry_candidates.json',
    'final_test_eligibility.json','resource_preflight.json','test_results.json',
}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def load_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def check_artifacts(path, receipt):
    manifest_path = path/'output_sha256.json'
    if not manifest_path.is_file():
        return ['MISSING_OUTPUT_SHA256']
    manifest = json.loads(manifest_path.read_bytes())
    errors = []
    if set(manifest)!=REQUIRED_OUTPUTS:
        errors.append('OUTPUT_SET_MISMATCH')
    if manifest!=receipt:
        errors.append('APPROVED_RECEIPT_MISMATCH')
    for name in sorted(REQUIRED_OUTPUTS):
        file = path/name
        if not file.is_file() or digest(file)!=manifest.get(name):
            errors.append(f'OUTPUT_MISSING_OR_HASH_MISMATCH:{name}')
    return errors


def merged_signature(info, group, parts):
    declared = info.get('relative_file_paths')
    if (info.get('storage_identifier')!='mcap' or group['order_source']!='sidecar' or
            group['missing'] or not declared or len(declared)!=len(set(declared)) or
            declared!=group['parts'] or len(parts)!=len(declared) or
            any(not a.get('file_sha256') or a.get('checksum_state')!='VERIFIED' for a in parts)):
        return None
    listed = info.get('files') or []
    if ([item.get('path') for item in listed]!=declared or any(
            not isinstance(item.get('message_count'),int) or
            item['message_count']!=part.get('verified_message_count') for item,part in zip(listed,parts))):
        return None
    return 'rosbag2', tuple(a['file_sha256'] for a in parts)


def signature_conflicts(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get('signature'):
            groups[row['signature']].append(row)
    conflicts = []
    for key,group in groups.items():
        if len(group)<2:
            continue
        fields = {
            'producer_kind': {r.get('producer_kind') for r in group},
            'metadata_sha256': {r['evidence'][0].get('sha256') for r in group if r.get('evidence')},
            'calibration_sha256': {tuple(sorted(ref['sha256'] for ref in r.get('calibration_refs',())))
                                   for r in group},
            'recorder_config_sha256': {r['evidence'][1].get('sha256') if len(r.get('evidence',[]))>1
                                       else None for r in group},
        }
        differs = {field:sorted(value,key=str) for field,value in fields.items() if len(value)>1}
        if differs:
            conflicts.append({'record_ids':sorted(r['id'] for r in group),
                              'differences':differs,'signature':key})
    return sorted(conflicts,key=lambda r:r['record_ids'])


def ordered_payload(row, by_id):
    parts = [by_id.get(part,{}) for part in row.get('source_asset_ids',())]
    return [(a.get('role'),a.get('file_sha256')) for a in parts
            if row.get('format')!='dex-native-v2' or a.get('role','').startswith('streams/')]


def evaluate_registry(rows, lineage, inventory, artifact_errors=()):
    """Return a technical decision separately from whether the checker ran."""
    errors = list(artifact_errors)
    by_id = {r.get('id'):r for r in rows}
    graph_rows = {x.get('episode_id'):x for x in lineage}
    roles = Counter()
    seen_parts = set()
    admitted = []
    if None in by_id or len(by_id)!=len(rows):
        errors.append('INVALID_OR_REPEATED_REGISTRY_ID')
    if None in graph_rows or len(graph_rows)!=len(lineage):
        errors.append('INVALID_OR_REPEATED_LINEAGE_ID')
    for row in rows:
        ident,kind,status = row.get('id'),row.get('record_type'),row.get('status')
        if status not in STATUSES or kind not in {'recording','asset','archive_container','historical_artifact'}:
            errors.append(f'REGISTRY_FIELDS:{ident}')
            continue
        roles[kind,status] += 1
        if kind=='recording':
            parts = row.get('source_asset_ids')
            if not isinstance(parts,list) or len(parts)!=row.get('part_count') or len(parts)!=len(set(parts)):
                errors.append(f'PART_COUNT:{ident}')
                continue
            for asset_id in parts:
                if by_id.get(asset_id,{}).get('source_id')!=ident or asset_id in seen_parts:
                    errors.append(f'PART_PARENT:{ident}:{asset_id}')
                seen_parts.add(asset_id)
            if not row.get('producer_kind') or ident not in graph_rows:
                errors.append(f'SOURCE_OR_LINEAGE_FIELDS:{ident}')
            if status=='ADMITTED_CANDIDATE':
                root = row.get('root_episode_id')
                refs = row.get('calibration_refs') or []
                topics = row.get('capabilities',{}).get('human_topics') or {}
                if (row.get('producer_kind')!='LIVE_TELEOP' or not root or root not in graph_rows or
                        graph_rows[ident].get('root_episode_ids')!=[root] or not refs or
                        not all(ref.get('sha256') for ref in refs) or not topics or not all(topics.values()) or
                        not parts or row.get('flags') or any(
                            by_id.get(key,{}).get('checksum_state')!='VERIFIED' for key in parts)):
                    errors.append(f'UNSAFE_OR_UNPROVEN_ADMISSION:{ident}')
                admitted.append(ident)
            if status=='DUPLICATE_ALIAS':
                canonical = by_id.get(row.get('alias_of'),{})
                if (canonical.get('record_type')!='recording' or canonical.get('status')=='DUPLICATE_ALIAS' or
                        canonical.get('id')==ident or not parts or
                        ordered_payload(row,by_id)!=ordered_payload(canonical,by_id)):
                    errors.append(f'INVALID_RECORDING_ALIAS:{ident}')
        elif kind=='asset' and status=='DUPLICATE_ALIAS':
            canonical = by_id.get(row.get('alias_of'),{})
            if (canonical.get('record_type')!='asset' or canonical.get('status')=='DUPLICATE_ALIAS' or
                    not row.get('file_sha256') or row['file_sha256']!=canonical.get('file_sha256')):
                errors.append(f'INVALID_ASSET_ALIAS:{ident}')
    if seen_parts!={r.get('id') for r in rows if r.get('record_type')=='asset'}:
        errors.append('ASSET_CONSERVATION')
    counts = inventory.get('counts',{})
    measured = {'logical_recording_views':sum(v for (k,_),v in roles.items() if k=='recording'),
                'mcap_instances':len(seen_parts),'fully_admitted_metadata_recordings':len(admitted)}
    for key,expected in measured.items():
        if key in counts and counts[key]!=expected:
            errors.append(f'COUNT_MISMATCH:{key}')
    for kind, statuses in counts.get('status_counts',{}).items():
        if any(roles[kind,state]!=n for state,n in statuses.items()):
            errors.append(f'STATUS_COUNT_MISMATCH:{kind}')
    if any(not isinstance(row.get('parent_ids'),list) for row in lineage):
        errors.append('MISSING_LINEAGE_PARENTS')
    else:
        graph = Lineage({ident:row['parent_ids'] for ident,row in graph_rows.items()})
        try:
            for ident,row in graph_rows.items():
                roots,unresolved = graph.resolve(ident)
                if list(roots)!=row.get('root_episode_ids') or list(unresolved)!=row.get('unresolved_refs'):
                    errors.append(f'LINEAGE_RESOLUTION_MISMATCH:{ident}')
        except ValueError as exc:
            errors.append(str(exc))
    gate = 'FAIL' if errors else 'PASS' if admitted and not inventory.get('errors') else 'BLOCKED'
    return {'gate':gate,'errors':errors,'admitted_count':len(admitted),'measured':measured,
            'process_semantics':'Successful checker execution is independent of gate and human approval.'}


def inspect_snapshot(path, receipt):
    errors = check_artifacts(path,receipt)
    if errors:
        return {'gate':'FAIL','errors':errors}
    return evaluate_registry(load_rows(path/'registry.jsonl'),load_rows(path/'lineage.jsonl'),
                             json.loads((path/'source_inventory.json').read_bytes()))
