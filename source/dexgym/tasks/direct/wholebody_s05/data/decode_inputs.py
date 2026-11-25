"""Immutable P01-A inputs, B catalogue and reproducible bounded sample selection."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import yaml

from .adapters.ros2_merged_mcap import catalogue_asset
from .registry_gate import digest, merged_signature, ordered_payload, signature_conflicts
from .telemetry import TOPICS

MANUS = {'/manus_glove_left','/manus_glove_right'}
WRISTS = {f'/vive/{kind}/{side}_wrist' for kind in ('raw','tracker') for side in ('left','right')}
CONTROL_METADATA = {'/dex/collection/status','/dex/collection/topics','/events/write_split'}
TARGET_TOPICS = MANUS | WRISTS | TOPICS.keys() | CONTROL_METADATA
OBSERVATION_STREAMS = {'manus_left','manus_right','tracker_left','tracker_right',
                       'tracker_left_raw','tracker_right_raw','tracker_safety'}


def relevant(topic):
    if topic in TARGET_TOPICS:
        return True
    if topic.startswith('/dex/collection/observations/'):
        return topic.rsplit('/',1)[-1] in OBSERVATION_STREAMS
    return False


def pack_catalogue(assets, progress=None):
    """Catalogue each payload once and retain every original asset reference."""
    groups = defaultdict(list)
    for asset in assets.values():
        key = asset.get('file_sha256') or asset['id']
        groups[key].append(asset)
    schemas, contents, association = {}, {}, {}
    for i,(key,group) in enumerate(sorted(groups.items()),1):
        supported = []
        failures = []
        selected = None
        for asset in sorted(group,key=lambda a:(a.get('storage')=='ZIP_MEMBER',a['canonical_uri'])):
            try:
                channels, status = catalogue_asset(asset)
            except Exception as error:
                failures.append({'part_id':asset['id'],'detail':f'{type(error).__name__}: {error}'})
                continue
            if status=='CATALOGUED':
                selected = asset['id']
                supported = channels
                break
        for row in supported:
            schema = row.pop('schema')
            if schema:
                schemas[schema['key']] = schema
            row['schema_key'] = schema['key'] if schema else None
        contents[key] = {'catalogue_state':'CATALOGUED' if selected else 'UNAVAILABLE',
                         'catalogue_part_id':selected,'channels':supported,'inspection_failures':failures}
        for asset in group:
            association[asset['id']] = key
        if progress and i%50==0:
            progress(i,len(groups))
    return {'schema':'s05.schema-registry.p01b.v1','schemas':schemas,'payloads':contents,'part_map':association}


def channels_for(registry, asset_id):
    return registry['payloads'][registry['part_map'][asset_id]]['channels']


def read_sidecar(row):
    locator = row['locator']
    if '!' in locator:
        container, member = locator.split('!',1)
        logical = member.rsplit('#',1)[0]
        path = str(Path(logical)/'metadata.yaml')
        with ZipFile(container) as archive:
            data = archive.read(path)
    else:
        data = (Path(locator.split('#',1)[0])/'metadata.yaml').read_bytes()
    expected = row['evidence'][0].get('sha256') if row.get('evidence') else None
    if not expected or sha256(data).hexdigest()!=expected:
        raise ValueError('Legacy sidecar is unbound or has changed since P01-A')
    return yaml.safe_load(data).get('rosbag2_bagfile_information',{})


def capture_notes(recording):
    """Inventory supporting files without treating a recorder subscription as producer proof."""
    locator = recording['locator']
    if recording['format']!='ros2-merged-mcap' or '!' in locator:
        return []
    folder = Path(locator.split('#',1)[0])
    files = {
        'recorder.log':'RECORDER_LIFECYCLE_ONLY',
        'calibration_windows.json':'ANNOTATED_CALIBRATION_WINDOW_ONLY',
        'run_snapshot.json':'CAPTURE_CONFIGURATION_REQUIRES_REVIEW',
        'effective_config.yaml':'CAPTURE_CONFIGURATION_REQUIRES_REVIEW',
        'events.jsonl':'RECORDED_EVENTS_REQUIRE_PRODUCER_BINDING',
    }
    found = []
    for name,role in files.items():
        path = folder/name
        if path.is_file() and path.stat().st_size<=4*2**20:
            found.append(dict(path=str(path),sha256=digest(path),scope=role))
    return found


def signature_followup(recordings, assets, registry, old_inventory):
    retrospective, provisional, weak, inaccessible = [], [], [], []
    for row in recordings:
        payload = ordered_payload(row,assets)
        if row['format']=='dex-native-v2':
            key = ('native',tuple(payload)) if payload and all(item[1] for item in payload) else None
        else:
            parts = [assets[i] for i in row['source_asset_ids']]
            try:
                info = read_sidecar(row)
            except (OSError,KeyError,ValueError,yaml.YAMLError) as exc:
                inaccessible.append({'recording_id':row['id'],'detail':f'{type(exc).__name__}: {exc}'})
                info = {}
            declared = info.get('relative_file_paths') or []
            names = [Path(p['uri_rel'].split('!')[-1]).name for p in parts]
            witness = row['evidence'][1] if len(row['evidence'])>1 else {}
            group = {'parts':names,'missing':witness.get('missing_parts') or [],
                     'order_source':witness.get('part_order_source')}
            observed = [[c['declared_count'] for c in channels_for(registry,p['id'])] for p in parts]
            checked = [dict(p,verified_message_count=sum(counts) if counts and None not in counts else None)
                       for p,counts in zip(parts,observed)]
            key = merged_signature(info,group,checked)
            if payload and key is None:
                weak.append({'recording_id':row['id'],'old_status':row['status'],'old_alias_of':row.get('alias_of'),
                             'reason':'COMPLETE_ORDER_OR_FOOTER_COUNTS_UNATTESTED'})
        retrospective.append(dict(row,signature=key))
        if payload and all(item[1] for item in payload):
            provisional.append(dict(row,signature=(row['format'],tuple(payload))))
    conflicts = signature_conflicts(retrospective)
    return {'schema':'s05.lineage-followup.p01b.v1','weak_merged_signatures':weak,
            'annotation':'Historic A1 statuses are preserved. Weak copies excluded from new positive admissions.',
            'complete_payload_conflicts':conflicts,
            'equal_observed_payload_declaration_conflicts':signature_conflicts(provisional),
            'inaccessible_sidecars':inaccessible,
            'old_unresolved_prior_use_events':old_inventory['counts']['unresolved_usage_events'],
            'unresolved_semantic_matches':[]}


def select_recordings(recordings, registry, config):
    by_id = {row['id']:row for row in recordings}
    schema_keys = {}
    for row in recordings:
        schema_keys[row['id']] = {(row['format'],item['schema_key'])
                                   for asset in row['source_asset_ids']
                                   for item in channels_for(registry,asset)
                                   if item['schema_key'] and relevant(item['topic']) and item['declared_count']!=0}
    selected = {}
    for name in config['required_recordings']:
        match = [r['id'] for r in recordings if r['locator'].endswith(name) and r['status']!='DUPLICATE_ALIAS']
        if len(match)!=1:
            raise ValueError(f'Required fixture did not resolve uniquely: {name}')
        selected[match[0]] = ['REQUIRED_FIXTURE']
    native = sorted((r for r in recordings if r['format']=='dex-native-v2' and r['status']=='ADMITTED_CANDIDATE'),
                    key=lambda r:(r.get('capture_date') or '',r['locator']))
    strata = defaultdict(list)
    for row in native:
        for key in (('date',row.get('capture_date')),('task',row.get('task')),
                    ('profile',row['evidence'][-1].get('profile'))):
            strata[key].append(row['id'])
    for key, ids in sorted(strata.items(),key=str):
        ident = next((i for i in ids if i in selected),ids[0])
        selected.setdefault(ident,[]).append(f'STRATUM:{key[0]}:{key[1]}')
    for row in native:
        if row.get('task')=='bi_hand':
            selected.setdefault(row['id'],[]).append('BIMANUAL_TAG_ALL')
    comparable = [r for r in recordings if r['status']!='DUPLICATE_ALIAS' and
                  r['producer_kind'] not in {'EGOPRO','DERIVED_ONLY','SIM_REPLAY'}]
    covered = set().union(*(schema_keys[i] for i in selected))
    while len(selected)<config['max_recordings']:
        candidates = [r for r in comparable if r['id'] not in selected and schema_keys[r['id']]-covered]
        if not candidates:
            break
        best = min(candidates,key=lambda r:(-len(schema_keys[r['id']]-covered),
                    r['status']!='ADMITTED_CANDIDATE', '!' in r['locator'], r['locator']))
        selected[best['id']] = ['ADDITIONAL_SCHEMA_VERSION']
        covered.update(schema_keys[best['id']])
    return [(by_id[i],reasons) for i,reasons in selected.items()]


def native_manifest(row, config):
    folder = Path(row['canonical_uri'])
    args = row['evidence']
    path = folder/'manifest.yaml'
    if digest(path)!=args[0].get('sha256'):
        raise ValueError(f'Native manifest SHA changed: {path}')
    manifest = yaml.safe_load(path.read_bytes())
    if manifest.get('format')!='dex-native-v2' or manifest.get('episode_id')!=row.get('episode_ref'):
        raise ValueError(f'Native episode identity mismatch: {folder}')
    snapshot = folder/'run_snapshot.json'
    expected = args[1].get('sha256')
    if not expected and row['status']!='ADMITTED_CANDIDATE':
        # Decode the immutable manifest + original CDR for diagnostics only.
        return manifest,False
    if digest(snapshot)!=expected:
        raise ValueError(f'Native snapshot SHA changed: {snapshot}')
    snap = json.loads(snapshot.read_bytes())
    recorder = config['native_writer']
    binding = any(item.get('root')==recorder['repo'] and item.get('commit')==recorder['commit']
                  for item in snap.get('sources',()))
    if binding:
        result = subprocess.check_output(['git','-C',recorder['repo'],'show',
                                          f"{recorder['commit']}:{recorder['path']}"])
        binding = sha256(result).hexdigest()==recorder['sha256']
    return manifest,binding
