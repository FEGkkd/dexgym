"""Read-only inventory adapters for P01-A recording and historical metadata."""
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

import yaml

from .historical_usage import ARTIFACT_KEYS, history_events, iter_context, references
from .lineage import Lineage
from .source_registry import (
    HUMAN_TOPICS, STATUSES, capability, classify, coverage, date_from_ns, final_test,
    legacy_groups, record_aliases, root_use, same_bytes, stable_id, stable_json, stat_key,
)

SKIP_DIRS = {'.git','.venv','__pycache__','site-packages','node_modules',
             'training_source','checkout','source_snapshot','video','videos'}
REPLAY_NAME = re.compile(r'(^|[_-])(sim|replay|retime|slow|egopro)([_-]|$)',re.I)


def document(data):
    value = yaml.safe_load(data)
    if not isinstance(value,dict):
        raise ValueError('metadata is not an object')
    return value


def metadata(path):
    data = path.read_bytes()
    return document(data), {'path':str(path), 'sha256':sha256(data).hexdigest()}


def safe_part(name):
    p = PurePosixPath(name)
    return not p.is_absolute() and '..' not in p.parts and '\\' not in name


def legacy_catalogue(info):
    output = []
    for row in info.get('topics_with_message_count',[]):
        meta = row.get('topic_metadata',{})
        if meta.get('name'):
            output.append(dict(topic=meta['name'],type=meta.get('type'),
                               count=row.get('message_count'),definition_hash=meta.get('type_description_hash')))
    return sorted(output,key=lambda r:r['topic'])


def summary_catalogue(path):
    from mcap.reader import make_reader

    with path.open('rb') as f:
        summary = make_reader(f).get_summary()
    if not summary:
        return []
    counts = summary.statistics.channel_message_counts if summary.statistics else {}
    return sorted((dict(topic=c.topic,type=summary.schemas[c.schema_id].name if c.schema_id in summary.schemas else None,
                        count=counts.get(c.id),definition_hash=sha256(summary.schemas[c.schema_id].data).hexdigest()
                        if c.schema_id in summary.schemas else None) for c in summary.channels.values()),
                  key=lambda r:r['topic'])


def checked_meta(path):
    try:
        return metadata(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {}, {'path':str(path),'error':f'{type(exc).__name__}: {exc}'}


def calibration_witnesses(snapshot):
    """Verify the recorded tracker launch configuration against a readable copy."""
    witnesses = []
    for unit in snapshot.get('units', ()):
        if not isinstance(unit,dict) or unit.get('unit') != 'tracker':
            continue
        for entry in unit.get('startedConfigurations', ()):
            if not isinstance(entry,dict) or not isinstance(entry.get('sha256'),str):
                continue
            for source in (entry.get('snapshot'),entry.get('path')):
                if not isinstance(source,str) or not Path(source).is_absolute():
                    continue
                config,witness = checked_meta(Path(source))
                node = config.get('tracker_calibration_node')
                if (witness.get('sha256') == entry['sha256'] and isinstance(node,dict)
                        and isinstance(node.get('ros__parameters'),dict) and node['ros__parameters']):
                    witnesses.append(dict(**witness, source='capture_tracker_started_configuration',
                                          unit_revision=unit.get('revision')))
                    break
    return witnesses


def role_topic_map(catalogue):
    present = {r['topic'] for r in catalogue if (r['count'] is None or r['count'] > 0)}
    return all(present.intersection(names) for names in HUMAN_TOPICS.values())


def positive_native_origin(manifest, snapshot, publishers, trusted):
    allowed = trusted['publisher_by_stream']
    names_match = all(any(p.startswith(node+' [') for p in publishers.get(stream,()))
                      for stream,node in allowed.items())
    named_replay = any('rosbag2_player' in p or '/mujoco' in p.lower()
                      for items in publishers.values() for p in items)
    if named_replay:
        return 'SIM_REPLAY', ['RECORDED_REPLAY_OR_SIM_PUBLISHER']
    if not names_match or manifest.get('format')!='dex-native-v2':
        return 'UNKNOWN', ['LIVE_DRIVER_PUBLISHERS_NOT_VERIFIED']
    if (snapshot.get('scope') != 'source_and_started_configuration' or
            snapshot.get('profile') not in trusted['live_profiles']):
        return 'UNKNOWN', ['CONTEMPORARY_REAL_PROFILE_NOT_VERIFIED']
    if manifest.get('runtime',{}).get('source_mode') not in trusted['source_modes']:
        return 'UNKNOWN', ['COLLECTOR_SOURCE_MODE_NOT_VERIFIED']
    return 'LIVE_TELEOP', ['COLLECTOR_CAPTURE_SNAPSHOT_REAL_PROFILE','LIVE_PUBLISHER_IDENTITIES']


def timed_metadata(info):
    start = info.get('starting_time',{}).get('nanoseconds_since_epoch')
    duration = info.get('duration',{}).get('nanoseconds')
    return start, duration


class Inventory:
    def __init__(self, config):
        self.config = config
        self.records = []
        self.assets = []
        self.artifacts = []
        self.discovery = []
        self.errors = []
        self.pending = []
        self.cache = {}
        self.archive_hashed_bytes = 0

    def error(self, path, reason):
        self.errors.append(dict(path=str(path), reason=str(reason)))

    def iter_tree(self, root):
        if not root.is_dir():
            self.error(root,'ROOT_MISSING_OR_UNREADABLE')
            return
        def onerror(err):
            self.error(err.filename,err)
        for directory,subdirs,names in os.walk(root,onerror=onerror,followlinks=False):
            subdirs[:] = sorted(set(subdirs)-SKIP_DIRS)
            yield Path(directory), sorted(names)

    def asset(self, path, logical, record, part_order, role):
        row = dict(schema='raw_registry.v1',record_type='asset',id=stable_id('asset',[record,logical]),
                   source_id=record,uri_rel=logical,part_order=part_order,role=role,
                   status='ADMITTED_CANDIDATE',admission_reason='BYTES_HASHED_FOR_INVENTORY')
        try:
            row.update(same_bytes(path,self.cache))
            row.update(file_sha256=row['sha256'],checksum_state='VERIFIED')
        except (OSError, ValueError) as exc:
            row.update(status='QUARANTINED',admission_reason='MCAP_READ_FAILED',
                       checksum_state='FAILED',file_sha256=None,error=str(exc))
        self.assets.append(row)
        return row

    def archive_asset(self, archive, path, member, record, part_order):
        uri = f'{path}!{member}'
        row = dict(schema='raw_registry.v1',record_type='asset',id=stable_id('asset',[record,uri]),
                   source_id=record,uri_rel=uri,canonical_uri=uri,part_order=part_order,role='episode',
                   storage='ZIP_MEMBER',status='ADMITTED_CANDIDATE',admission_reason='BYTES_HASHED_FOR_INVENTORY')
        try:
            with archive.open(member) as f:
                row['file_sha256'] = file_digest(f,'sha256').hexdigest()
            row.update(size_bytes=archive.getinfo(member).file_size, checksum_state='VERIFIED')
            self.archive_hashed_bytes += row['size_bytes']
        except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            row.update(status='QUARANTINED',admission_reason='ARCHIVE_MEMBER_READ_FAILED',
                       checksum_state='FAILED',file_sha256=None,error=str(exc))
        self.assets.append(row)
        return row

    def recording(self, *, locator, source, fmt, origin, evidence, catalogue, parts, start, duration,
                  flags=(), task=None, session=None, collector=None, episode=None, hardware=None,
                  calibration=(), signature=None, priority=0, complete=True, view=None):
        flags = sorted(set(flags))
        status,reason = classify(origin,complete and role_topic_map(catalogue),flags)
        quality = dict(start_ns=start, stop_ns=(start+duration if start is not None and duration is not None else None),
                       clock='RECORDER_REALTIME', controller_receive='UNKNOWN')
        rid = stable_id('recording',locator)
        cs = capability(catalogue,origin,hardware,calibration,start is not None,status)
        row = dict(schema='raw_registry.v1',record_type='recording',id=rid,source_id=source,locator=locator,
                   canonical_uri=str(Path(view).resolve()) if view else locator,format=fmt,producer_kind=origin,
                   status=status,admission_reason=reason,evidence=evidence,flags=flags, catalogue=catalogue,
                   source_asset_ids=[a['id'] for a in parts],part_count=len(parts),episode_ref=episode,
                   interval=quality,duration_ns=duration,capture_date=date_from_ns(start),
                   task=task or None,session_id=session or None,collector_instance=collector,
                   calibration_refs=calibration,capabilities=cs,signature=signature,priority=priority,
                   checksum_state='VERIFIED' if parts and all(a['file_sha256'] for a in parts) else 'INCOMPLETE')
        self.records.append(row)
        return row

    def native_dirs(self, root, indexed):
        if not root.is_dir():
            self.error(root,'ROOT_MISSING_OR_UNREADABLE')
            return
        candidates = sorted((p for p in root.iterdir() if p.is_dir() or p.is_symlink()))
        if indexed:
            for day in candidates:
                if day.is_dir():
                    yield from sorted(p for p in day.iterdir() if p.is_dir() or p.is_symlink())
                else:
                    self.error(day,'UNREADABLE_DATE_INDEX')
        else:
            yield from candidates

    def scan_native(self, cfg):
        root = Path(cfg['path'])
        collector_root = Path(self.config['native_storage']).resolve()
        count = 0
        for path in self.native_dirs(root,cfg['mode']=='native_index'):
            count += 1
            real = path.resolve()
            rid = stable_id('recording',str(path))
            inside = real.is_relative_to(collector_root)
            manifest,sample = checked_meta(path/'manifest.yaml') if inside else ({},{'error':'TARGET_OUTSIDE_SOURCE_ROOT'})
            snapshot,config_witness = checked_meta(path/'run_snapshot.json') if inside else ({},{})
            flags = []
            files = sorted(path.rglob('*.mcap')) if inside else []
            parts = []
            for idx,file in enumerate(files):
                rel = file.relative_to(path).as_posix()
                role = str(PurePosixPath(rel).parent)
                part = self.asset(file,rel,rid,idx,role)
                parts.append(part)
            currents = [p for p in parts if p['role'].startswith('streams/')]
            catalogue = []
            publishers = {}
            for stream,row in sorted(manifest.get('streams',{}).items()):
                if not isinstance(row,dict):
                    continue
                topic = row.get('topic')
                if topic and row.get('recorded') is True:
                    catalogue.append(dict(topic=topic,type=row.get('type'),count=row.get('written'),
                                          definition_hash=None))
                publishers[stream]=row.get('publishers_at_start') or []
            origin,origin_evidence = positive_native_origin(manifest,snapshot,publishers,self.config['native_attestation'])
            start, stop = manifest.get('start_realtime_ns'), manifest.get('stop_realtime_ns')
            duration = stop-start if isinstance(start,int) and isinstance(stop,int) and stop>=start else None
            sources = snapshot.get('sources',[])
            calibration = calibration_witnesses(snapshot)
            identity = (manifest.get('collector_instance_id'), manifest.get('episode_id'),
                        manifest.get('start_monotonic_ns'),manifest.get('stop_monotonic_ns'))
            if manifest.get('state')!='COMPLETED':
                flags.append('CAPTURE_INCOMPLETE')
            if not inside or not all(identity):
                flags.append('CAPTURE_ID_OR_APPROVED_PATH_MISSING')
            if not calibration:
                flags.append('CALIBRATION_REFERENCE_PENDING')
            if not currents or any(not a['file_sha256'] for a in parts):
                flags.append('MCAP_PARTS_UNVERIFIED')
            if not role_topic_map(catalogue):
                flags.append('REQUIRED_HUMAN_STREAMS_UNVERIFIED')
            if duration is None:
                flags.append('CAPTURE_DURATION_UNVERIFIED')
            found_roles = {p['role'] for p in currents}
            if any(r.get('recorded') and r.get('path') not in found_roles
                   for r in manifest.get('streams',{}).values() if isinstance(r,dict)):
                flags.append('DECLARED_STREAM_WITHOUT_MCAP')
            sig = None
            if currents and all(p['file_sha256'] for p in currents):
                ordered = tuple((p['role'],p['file_sha256']) for p in currents)
                sig = ('native',ordered)
            mm = manifest.get('metadata') or {}
            row = self.recording(locator=str(path),source=cfg['label'],fmt='dex-native-v2',
                origin=origin,evidence=[sample,config_witness,dict(publishers=publishers,attestation=origin_evidence,
                    capture_identity=identity,profile=snapshot.get('profile'),
                    code_source_commits=sources, control_observation=snapshot.get('controlObservation'))],
                catalogue=catalogue,parts=parts,
                start=start,duration=duration,flags=flags,task=mm.get('task_name'),session=mm.get('session_name'),
                collector=identity[0],episode=identity[1],hardware=(snapshot.get('profile') if
                    (snapshot.get('controlObservation') or {}).get('robot',{}).get('hardware_state_name')
                    not in {None,'DISCONNECTED'} else None),
                calibration=calibration,signature=sig,priority=3 if cfg['mode']=='native' else 2,view=path,
                complete=duration is not None)
            row['collector_interval_ns'] = list(identity[2:])
        self.discovery.append(dict(root=str(root),mode=cfg['mode'],recording_directories=count))

    def merged(self, locator, info, witness, group, parts, label, catalogue, archived=False):
        origin = 'UNKNOWN'
        flags = []
        conf = self.config['positive_replays'].get(locator)
        signature = None
        if parts and all(p['file_sha256'] for p in parts):
            signature = ('rosbag2',tuple(p['file_sha256'] for p in parts))
        if conf and conf['metadata_sha256']==witness.get('sha256'):
            origin = 'SIM_REPLAY'
        elif REPLAY_NAME.search(locator):
            flags.append('SUSPECTED_REPLAY_OR_DERIVED_NAME')
        if not catalogue:
            flags.append('TOPIC_CATALOGUE_PENDING')
        if group['missing'] or not signature:
            flags.append('MCAP_PARTS_UNVERIFIED')
        if archived:
            flags.append('ARCHIVE_MEMBER_ORIGIN_PENDING')
        start,duration = timed_metadata(info)
        row = self.recording(locator=locator,source=label,fmt='ros2-merged-mcap',origin=origin,
            evidence=[witness,{'part_order_source':group['order_source'],'missing_parts':group['missing'],
                               'replay_evidence_ref':conf['evidence_ref'] if origin=='SIM_REPLAY' else None}],
            catalogue=catalogue,parts=parts,start=start,duration=duration,flags=flags,
            priority=1 if not archived else 0, signature=signature,complete=bool(signature))
        row['asset_group_view']='ZIP' if archived else 'DIRECTORY'

    def scan_legacy(self, cfg):
        root = Path(cfg['path'])
        count = 0
        archives = []
        for path,names in self.iter_tree(root):
            archives.extend(path/n for n in names if n.lower().endswith('.zip'))
            files = [n for n in names if n.lower().endswith('.mcap')]
            if not files and 'metadata.yaml' not in names:
                continue
            count += 1
            doc,witness = checked_meta(path/'metadata.yaml')
            info = doc.get('rosbag2_bagfile_information',{})
            declared = [p for p in (info.get('relative_file_paths') or []) if isinstance(p,str) and safe_part(p)]
            for i,group in enumerate(legacy_groups(files,declared)):
                loc = str(path) if i==0 else f'{path}#group-{i+1}'
                rid = stable_id('recording',loc)
                parts = [self.asset(path/n,n,rid,j,'episode') for j,n in enumerate(group['parts']) if n in files]
                cat = legacy_catalogue(info) if i==0 else []
                if not cat:
                    for n in group['parts']:
                        if n in files:
                            try:
                                cat.extend(summary_catalogue(path/n))
                            except Exception as exc:
                                self.pending.append(dict(source_id=rid,reason='MCAP_SUMMARY_UNAVAILABLE',
                                                         path=str(path/n),error=str(exc)))
                self.merged(loc,info if i==0 else {},witness,group,parts,cfg['label'],cat)
            if not files and not declared:
                self.recording(locator=str(path),source=cfg['label'],fmt='ros2-merged-mcap',origin='UNKNOWN',
                    evidence=[witness],catalogue=[],parts=[],start=None,duration=None,flags=('NO_MCAP_PARTS',))
        for path in archives:
            self.scan_archive(path,cfg['label'])
        self.discovery.append(dict(root=str(root),mode='legacy',
                                   recording_directories=count,zip_containers=len(archives)))

    def scan_archive(self,path,label):
        container=dict(schema='raw_registry.v1',record_type='archive_container',id=stable_id('archive',str(path)),
                       locator=str(path),status='EXCLUDED',admission_reason='CONTAINER_INDEXED_BY_MEMBER')
        self.artifacts.append(container)
        try:
            initial = stat_key(path.stat())
            with zipfile.ZipFile(path) as archive:
                members = [n for n in archive.namelist() if safe_part(n)]
                grouped = defaultdict(list)
                for n in members:
                    if n.lower().endswith('.mcap'):
                        grouped[str(PurePosixPath(n).parent)].append(PurePosixPath(n).name)
                if not grouped:
                    self.pending.append(dict(path=str(path),reason='ARCHIVE_WITHOUT_MCAP_MEMBERS'))
                for directory,files in sorted(grouped.items()):
                    md = str(PurePosixPath(directory)/'metadata.yaml')
                    info,witness = {},{'path':f'{path}!{md}'}
                    if md in members:
                        payload = archive.read(md)
                        info = document(payload).get('rosbag2_bagfile_information',{})
                        witness['sha256'] = sha256(payload).hexdigest()
                    declared = [n for n in (info.get('relative_file_paths') or []) if isinstance(n,str) and safe_part(n)]
                    for i,group in enumerate(legacy_groups(files,declared)):
                        loc = f'{path}!{directory}#{i+1}'
                        rid = stable_id('recording',loc)
                        parts = [self.archive_asset(archive,path,str(PurePosixPath(directory)/n),rid,j)
                                 for j,n in enumerate(group['parts']) if n in files]
                        self.merged(loc,info if i==0 else {},witness,group,parts,label,
                                    legacy_catalogue(info) if i==0 else [],archived=True)
            if stat_key(path.stat())!=initial:
                self.error(path,'ARCHIVE_CHANGED_DURING_SCAN')
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, yaml.YAMLError) as exc:
            container.update(status='QUARANTINED',admission_reason='ARCHIVE_INDEX_UNAVAILABLE',error=str(exc))
            self.pending.append(dict(reason='ARCHIVE_INDEX_UNAVAILABLE',path=str(path),error=str(exc)))

    def candidate_relations(self):
        """Use shared bytes, identity and overlapping metadata as review cues, not semantic equality."""
        by_episode = defaultdict(list)
        by_capture = defaultdict(list)
        by_part = defaultdict(set)
        lookup = {r['id']:r for r in self.records}
        for r in self.records:
            if r.get('alias_of'):
                continue
            if r['episode_ref']:
                by_episode[r['episode_ref']].append(r['id'])
            if r['collector_instance'] and r['interval']['start_ns'] is not None and r['interval']['stop_ns'] is not None:
                by_capture[r['collector_instance']].append(r)
        for a in self.assets:
            r=lookup[a['source_id']]
            if a['file_sha256'] and not r.get('alias_of') and 'initial_state' not in a['role']:
                by_part[a['file_sha256']].add(a['source_id'])
        rows=[]
        for ref,ids in by_episode.items():
            if len(ids)>1:
                rows.append(dict(reason='REUSED_EPISODE_ID_NONEXACT',key=ref,record_ids=sorted(ids)))
        for digest,ids in by_part.items():
            if len(ids)>1:
                rows.append(dict(reason='SHARED_MCAP_PART_NONEXACT',key=digest,record_ids=sorted(ids)))
        for ref,records in by_capture.items():
            active=[]
            for r in sorted(records,key=lambda r:(r['interval']['start_ns'],r['id'])):
                start=r['interval']['start_ns']
                active=[x for x in active if x['interval']['stop_ns']>start]
                if active:
                    rows.append(dict(reason='COLLECTOR_CLOCK_INTERVAL_OVERLAP',key=ref,
                                     record_ids=sorted([r['id'],*(x['id'] for x in active)])))
                active.append(r)
        for relation in rows:
            for rid in relation['record_ids']:
                row=lookup[rid]
                if row['status']=='ADMITTED_CANDIDATE':
                    row.update(status='PENDING_DECODE_VERIFICATION',admission_reason='SEMANTIC_LINEAGE_UNRESOLVED')
                    row['capabilities']['human_status']=row['status']
                row['flags']=sorted(set(row['flags'])|{'SEMANTIC_LINEAGE_UNRESOLVED'})
        self.pending.extend(rows)

    def history_paths(self, path):
        for directory,names in self.iter_tree(path):
            for name in names:
                if (name in {'manifest.json','provenance.json','training_proof.json','evaluation.json',
                             'data_summary.json','state.json','library.json','motion_library.json','source_index.json'}
                        or re.fullmatch(r'review[_-].*\.json',name)
                        or name.endswith('_manifest.json')):
                    yield directory/name

    def historical(self, parents, source_refs):
        items = {}
        group_counts = defaultdict(Counter)
        for group in self.config['history_roots']:
            family = group['family']
            for leaf in group['paths']:
                root = Path(leaf)
                if not root.exists():
                    group_counts[family]['missing_roots'] += 1
                    self.pending.append(dict(reason='HISTORICAL_ROOT_MISSING',path=str(root),family=family))
                    continue
                paths = [root] if root.is_file() else self.history_paths(root)
                for path in paths:
                    if path in items:
                        continue
                    try:
                        raw=path.read_bytes()
                        data=json.loads(raw)
                        if not isinstance(data,(dict,list)):
                            raise ValueError('historical metadata is not an object or list')
                        digest=sha256(raw).hexdigest()
                    except (OSError,ValueError) as exc:
                        group_counts[family]['unreadable_documents'] += 1
                        self.pending.append(dict(reason='HISTORICAL_METADATA_UNREADABLE',path=str(path),family=family,error=str(exc)))
                        self.artifacts.append(dict(schema='raw_registry.v1',record_type='historical_artifact',
                            id=stable_id('artifact',str(path)),locator=str(path),family=family,
                            producer_kind='UNKNOWN',status='QUARANTINED',admission_reason='HISTORICAL_METADATA_UNREADABLE'))
                        continue
                    aid=stable_id('artifact',str(path))
                    items[path]=(family,data,aid,digest)
                    group_counts[family]['metadata_documents']+=1
        lookup={str(p.resolve()):v[2] for p,v in items.items()}
        events=[]
        for path,(family,data,aid,digest) in items.items():
            rows=history_events(data,aid,family)
            group_counts[family]['source_reference_events']+=len(rows)
            events.extend(rows)
            attached={r['source_ref'] for r in rows}
            edges=[]
            for node,field,role in iter_context(data):
                refs=references(node)
                clip=next((node[k] for k in ('clip_id','id','path') if isinstance(node.get(k),str)), None)
                if refs and clip:
                    cid=stable_id('fragment',[aid,field,clip])
                    parents[cid]=[source_refs.get(r,r) for r in refs]
                    edges.append(cid)
                for key in ARTIFACT_KEYS:
                    value=node.get(key)
                    if isinstance(value,str) and value.endswith('.json'):
                        target=str((path.parent/value).resolve())
                        related=lookup.get(target,'unknown-artifact:'+target)
                        if related!=aid:
                            edges.append(related)
            parents[aid] = sorted(set(edges + [source_refs.get(ref,ref) for ref in attached])) or ['unknown-usage:'+aid]
            self.artifacts.append(dict(schema='raw_registry.v1',record_type='historical_artifact', id=aid,locator=str(path),
                                       family=family,file_sha256=digest,producer_kind='DERIVED_ONLY',status='EXCLUDED',
                                       admission_reason='DERIVED_HISTORICAL_METADATA',source_reference_events=len(rows)))
            if not rows:
                events.append(dict(schema='historical_usage.v1',source_ref=aid,artifact_id=aid,family=family,
                                   role='USAGE_UNKNOWN',confidence='NO_EXPLICIT_SOURCE_ROLES',evidence_field='/',interval={}))
        self.record_s04_usage(items,events)
        for event in events:
            event['source_ref_original']=event['source_ref']
            event['source_ref']=source_refs.get(event['source_ref'],event['source_ref'])
        return events,dict(sorted((k,dict(v)) for k,v in group_counts.items()))

    def record_s04_usage(self, items, events):
        effective=self.config.get('s04_used_manifest',{})
        if not effective:
            return
        paths=[Path(effective[k]) for k in ('provenance','state')]
        if not all(p in items for p in paths):
            self.pending.append(dict(reason='S04_USAGE_PROOF_MISSING',paths=list(map(str,paths))))
            return
        _,proof,_,_=items[paths[0]]
        _,state,aid,_=items[paths[1]]
        key=proof.get('bank') or proof.get('manifest') or effective.get('bank')
        manifest=Path(key) if isinstance(key,str) else None
        if manifest not in items:
            self.pending.append(dict(reason='S04_EFFECTIVE_BANK_NOT_INDEXED',reference=key))
            return
        _,bank,bank_id,_=items[manifest]
        evaluations=state.get('reviews') or state.get('evaluations') or []
        tested=bool(evaluations) or isinstance(state.get('reviewed_iteration'),int) and state['reviewed_iteration']>0
        trained=isinstance(state.get('trained_iteration'),int) and state['trained_iteration']>0
        best=isinstance(state.get('recommended_iteration'),int)
        for row in history_events(bank,bank_id,'wholebody_s04'):
            roles=[]
            if row['role']=='TRAIN' and trained:
                roles.append('TRAIN')
            if row['role']=='VAL' and tested:
                roles.append('VAL')
                if best:
                    roles.append('MODEL_SELECTION')
            if row['role']=='HISTORICAL_TEST_ASSIGNMENT' and tested:
                roles.append('REPEATED_TEST')
            for role in roles:
                events.append(dict(**{k:v for k,v in row.items() if k not in {'artifact_id','role','confidence'}},
                                   role=role,confidence='RUN_STATE_AND_BANK_SPLIT',
                                   artifact_id=aid,bank_artifact_id=bank_id))

    def add_usage_index(self, events, graph, recordings):
        artifacts = {row['id']:row for row in self.artifacts}
        count = Counter()
        for event in events:
            roots, unresolved = graph.resolve(event['source_ref'])
            artifact = artifacts.get(event.get('artifact_id'), {})
            recording = recordings.get(event['source_ref'], {})
            path = artifact.get('locator') or recording.get('locator')
            fingerprint = artifact.get('file_sha256')
            evidence = dict(artifact_id=event.get('artifact_id'), field=event['evidence_field'],
                            source_ref=event.get('source_ref_original',event['source_ref']))
            key = stable_id('usage',[evidence,event['role'],event['confidence'],event['interval']])
            count[key] += 1
            event.update(usage_id=f'{key}:{count[key]}', artifact_family=event.get('family'),
                         artifact_path_or_uri=path, artifact_fingerprint=fingerprint,
                         observed_role=event['role'], method=event['confidence'],date_or_UNKNOWN='UNKNOWN',
                         episode_id=event['source_ref'], root_episode_ids=roots, unresolved_refs=unresolved,
                         root_episode_id=roots[0] if len(roots)==1 and not unresolved else None,
                         event_interval=event['interval'], transform_chain=None,
                         source_evidence=[evidence], reviewed_by=None)

    def add_source_index(self, records):
        for asset in self.assets:
            asset.setdefault('size_bytes',None)
            asset.update(asset_id=asset['id'],format='mcap',producer_kind=records[asset['source_id']]['producer_kind'],
                         sidecar_sha256=None,recorder_config_sha256=None,calibration_id=None,type_schema_sha256=None)
        for row in records.values():
            hashes = [w.get('sha256') for w in row['evidence']]
            row.update(sidecar_sha256=hashes[0] if hashes else None,
                       recorder_config_sha256=hashes[1] if len(hashes)>1 else None,
                       calibration_id=(sha256(stable_json(sorted(w['sha256'] for w in row['calibration_refs'])).encode()).hexdigest()
                                       if row['calibration_refs'] else None),
                       type_schema_sha256=None,
                       recorded_interval_ns=[row['interval']['start_ns'],row['interval']['stop_ns']])

    def run(self):
        for root in self.config['raw_roots']:
            print(f"P01-A inventory {root['mode']} {root['path']}",file=sys.stderr,flush=True)
            if root['mode']=='legacy':
                self.scan_legacy(root)
            else:
                self.scan_native(root)
        aliases=record_aliases(self.records)
        self.candidate_relations()
        by_sha=defaultdict(list)
        for a in self.assets:
            if a['file_sha256']:
                by_sha[a['file_sha256']].append(a)
        for group in by_sha.values():
            ordered=sorted(group,key=lambda a:(a['source_id'] in aliases,a['id']))
            if len(ordered)>1:
                for a in ordered[1:]:
                    a.update(status='DUPLICATE_ALIAS',admission_reason='IDENTICAL_MCAP_BYTES',alias_of=ordered[0]['id'])
        lookup=defaultdict(set)
        parents={}
        for r in self.records:
            canonical=aliases.get(r['id'],r['id'])
            if r.get('episode_ref'):
                lookup[r['episode_ref']].add(canonical)
            known=(r['producer_kind']=='LIVE_TELEOP' and not r.get('alias_of') and
                   r['checksum_state']=='VERIFIED' and not set(r['flags']).difference({'CALIBRATION_REFERENCE_PENDING'}))
            capture=stable_id('capture',[r['episode_ref'],r['collector_instance'],r['collector_interval_ns']]) if known else None
            parents[r['id']]=([canonical] if canonical!=r['id'] else [capture] if capture else ['unverified:'+r['id']])
            if capture:
                parents[capture]=[]
            r['root_episode_id']=capture
            r['split']='DEVELOPMENT_CANDIDATE' if known else 'UNASSIGNED'
            if r['status'] in {'QUARANTINED','PENDING_DECODE_VERIFICATION'}:
                self.pending.append(dict(source_id=r['id'],reason=r['admission_reason'],flags=r['flags']))
            r['capabilities']['human_status']=r['status']
        refs={k:next(iter(values)) for k,values in lookup.items() if len(values)==1}
        print('P01-A historical metadata and lineage',file=sys.stderr,flush=True)
        events,hist=self.historical(parents,refs)
        graph=Lineage(parents)
        try:
            for node in parents:
                graph.resolve(node)
        except ValueError as exc:
            self.errors.append(dict(scope='lineage',reason=str(exc)))
            raise
        uses=root_use(events,graph)
        roots=sorted({r['root_episode_id'] for r in self.records if r['root_episode_id']})
        for root in roots:
            if not uses.get(root):
                events.append(dict(schema='historical_usage.v1', source_ref=root, role='USAGE_UNKNOWN',
                                   confidence='HISTORY_INCOMPLETE', interval={}, evidence_field='/'))
        by_id = {row['id']:row for row in self.records}
        self.add_usage_index(events,graph,by_id)
        self.add_source_index(by_id)
        root_views = {row['root_episode_id']:row for row in self.records if row['root_episode_id']}
        lineage=[]
        for node,ps in sorted(parents.items()):
            resolved,unknown=graph.resolve(node)
            source = by_id.get(node) or root_views.get(node) or {}
            lineage.append(dict(schema='episode_lineage.v1',episode_id=node,
                root_episode_id=resolved[0] if len(resolved)==1 and not unknown else None,
                root_episode_ids=resolved,parent_ids=ps,unresolved_refs=unknown,
                relationship=('EXACT_COPY' if node in aliases else 'METADATA_PARENT' if ps else 'CAPTURE_ROOT'),
                adjudication='PENDING' if unknown else 'METADATA_WITNESSED',
                usage_roles=sorted({role for root in resolved for role in uses.get(root,['USAGE_UNKNOWN'])}),
                final_attestation_id=None,split=source.get('split','UNASSIGNED'),
                source_asset_ids=source.get('source_asset_ids',[]),
                evidence_ids=[item['sha256'] for item in source.get('evidence',()) if item.get('sha256')]))
        records=sorted(self.records,key=lambda x:x['id'])
        objects=sorted(records+self.assets+self.artifacts,key=lambda x:(x['record_type'],x['id']))
        for r in records:
            r.pop('signature',None)
            r.pop('priority',None)
        status_counts={kind:{s:sum(x['record_type']==kind and x['status']==s for x in objects) for s in STATUSES}
                       for kind in ('recording','asset','historical_artifact','archive_container')}
        assert sum(sum(c.values()) for c in status_counts.values())==len(objects)
        origins=dict(sorted(Counter(r['producer_kind'] for r in records if not r.get('alias_of')).items()))
        slots=Counter((a['source_id'],a['role']) for a in self.assets)
        counts=dict(recording_directories=sum(r['recording_directories'] for r in self.discovery),
                    mcap_instances=len(self.assets),
                    direct_mcap_instances=sum(a.get('storage')!='ZIP_MEMBER' for a in self.assets),
                    archived_mcap_instances=sum(a.get('storage')=='ZIP_MEMBER' for a in self.assets),
                    unique_byte_payloads=len(by_sha),
                    multi_part_recordings=sum(r['part_count']>1 for r in records),
                    parts_in_multi_part_recordings=sum(r['part_count'] for r in records if r['part_count']>1),
                    writer_split_groups=sum(n>1 for n in slots.values()),
                    pieces_in_writer_split_groups=sum(n for n in slots.values() if n>1),
                    logical_recording_views=len(records), canonical_recordings=len(records)-len(aliases),
                    independent_capture_roots=len(roots), raw_teleop_metadata_candidates=len(roots),
                    fully_admitted_metadata_recordings=status_counts['recording']['ADMITTED_CANDIDATE'],
                    by_recording_origin=origins, status_counts=status_counts,
                    historical_documents=sum(r['record_type']=='historical_artifact' for r in self.artifacts),
                    unresolved_usage_events=sum(bool(x['unresolved_refs']) for x in events))
        gate='PASS' if counts['fully_admitted_metadata_recordings'] and not self.errors else 'BLOCKED'
        inventory=dict(schema='s05.p01a.inventory.v1',counts=counts,discovery=self.discovery,
                       history=hist,errors=self.errors,gate_A1=gate,
                       gate_scope='source metadata and explicit disposition only; CDR, semantic content and splits are separate gates',
                       io=dict(unique_inode_bytes_hashed=sum(key[2] for key in self.cache),
                               archive_member_bytes_hashed=self.archive_hashed_bytes))
        return {'registry.jsonl':objects,'lineage.jsonl':lineage,
                'prior_use.jsonl':sorted(events,key=lambda x: (x['source_ref'],x.get('artifact_id',''),x['evidence_field'],x['role'])),
                'rejects.jsonl':[dict(id=x['id'],record_type=x['record_type'],status=x['status'],
                                      reason=x['admission_reason'],alias_of=x.get('alias_of'))
                                 for x in objects if x['status']!='ADMITTED_CANDIDATE'],
                'pending_verification.jsonl':sorted(self.pending,key=lambda x:json.dumps(x,sort_keys=True)),
                'source_inventory.json':inventory,'dataset_coverage.json':coverage(records),
                'telemetry_candidates.json':dict(schema='s05.telemetry-candidates.v1',system_id_certified=0,
                    rows=[dict(record_id=r['id'],status=r['status'],**r['capabilities']) for r in records]),
                'final_test_eligibility.json':final_test(roots,uses)}
