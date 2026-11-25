"""P01-A metadata registry. Reads source bytes for checksums, never decodes CDR."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import shutil
import sys
from time import monotonic
from zoneinfo import ZoneInfo

from .lineage import Lineage, exact_aliases
from .historical_usage import root_use

STATUSES = ('ADMITTED_CANDIDATE', 'EXCLUDED', 'QUARANTINED', 'DUPLICATE_ALIAS',
            'PENDING_DECODE_VERIFICATION')
HUMAN_TOPICS = {
    'manus_left': ('/manus_glove_left',), 'manus_right': ('/manus_glove_right',),
    'tracker_left': ('/vive/tracker/left_wrist', '/vive/raw/left_wrist'),
    'tracker_right': ('/vive/tracker/right_wrist', '/vive/raw/right_wrist'),
}
TELEMETRY_TOPICS = {
    'arm_command': ('/tianji/joint_command',),
    'hand_command_left': ('/left_hand/command',), 'hand_command_right': ('/right_hand/command',),
    'arm_feedback_left': ('/left_arm/joint_states',), 'arm_feedback_right': ('/right_arm/joint_states',),
    'hand_feedback_left': ('/left_hand/state',), 'hand_feedback_right': ('/right_hand/state',),
    'diagnostics': ('/tianji/teleop_diagnostics', '/left_hand/diagnostics', '/right_hand/diagnostics'),
    'safety': ('/tianji/safety','/tianji_real_driver/safety','/vive/tracker/safety_status'),
}
PART = re.compile(r'^(.*)_(\d+)\.mcap$', re.I)


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def stable_id(kind, value):
    return f'{kind}:{sha256(stable_json(value).encode()).hexdigest()[:24]}'


def stat_key(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def same_bytes(path, cache=None):
    """Full SHA-256 with a per-run inode cache; reject files that move during hashing."""
    path = Path(path)
    real = path.resolve(strict=True)
    key = stat_key(real.stat())
    if not real.is_file():
        raise ValueError(f'not a regular file: {path}')
    if cache is not None and key in cache:
        digest = cache[key]
    else:
        with real.open('rb') as f:
            if stat_key(os.fstat(f.fileno())) != key:
                raise OSError(f'file changed before read: {path}')
            digest = file_digest(f, 'sha256').hexdigest()
            if stat_key(os.fstat(f.fileno())) != key:
                raise OSError(f'file changed during read: {path}')
        if cache is not None:
            cache[key] = digest
    if stat_key(path.stat()) != key:
        raise OSError(f'file replaced after read: {path}')
    return dict(sha256=digest, size_bytes=key[2], physical_identity=f'{key[0]}:{key[1]}',
                canonical_uri=str(real), mtime_ns=key[3])


def legacy_groups(paths, declared=()):
    """Honor split order in the sidecar, then inventory extra parts by numbered stem."""
    discovered = set(paths)
    groups = []
    if declared:
        ordered = list(dict.fromkeys(declared))
        groups.append(dict(parts=ordered, missing=[p for p in ordered if p not in discovered],
                           order_source='sidecar'))
        discovered.difference_update(ordered)
    by_stem = defaultdict(list)
    for path in sorted(discovered):
        match = PART.fullmatch(path)
        stem, number = (match.group(1), int(match.group(2))) if match else (path, 0)
        by_stem[stem].append((number,path))
    groups.extend(dict(parts=[p for _,p in sorted(rows)], missing=[], order_source='filename')
                  for _,rows in sorted(by_stem.items()))
    return groups


def classify(origin, complete, flags):
    if origin in {'EGOPRO','SIM_REPLAY','DERIVED_ONLY'}:
        return 'EXCLUDED', origin
    if origin != 'LIVE_TELEOP':
        return 'QUARANTINED', 'UNVERIFIED_CAPTURE_ORIGIN'
    if not complete or flags:
        return 'PENDING_DECODE_VERIFICATION', 'INCOMPLETE_METADATA_OR_LINEAGE'
    return 'ADMITTED_CANDIDATE', 'LIVE_CAPTURE_METADATA_AND_REQUIRED_STREAMS'


def capability(catalogue, origin, hardware, calibration, has_time, hmb_status):
    topics = {row['topic'] for row in catalogue if row['count'] is None or row['count'] > 0}
    human = {name: bool(topics.intersection(options)) for name,options in HUMAN_TOPICS.items()}
    robot = {name: bool(topics.intersection(options)) for name,options in TELEMETRY_TOPICS.items()}
    feedback = all(robot[f'{kind}_feedback_{side}'] for kind in ('arm','hand') for side in ('left','right'))
    commands = all(robot[k] for k in ('arm_command','hand_command_left','hand_command_right'))
    telemetry = 'NO_COMMAND_FEEDBACK_PAIR'
    if commands and feedback:
        telemetry = ('PENDING_FIELD_AND_CLOCK_AUDIT' if hardware and origin=='LIVE_TELEOP'
                     else 'PENDING_HARDWARE_PROVENANCE')
    if origin in {'SIM_REPLAY','EGOPRO','DERIVED_ONLY'}:
        telemetry = 'INELIGIBLE_SOURCE'
    return dict(human_topics=human, human_status=hmb_status,
                raw_recording_attested=origin=='LIVE_TELEOP', calibration_reference=bool(calibration),
                basic_record_time=has_time, robot_topic_capability=robot,
                hardware_profile=hardware, telemetry_status=telemetry,
                system_id_timing='UNKNOWN', system_id_certified=False)


def final_test(roots, use):
    return dict(schema='s05.final-test-eligibility.v1', status='BLOCKED', eligible_root_ids=[],
                reason='NO_INDEPENDENT_ATTESTATION_OR_SEAL',
                development_candidates=[dict(root_episode_id=x, historical_roles=use.get(x,['USAGE_UNKNOWN']),
                                             final_test='INELIGIBLE_WITHOUT_INDEPENDENCE_PROOF')
                                        for x in sorted(roots)],
                next_gate='P09 requires independent sources, sealed manifest and access log')


def record_aliases(records):
    aliases = exact_aliases(records)
    for row in records:
        if row['id'] in aliases:
            row.update(status='DUPLICATE_ALIAS', admission_reason='EXACT_ORDERED_MCAP_COPY',
                       alias_of=aliases[row['id']])
    return aliases


def coverage(records):
    candidates = [r for r in records if r['root_episode_id']]
    count = lambda f: dict(sorted(Counter(str(f(r) or 'UNKNOWN') for r in candidates).items()))
    spans = [r['duration_ns'] for r in candidates if r.get('duration_ns') is not None]
    return dict(confirmed_capture_roots=len(candidates),
                admitted_metadata_candidates=sum(r['status']=='ADMITTED_CANDIDATE' for r in candidates),
                eligible_final_test=0,
                sum_recorded_duration_ns=sum(spans), duration_known=len(spans),
                recording_dates=count(lambda r:r['capture_date']),
                capture_sessions=count(lambda r:r['session_id']), task_types=count(lambda r:r['task']),
                human_topics={key:sum(r['capabilities']['human_topics'][key] for r in candidates)
                              for key in HUMAN_TOPICS},
                per_root=[dict(root_episode_id=r['root_episode_id'], record_id=r['id'],
                               duration_ns=r['duration_ns'], date=r['capture_date'], status=r['status'],
                               session_id=r['session_id'], task=r['task'],
                               human_topics=r['capabilities']['human_topics']) for r in candidates],
                duration_semantics='metadata recording duration, not decoded-quality duration')


def publish(destination, products):
    """Write complete immutable output for one Run ID, then atomically publish."""
    destination = Path(destination)
    parent = destination.parent
    stage = parent / ('.' + destination.name + '.staging')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    stage.mkdir()
    checksums = {}
    for name,value in sorted(products.items()):
        path = stage / name
        with path.open('xb') as f:
            if name.endswith('.jsonl'):
                for row in value:
                    f.write((stable_json(row)+'\n').encode())
            else:
                f.write((stable_json(value)+'\n').encode())
            f.flush()
            os.fsync(f.fileno())
        checksums[name] = same_bytes(path)['sha256']
    with (stage/'output_sha256.json').open('xb') as f:
        f.write((stable_json(checksums)+'\n').encode())
        f.flush()
        os.fsync(f.fileno())
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    stage.rename(destination)


def main(argv=None):
    from .inventory import Inventory

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    cli.add_argument('--resource', required=True, type=Path)
    cli.add_argument('--test-results', required=True, type=Path)
    args = cli.parse_args(argv)
    config = json.loads(args.config.read_bytes())
    if not args.output.resolve().is_relative_to(Path(config['output_root']).resolve()):
        cli.error('output must be within the approved P01 root')
    aliases = {str(Path(root['path']).resolve()) for root in config['raw_roots']}
    if any(args.output.resolve().is_relative_to(Path(root)) for root in aliases):
        cli.error('output overlaps a source root')
    resource = json.loads(args.resource.read_bytes())
    tests = json.loads(args.test_results.read_bytes())
    if tests.get('exit')!=0:
        cli.error('unit tests must pass before scanning real sources')
    runtime = resource['result']
    if runtime['foreign'] or runtime['wrong_dexgym'] or not runtime['output']['writable']:
        cli.error('preflight failed to establish an isolated writable runtime')
    for module, expected in config['pinned_runtime_sha256'].items():
        if runtime['imports'].get(module,{}).get('file',{}).get('sha256') != expected:
            cli.error(f'runtime anchor changed: {module}')
    started = monotonic()
    inventory = Inventory(config)
    products = inventory.run()
    products['resource_preflight.json'] = resource
    products['test_results.json'] = tests
    products['source_inventory.json']['execution'] = {
        'arguments': [str(x) for x in sys.argv] if argv is None else list(argv),
        'config_sha256':same_bytes(args.config)['sha256'],
        'module_sha256':{name:same_bytes(Path(__file__).with_name(name))['sha256']
                         for name in ('source_registry.py','inventory.py','lineage.py','historical_usage.py')},
        'seconds_before_write':round(monotonic()-started,3),
        'disk_free_bytes_before_publish':shutil.disk_usage(args.output.parent).free,
    }
    if args.output.exists():
        cli.error('Run ID already published')
    publish(args.output,products)
    print(stable_json({'output':str(args.output),'counts':products['source_inventory.json']['counts'],
                       'gate_A1':products['source_inventory.json']['gate_A1']}))


def date_from_ns(value):
    return datetime.fromtimestamp(value//1_000_000_000,ZoneInfo('Asia/Shanghai')).date().isoformat() if value else None


if __name__ == '__main__':
    main()
