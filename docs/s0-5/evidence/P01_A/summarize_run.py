"""Create a compact P01-A review receipt without copying raw recording indexes."""
from collections import Counter
from datetime import datetime
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


def read_json(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def summarize(run):
    registry = run/'registry'
    spec = Path('docs/s0-5/specs/P01_teleop_data.md')
    receipt = read_json(registry/'output_sha256.json')
    for name, expected in receipt.items():
        if digest(registry/name) != expected:
            raise ValueError(f'Published output hash mismatch: {name}')
    test = read_json(run/'test_results.json')
    inventory = read_json(registry/'source_inventory.json')
    coverage = read_json(registry/'dataset_coverage.json')
    coverage.pop('per_root')
    timing = read_json(run/'timing_source.json')
    runtime = read_json(run/'resource_preflight.json')
    env = runtime['result']
    memory = Path('/proc/meminfo').read_text().splitlines()
    mem_available = next(int(line.split()[1])*1024 for line in memory if line.startswith('MemAvailable:'))
    uses = list(rows(registry/'prior_use.jsonl'))
    pending = list(rows(registry/'pending_verification.jsonl'))
    recordings = [r for r in rows(registry/'registry.jsonl') if r['record_type']=='recording']
    original = [r for r in recordings if r.get('root_episode_id')]
    roots_with_role = {}
    for event in uses:
        roots_with_role.setdefault(event['role'],set()).update(event['root_episode_ids'])
    unchanged = {}
    for root in (Path('/home/ubuntu/dexgym'),Path('/home/ubuntu/teleop_ws')):
        cmd = ['git','-C',str(root),'diff','--binary','HEAD']
        from hashlib import sha256
        current = sha256(subprocess.check_output(cmd)).hexdigest()
        initial = env['repositories'][str(root)]['tracked_diff_sha256']['stdout'].strip()
        unchanged[str(root)] = {'before':initial,'after':current,'matches':initial==current}
    return {
        'schema':'s05.p01a.review-receipt.v1',
        'run_id':run.name,'absolute_run_directory':str(run),
        'observed':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'reviewed_parent':env['repositories'][env['cwd']]['head']['stdout'].strip(),
        'approved_spec_sha256':digest(spec),'scope':'P01-A only',
        'output_sha256':receipt,
        'external_evidence_sha256':{p:digest(run/p) for p in
                                    ('timing_source.json','test_results.json','runner01.log')},
        'inventory':inventory,'coverage':coverage,
        'raw_root_flags':dict(Counter(flag for row in original for flag in row['flags'])),
        'launch_calibration_references':sum(bool(r['calibration_refs']) for r in original),
        'pending_by_reason':dict(Counter(row['reason'] for row in pending)),
        'usage_by_role':dict(Counter(row['role'] for row in uses)),
        'unique_known_roots_by_use':{k:len(v) for k,v in sorted(roots_with_role.items())},
        'usage_confidence':dict(Counter(row['confidence'] for row in uses)),
        'telemetry_by_recording_view':dict(Counter(row['telemetry_status'] for row in
                                                   read_json(registry/'telemetry_candidates.json')['rows'])),
        'final_test_status':read_json(registry/'final_test_eligibility.json')['status'],
        'runtime':{'observed':runtime['observed'],'python':env['python'],'executable':env['executable'],
                   'distributions':env['distributions'],'sys_path':env['sys_path'],
                   'overlay_setup_files':env['overlay_setup_files'],'output':env['output'],
                   'foreign':env['foreign'],'wrong_dexgym':env['wrong_dexgym'],
                   'memory_available_bytes_at_receipt':mem_available,'tracked_external_diffs':unchanged,
                   'message_versions':{k:{'prefix':v.get('prefix'),
                        'manifest_sha256':(v.get('manifest') or {}).get('sha256'),
                        'message_sha256':{Path(d['path']).name:d.get('sha256') for d in v.get('definitions',[])}}
                       for k,v in env['interfaces'].items()},
                   'core_paths':{k:v['file'] for k,v in env['imports'].items() if 'file' in v}},
        'timing_sample_stats':{side:{clock:{k:values[k] for k in ('count','repeats','step_p50')}
                                    for clock,values in fields.items()}
                               for side,fields in timing['sample_stats'].items()},
        'tests':{'unit':test['unit_before_inventory']['exit'],'registry':test['registry_exit'],
                 'integration':test['real_integration']['exit']},
    }


if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])),ensure_ascii=False,indent=2,sort_keys=True))
