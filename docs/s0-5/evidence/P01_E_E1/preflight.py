"""Independent raw-interval accounting and locked E1 start checks."""
from collections import Counter
from pathlib import Path
import json
import os
import subprocess
import sys

import yaml

from dexgym.tasks.direct.wholebody_s05.data.hmb_build import json_bytes
from dexgym.tasks.direct.wholebody_s05.data.p01d_freeze import resources
from dexgym.tasks.direct.wholebody_s05.data.p01e_build import E1, plan_blocks
from dexgym.tasks.direct.wholebody_s05.data.p01e_lock import D_DIR, PLAN, authorize, verify_lock
from dexgym.tasks.direct.wholebody_s05.data.registry_gate import digest, load_rows


def check(condition, message):
    if not condition:
        raise ValueError(message)


def put(path, value):
    with path.open('xb') as f:
        f.write(json_bytes(value))
        f.flush()
        os.fsync(f.fileno())


def preflight(run, session, old_a, config, lock):
    authorize(E1, run/'products', digest(config), digest(lock))
    cfg = verify_lock(old_a, config, lock)
    pool = json.loads((D_DIR/'source_whitelist.json').read_bytes())['sources']
    registry = {r['id']: r for r in load_rows(old_a/'registry.jsonl')}
    allocation = json.loads((D_DIR/'split.json').read_bytes())
    roots = {role: {r['root_episode_id'] for r in allocation[role]} for role in ('Train', 'Validation')}
    check(not roots['Train'] & roots['Validation'] and len(pool) == 32, 'Frozen root isolation failed')
    check(not allocation['Final_Test']['roots'] and allocation['Final_Test']['status'] == 'BLOCKED',
          'Unexpected Final Test admission')
    raw, assets = [], {}
    counts = {role: Counter() for role in roots}
    for source in pool:
        row = registry[source['recording_id']]
        manifest = Path(row['canonical_uri'])/'manifest.yaml'
        check(digest(manifest) == source['manifest_sha256'], f'Raw manifest differs: {row["id"]}')
        bag = yaml.safe_load(manifest.read_bytes())
        interval = [bag['start_realtime_ns'], bag['stop_realtime_ns']]
        check(all(type(t) is int and 0 <= t < 2**63 for t in interval) and interval[0] < interval[1] and
              interval == row['recorded_interval_ns'] == source['recorded_interval_ns'],
              f'Raw int64 interval differs: {row["id"]}')
        role, root = source['split'], source['root_episode_id']
        check(root == row['root_episode_id'] and root in roots[role] and row['producer_kind'] == 'LIVE_TELEOP',
              f'Unapproved source: {row["id"]}')
        seconds = interval[1] - interval[0]
        full, tail = divmod(seconds, 8_000_000_000)
        blocks = full + bool(tail)
        ticks = full*400 + (tail+19_999_999)//20_000_000
        counts[role].update(dict(sources=1, partitions=blocks, planned_ticks=ticks, duration_ns=seconds))
        raw.append(dict(recording_id=row['id'], root=root, split=role, task=row['task'], capture_date=row['capture_date'],
                        recorder_config_sha256=row['recorder_config_sha256'],
                        interval_ns=interval, partitions=blocks, planned_ticks=ticks,
                        manifest_sha256=source['manifest_sha256']))
        for values in source['assets'].values():
            for asset in values:
                part_id = asset['id']
                if part_id in assets:
                    check(assets[part_id]['sha256'] == asset['sha256'], f'Conflicting duplicate asset: {part_id}')
                    continue
                path = Path(registry[part_id]['canonical_uri'])
                check(digest(path)==asset['sha256'] and path.stat().st_size==asset['bytes'],
                      f'Frozen MCAP identity changed: {part_id}')
                assets[part_id]=asset
    check(len({r['root'] for r in raw}) == len(pool), 'Duplicate parent in the source plan')
    declared = plan_blocks(pool, json.loads(PLAN.read_bytes()), digest(config), E1)
    for r in raw:
        blocks = [d for d in declared if d['recording_id']==r['recording_id']]
        check(len(blocks)==r['partitions'] and sum(b['planned_ticks'] for b in blocks)==r['planned_ticks'] and
              blocks[0]['start_ns']==r['interval_ns'][0] and blocks[-1]['stop_ns']==r['interval_ns'][1] and
              all(a['stop_ns']==b['start_ns'] for a,b in zip(blocks,blocks[1:])), f'Production plan mismatch: {r["root"]}')
    reference = dict(Train=dict(sources=24,partitions=228,planned_ticks=86338),
                     Validation=dict(sources=8,partitions=80,planned_ticks=30275))
    delta = {role:{metric:counts[role][metric]-value for metric,value in expected.items()}
             for role,expected in reference.items()}
    plan = dict(scope=E1,run_id=run.name,config_sha256=digest(config),lock_sha256=digest(lock),
                source_totals=counts,total=sum(counts.values(),Counter()),review_reference=reference,
                delta_from_reference=delta,sources=raw,partitions=declared,
                asset_count=len(assets),asset_bytes=sum(a['bytes'] for a in assets.values()))
    target = run/'production_plan.json'
    if target.exists():
        check(target.read_bytes()==json_bytes(plan), 'Resume plan differs from original')
    else:
        put(target,plan)
    budget=json.loads((D_DIR/'resource_budget.json').read_bytes())['production']
    capacity=resources(cfg['plan'],budget['required_peak_including_staging_bytes'])
    check(capacity['ram_available_bytes']>=budget['minimum_available_ram_bytes'], 'Frozen production RAM reserve')
    put(run/f'session-{session}/resource_preflight.json',
        dict(resources=capacity,budget=budget,workers=1,verified_plan_sha256=digest(target),
             git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print(json.dumps({'totals':counts,'delta':delta,'partitions':len(declared),'resources':capacity,
                      'plan_sha256':digest(target)},sort_keys=True))


if __name__ == '__main__':
    run, session, old_a, config, lock = sys.argv[1:]
    preflight(Path(run), int(session), Path(old_a), Path(config), Path(lock))
