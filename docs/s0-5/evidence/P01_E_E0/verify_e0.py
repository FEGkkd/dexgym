"""Verify the real E0 trial, its recovery ledger, and representative MCAP bytes."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

from mcap.reader import NonSeekingReader

from dexgym.tasks.direct.wholebody_s05.data.hmb_build import json_bytes,read_partition
from dexgym.tasks.direct.wholebody_s05.data.p01e_lock import D_DIR,D_SHA
from dexgym.tasks.direct.wholebody_s05.data.registry_gate import digest,load_rows


def verify(step,a):
    output=step/'trial_products'
    index=json.loads((output/'build_index.json').read_bytes())
    plan=json.loads((step.parent/'fixture_plan.json').read_bytes())
    logs=[json.loads(line) for line in (output/'build_log.jsonl').read_bytes().splitlines()]
    prev=Counter(x['status'] for x in logs)
    assert set(prev)=={'PLANNED','RUNNING','PUBLISHED','FAILED','PAUSED','RESUMED'}
    assert prev['FAILED']==prev['PAUSED']==prev['RESUMED']==1
    failed=next(x for x in logs if x['status']=='FAILED')
    assert failed['evidence']['reason']=='INJECTED_AFTER_SEMANTIC_STAGING' and not failed['evidence']['target_exists']
    assert prev['PUBLISHED']==plan['full_blocks']+1
    assert index['scope']=='E0_TRANSIENT_NON_PRODUCTION' and len(index['partitions'])==prev['PUBLISHED']
    assert all(x['status']=='PUBLISHED' for x in index['partitions'])
    session=json.loads((output/'.session-3.json').read_bytes())
    assert set(session['previous_publications_revalidated'])=={x['partition']['key'] for x in index['partitions']}
    second=json.loads((output/'.session-2.json').read_bytes())
    assert len(second['previous_publications_revalidated'])==1
    lookup={r['id']:r for r in load_rows(a/'registry.jsonl')}
    modes=Counter()
    tests=[]
    for i,row in enumerate(index['partitions']):
        p=row['partition']; e=row['evidence']; dest=output/e['path']
        assert p['index']==i and p['recording_id']==plan['recording_id']
        assert p['start_ns']==(index['partitions'][i-1]['partition']['stop_ns'] if i else plan['start_ns'])
        assert digest(dest/'manifest.json')==e['manifest_sha256']
        for mode in ('causal','offline'):
            vals,meta,sources=read_partition(dest,mode)
            assert len(vals['tick_ns'])==p['planned_ticks']
            assert meta['identity']['publication_scope']==index['scope']
            assert meta['identity']['development_split']=='Train'
            n=int(vals['joint_valid'].sum())
            modes[f'{mode}_joint_valid']+=n
            modes[f'{mode}_joint_invalid']+=len(vals['tick_ns'])-n
            modes[f'{mode}_velocity_valid_stream_ticks']+=int(vals['velocity_valid'].sum())
        if i not in (0,1,len(index['partitions'])-1):
            continue
        candidates=[r for r in sources if r['stream']=='manus_left']
        assert candidates
        ref=candidates[len(candidates)//2]['record']
        asset=Path(lookup[ref['part_id']]['canonical_uri'])
        assert digest(asset)==ref['asset_sha256']
        with asset.open('rb') as stream:
            reader=NonSeekingReader(stream,validate_crcs=True,record_size_limit=32*2**20)
            schema,channel,raw=next(rec for j,rec in enumerate(reader.iter_messages(log_time_order=False))
                                    if j==ref['physical_message'])
        assert sha256(raw.data).hexdigest()==ref['cdr_sha256']
        assert sha256(schema.data).hexdigest()==ref['schema_sha256'] and channel.topic==ref['topic']
        tests.append(dict(block=p['label'],physical_message=ref['physical_message'],asset_sha256=ref['asset_sha256'],
                          cdr_sha256=ref['cdr_sha256'],schema_sha256=ref['schema_sha256']))
    assert index['partitions'][-1]['partition']['stop_ns']==plan['stop_ns']
    tail=index['partitions'][-1]
    assert tail['partition']['tail'] and tail['partition']['stop_ns']-tail['partition']['start_ns']==plan['tail_ns']
    assert all(modes[f'{mode}_joint_valid']+modes[f'{mode}_joint_invalid']==index['planned_ticks']
               for mode in ('causal','offline'))
    candidate_path=step/'candidate/candidate_config.json'
    cfg=json.loads(candidate_path.read_bytes())
    d=json.loads((D_DIR/'frozen_config.json').read_bytes())
    assert cfg['parent_d_config_sha256']==D_SHA
    assert all(cfg[k]==d[k] for k in ('parameters','numerical_qc','registry_sha256','artifact_sha256','allowed_use'))
    assert cfg['approval']=='AWAITING_HUMAN' and cfg['implementation_lock_sha256']==digest(
        step/'candidate/execution_lock.json')
    measured=[x['evidence']['measured'] for x in index['partitions']]
    summary=dict(schema='s05.p01e.e0_result.v1',scope=index['scope'],E1='NOT_RUN',
        attempts=prev,fixture=plan,planned_ticks=index['planned_ticks'],counts=modes,
        terminal_partitions=len(index['partitions']),tail_qualification=tail['evidence']['classification'],
        tail_pair_counts=tail['evidence']['qualification']['pairs'],physical_replays=tests,
        candidate_config_sha256=digest(candidate_path),implementation_lock_sha256=cfg['implementation_lock_sha256'],
        original_d_config_sha256=D_SHA,frozen_semantics_unchanged=True,
        current_peak_rss_bytes=max(m['peak_process_rss_bytes'] for m in measured),
        measured_decoder_messages=sum(m['cdr_messages'] for m in measured),
        measured_decoder_s=sum(m['cdr_seconds'] for m in measured),
        measured_sync_s=sum(m['sync_seconds'] for m in measured),
        published_bytes=sum(m['published_file_bytes'] for m in measured),
        retained_partial_bytes=failed['evidence']['retained_partial_bytes'],
        successful_process_rchar_bytes=sum(m['linux_proc_io_delta']['rchar'] for m in measured),
        minimum_free_bytes=min(m['disk_available_bytes'] for m in measured),approved_for_full=False)
    with (step/'verification.json').open('xb') as f:
        f.write(json_bytes(summary))
    print(json.dumps(summary))


if __name__=='__main__':
    verify(Path(sys.argv[1]),Path(sys.argv[2]))
