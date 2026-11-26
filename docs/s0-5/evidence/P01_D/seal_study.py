"""Seal bounded P01-D data findings and write a technical review receipt."""
from argparse import ArgumentParser
from datetime import datetime
from hashlib import file_digest
from pathlib import Path
import json
import re
from zoneinfo import ZoneInfo


DOCS=Path('docs/s0-5')
source=Path('source/dexgym/tasks/direct/wholebody_s05/data')
required=('source_whitelist','source_qualification','qc_threshold_report','qc_sensitivity_results',
          'frozen_config','hmb_schema','reason_code_registry','split','split_isolation_report',
          'dataset_coverage','resource_budget','build_protocol')


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path,data):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def process(path,step):
    files={ext:path/f'{step}.{ext}' for ext in ('command','stdout','stderr','exit')}
    if not files['exit'].is_file():
        return {'exit':None,'status':'NOT_RUN'}
    text=files['stderr'].read_text()
    count=re.search(r'^Ran (\d+) tests? in ',text,re.M)
    exit=int(files['exit'].read_text())
    return dict(exit=exit,tests=int(count[1]) if count else None,skipped='skipped=' in text,
                command=files['command'].read_text().strip(),status='PASS' if exit==0 else 'FAIL',
                hashes={p.name:sha(p) for p in files.values()},fail_tail=text.splitlines()[-14:] if exit else [])


def seal(attempt):
    root=attempt.parent
    out=attempt/'products'
    execs={s:process(attempt,s) for s in ('unit','study','integration')}
    tests={p.name:{s:process(p,s) for s in execs} for p in sorted(root.glob('attempt-*'))
           if p.is_dir() and p!=attempt}
    if any(r['exit']!=0 or r.get('skipped') for r in execs.values()):
        raise ValueError('Required real study or tests did not finish')
    if not (execs['unit']['tests'] and execs['integration']['tests']):
        raise ValueError('No actual tests ran')
    outputs={k:read(out/f'{k}.json') for k in required}
    w=outputs['source_whitelist']
    q=outputs['source_qualification']
    split=outputs['split']
    isolation=outputs['split_isolation_report']
    coverage=outputs['dataset_coverage']
    qc=outputs['qc_threshold_report']
    resource=outputs['resource_budget']
    config=outputs['frozen_config']
    roots={r['root_episode_id'] for r in w['sources']}
    train={r['root_episode_id'] for r in split['Train']}
    val={r['root_episode_id'] for r in split['Validation']}
    health={
        'source_coverage':len(q['rows'])==32 and len({r['root_episode_id'] for r in q['rows']})==32 and
            len(roots)>=3 and roots=={r['root_episode_id'] for r in q['rows'] if r['status']=='QUALIFIED'} and
            all(r.get('issues') for r in q['rows'] if r['status']!='QUALIFIED'),
        'lineage_split':bool(train and val and not train&val and train|val==roots and
                              isolation['overlap_count']==0 and split['Final_Test']['status']=='BLOCKED'),
        'qc_scope':bool(qc['window_results'] and outputs['qc_sensitivity_results']['axis_results'] and
                        not qc['validation_used_for_selection'] and coverage['verified_control_ticks']==0),
        'production_round_trip':bool(resource['pilot']['parallel_equal_sha'] and
                                     resource['pilot']['failed_target_hidden'] and resource['pilot']['idempotent_recovery']),
        'headroom':resource['disk_available_bytes']>=resource['production']['required_peak_including_staging_bytes']+
                    resource['production']['disk_hard_reserve_bytes'],
        'version_identity':all(sha(source/p)==h for p,h in config['implementation_sha256'].items()) and
                           all(sha(out/p)==h for p,h in config['artifact_sha256'].items()),
    }
    gate='PASS' if all(health.values()) else 'BLOCKED'
    write(out/'test_results.json',dict(current=execs,prior_attempts=tests,checks=health,
                                      technical_gates={'D1':gate},manual_approval=False))
    hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'output_sha256.json',hashes)
    receipt=dict(schema='s05.p01d.review-receipt.v1',phase='P01-D',run_id=root.name,attempt=attempt.name,
        observed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        baseline=(root/'baseline.git').read_text().strip(),technical_gates={'D1':gate},
        separate_gates={k:'BLOCKED' for k in ('C2-RT','robot-base','P01-T','Final Test')},
        phase_status='READY_FOR_REVIEW' if gate=='PASS' else 'BLOCKED',human_approval=False,
        candidate_config_sha256=sha(out/'frozen_config.json'),product_manifest_sha256=sha(out/'output_sha256.json'),
        contents={name:hashes[f'{name}.json'] for name in (*required,'test_results')},
        sync_source_sha256=config['sync_source_sha256'],source_scope=q['history'],
        execution_source_sha256={p.name:sha(p) for p in
           [DOCS/'evidence/P01_D/run_study.sh',Path(__file__),
            Path('tests/s0_5/unit/test_p01d_hmb.py'),Path('tests/s0_5/integration/test_p01d_real_hmb.py')]},
        frozen_plan_sha256=sha(DOCS/'evidence/P01_D/sample_plan.json'),source_counts={
            'qualified':len(roots),'blocked':32-len(roots),'train':len(train),'validation':len(val)},
        observed_windows=coverage['windows'],observed_ticks=coverage['observed_ticks'],
        frozen_parameters=config['parameters'],resources=resource,process=execs,
        product_directory=str(out))
    write(attempt/'review_receipt.json',receipt)
    print(attempt/'review_receipt.json',sha(attempt/'review_receipt.json'))
    print(receipt['phase_status'],receipt['source_counts'],receipt['frozen_parameters'])


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('attempt',type=Path)
    seal(p.parse_args().attempt.resolve())
