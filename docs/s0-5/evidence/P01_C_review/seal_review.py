"""Seal or verify the independent P01-C Review Fix evidence."""
import argparse
from collections import Counter
from datetime import datetime
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo


ROOT=Path('docs/s0-5/evidence/P01_C_review')
BASE='0d9d351b73e543eab07cfe6fef8f55a7d12d912d'
MODULES=('calibration','capture_windows','clock_domains','geometry','sample_identity','synchronize',
         'sync_audit','sync_evidence','review_audit','review_metrics')


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path,data):
    with path.open('x',encoding='utf-8') as output:
        json.dump(data,output,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        output.write('\n')


def step_files(folder,step):
    result={}
    for ext in ('command','stdout','stderr','exit'):
        path=folder/f'{step}.{ext}'
        if path.exists():
            result[path.name]=sha(path)
    code=folder/f'{step}.exit'
    text=(folder/f'{step}.stderr').read_text() if code.is_file() else ''
    match=re.search(r'Ran (\d+) tests? in ',text)
    status={'exit':int(code.read_text()) if code.exists() else None,
            'log_sha256':result,'tests':int(match.group(1)) if match else None,
            'failures':int(re.search(r'failures=(\d+)',text)[1]) if 'failures=' in text else 0,
            'skipped':int(re.search(r'skipped=(\d+)',text)[1]) if 'skipped=' in text else 0}
    cmd=folder/f'{step}.command'
    if cmd.exists():
        status['command']=cmd.read_text().strip()
    return status


def manifest(products):
    return {file.name:sha(file) for file in sorted(products.iterdir())
            if file.is_file() and file.name!='output_sha256.json'}


def seal(attempt):
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=BASE:
        raise ValueError('Incorrect review execution baseline')
    root=attempt.parent
    products=attempt/'products'
    execution={s:step_files(attempt,s) for s in ('unit','legacy_order','paired','integration')}
    if any(r['exit']!=0 or r['skipped'] for r in execution.values()):
        raise ValueError('Required process failed or skipped tests')
    if not (execution['unit']['tests'] and execution['integration']['tests']):
        raise ValueError('Zero-test execution does not qualify')
    previous={p.name:{s:step_files(p,s) for s in ('unit','paired','integration')}
              for p in sorted(root.glob('attempt-*')) if p.is_dir() and p!=attempt}
    neg=step_files(root,'baseline_controls')
    if neg['exit']!=1 or neg['tests']!=7 or neg['failures']!=6:
        raise ValueError('Original counterexamples did not reproduce')
    regression=read(products/'paired_regression.json')
    gate=read(products/'technical_gates.json')
    gates=gate['gates']
    status='READY_FOR_REVIEW' if gates['C1-GEO']==gates['C2-SIM']=='PASS' else 'BLOCKED'
    evidence_files=[
        'C-historical.log','AB-historical.exit','AB-historical.log','freeze.sha256','frozen_at.txt','baseline_controls.py']
    historical={f:sha(root/f) for f in evidence_files}
    suite=dict(execution=execution,original_negative_controls=neg,earlier_attempts=previous)
    write(products/'test_results.json',suite)
    write(products/'output_sha256.json',manifest(products))
    counts=Counter()
    shifts=[]
    for row in regression['results']:
        shifts.append(row['clock']['phase_shift_ns_from_original_ordinal'])
        counts['safety_neighbor_ticks']+=row['masks']['safety_neighborhood']['ticks']
        counts['safety_neighbor_old_joint_valid']+=row['masks']['safety_neighborhood']['old_valid']
        counts['safety_neighbor_new_joint_valid']+=row['masks']['safety_neighborhood']['new_valid']
        for stream,stats in row['masks']['streams'].items():
            for key in ('pose_old','pose_new','velocity_old','velocity_new','mask_lost','mask_gained'):
                counts[f'{stream}:{key}']+=stats[key]
    files=[Path(f'source/dexgym/tasks/direct/wholebody_s05/data/{name}.py') for name in MODULES]
    files.extend(Path(x) for x in ('tests/s0_5/unit/test_p01c_geometry_sync.py',
                                  'tests/s0_5/unit/test_p01c_review_controls.py',
                                  'tests/s0_5/integration/test_p01c_review_integration.py',
                                  f'{ROOT}/run_review.sh',f'{ROOT}/seal_review.py',f'{ROOT}/legacy_order_witness.py'))
    report=dict(schema='s05.p01c.review-fix.v1',phase='P01-C',scope='HMB_SIM_REVIEW',source_baseline=BASE,
                run_id=root.name,attempt=attempt.name,observed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                phase_status=status,technical_gates=gates,verified_control_ticks=0,
                authorizes_any_followup=False,source_products=str(products),tests=suite,
                counts=regression['totals'],paired_observations=dict(counts),
                historical_outputs=regression['baseline_hashes'],preflight_files_sha256=historical,
                products_sha256=read(products/'output_sha256.json'),
                products_manifest_sha256=sha(products/'output_sha256.json'),
                execution_source_sha256={str(path):sha(path) for path in files},
                recorder_clock_max_abs_residual_ns=regression['max_host_clock_pair_residual_ns'],
                old_grid_phase_shift_ns=[min(shifts),max(shifts)])
    write(attempt/'review_receipt.json',report)
    print(attempt/'review_receipt.json',sha(attempt/'review_receipt.json'))


def verify(attempt):
    receipt=ROOT/'review_receipt.json'
    if sha(receipt)!=sha(attempt/'review_receipt.json'):
        raise ValueError('Published receipt differs from the sealed run')
    report=read(receipt)
    products=attempt/'products'
    if sha(products/'output_sha256.json')!=report['products_manifest_sha256']:
        raise ValueError('Review manifest has changed')
    if manifest(products)!=report['products_sha256']:
        raise ValueError('One or more sealed review products have changed')
    if any(sha(Path(path))!=hexdigest for path,hexdigest in report['execution_source_sha256'].items()):
        raise ValueError('A review module changed after execution')
    if any(sha(attempt.parent/path)!=hexdigest for path,hexdigest in report['preflight_files_sha256'].items()):
        raise ValueError('A historical preflight evidence file changed')
    for folder,steps in ((attempt,report['tests']['execution']),
                         (attempt.parent,{'baseline':report['tests']['original_negative_controls']})):
        for status in steps.values():
            if any(sha(folder/file)!=expected for file,expected in status['log_sha256'].items()):
                raise ValueError('Recorded process logs changed')
    print(f"PASS {len(report['products_sha256'])} products plus source, process and preflight hashes")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt',type=Path)
    parser.add_argument('--verify',action='store_true')
    options=parser.parse_args()
    (verify if options.verify else seal)(options.attempt.resolve())
