"""Seal one finished P01-C attempt without changing any previous products."""
import argparse
from collections import Counter
from datetime import datetime
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path('docs/s0-5')
SOURCE = Path('source/dexgym/tasks/direct/wholebody_s05/data')
MODULES = ('geometry', 'calibration', 'clock_domains', 'sample_identity', 'synchronize',
           'capture_windows', 'sync_evidence', 'sync_audit')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def create(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def process_status(path, step):
    exit_file = path / f'{step}.exit'
    if not exit_file.is_file():
        return {'process_exit': None, 'status': 'NOT_RUN'}
    status = int(exit_file.read_text().strip())
    stderr = path / f'{step}.stderr'
    log = stderr.read_text(encoding='utf-8')
    counts = re.findall(r'^Ran (\d+) tests? in ', log, re.M)
    skipped = re.findall(r'skipped=(\d+)', log)
    return {
        'process_exit': status,
        'status': 'PASS' if status == 0 else 'FAIL',
        'total_tests': int(counts[-1]) if counts else None,
        'skipped': int(skipped[-1]) if skipped else 0,
        'command': (path / f'{step}.command').read_text().strip(),
        'evidence_sha256': {f'{step}.{ext}': sha(path / f'{step}.{ext}')
                            for ext in ('command', 'exit', 'stdout', 'stderr')},
        'failure_excerpt': log.splitlines()[-18:] if status else [],
    }


def seal(attempt):
    root = attempt.parent
    products = attempt / 'products'
    sealed = products / 'output_sha256.json'
    test_file = products / 'test_results.json'
    receipt = attempt / 'review_receipt.json'
    if any(path.exists() for path in (sealed, test_file, receipt)):
        raise ValueError('This attempt already contains seal artifacts.')
    required = (
        'geometry_contract.json', 'manus_topology_report.json', 'coordinate_frame_report.json',
        'calibration_report.json', 'clock_domain_registry.json', 'clock_quality.json',
        'clock_mapping_report.json', 'sample_identity_report.json', 'duplicate_source_report.json',
        'aligned_offline_sample_manifest.json', 'aligned_causal_sample_manifest.json',
        'source_trace.jsonl', 'prefix_mutation_results.json', 'validity_report.json',
        'invalid_rows.jsonl', 'sync_error_budget.json', 'qc_threshold_candidates.json',
        'dataset_coverage.json', 'fixtures_manifest.json', 'technical_gates.json',
    )
    coverage = read(products / 'dataset_coverage.json')
    gates = read(products / 'technical_gates.json')
    attempts = {
        path.name: {step: process_status(path, step) for step in ('unit', 'audit', 'integration')}
        for path in sorted(root.glob('attempt-*')) if path.is_dir()
    }
    current = attempts[attempt.name]
    verified = all(row['process_exit'] == 0 for row in current.values())
    executed = all(current[step].get('total_tests') and current[step]['skipped'] == 0
                   for step in ('unit', 'integration'))
    if not executed and verified:
        raise ValueError('A test process exited successfully without executing its required tests.')
    if not all((products / name).is_file() for name in required):
        raise ValueError('Required bounded output missing.')
    technical = {k: gates[k] for k in ('C1', 'C2_algorithm', 'C2_real_clock')}
    technical['C2'] = ('FAIL' if gates['C2_algorithm'] == 'FAIL' else
                       'PASS' if gates['C2_algorithm'] == gates['C2_real_clock'] == 'PASS' else 'BLOCKED')
    state = ('FAILED' if not verified or 'FAIL' in technical.values() else
             'READY_FOR_REVIEW' if all(v == 'PASS' for v in technical.values()) else 'BLOCKED')
    create(test_file, {'schema': 's05.p01c.test-results.v1', 'attempts': attempts, 'technical_status': state,
                       'process_exit_does_not_grant_technical_pass': True, 'manual_acceptance': False})
    hashes = {file.name: sha(file) for file in sorted(products.iterdir()) if file.is_file()}
    create(sealed, hashes)
    if not all(sha(products / name) == value for name, value in hashes.items()):
        raise ValueError('Product bytes changed while sealing.')
    tracked = [SOURCE / f'{module}.py' for module in MODULES]
    tracked.extend(Path(name) for name in (
        'tests/s0_5/unit/test_p01c_geometry_sync.py',
        'tests/s0_5/integration/test_p01c_real_sync.py',
        'docs/s0-5/evidence/P01_C/run_p01_c.sh',
        'docs/s0-5/evidence/P01_C/seal_p01_c.py',
    ))
    spec = ROOT / 'specs/P01_C_calibration_and_causal_sync.md'
    plan = ROOT / 'evidence/P01_C/sample_plan.json'
    identity = Counter()
    for row in read(products / 'duplicate_source_report.json')['rows']:
        for stream, counts in row['stream_identity'].items():
            for reason, n in counts.items():
                identity[(stream, reason)] += n
    brief = {
        'schema': 's05.p01c.review-receipt.v1',
        'phase': 'P01-C',
        'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'run_id': root.name,
        'attempt': attempt.name,
        'source_baseline': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'execution_source_sha256': {str(path): sha(path) for path in tracked},
        'historical_output_counts_verified': gates['history_sha_reverified'],
        'path': str(products),
        'products_total': len(hashes),
        'products_manifest_sha256': sha(sealed),
        'key_product_sha256': {name: hashes[name] for name in (*required, 'test_results.json')},
        'processes': current,
        'earlier_attempts': {name: value for name, value in attempts.items() if name != attempt.name},
        'technical_gates': technical,
        'phase_status': state,
        'sample_plan_sha256': sha(plan),
        'implementation_spec_sha256': sha(spec),
        'resource_preflight_sha256': sha(root / 'resource_preflight.json'),
        'coverage': coverage,
        'validity_reasons': read(products / 'validity_report.json')['reasons'],
        'observer_identity_by_window_aggregate': {f'{key[0]}:{key[1]}': n for key, n in sorted(identity.items())},
        'eligibility': {'real_verified': False, 'simulated_diagnostic_only': True, 'Final_Test': 'BLOCKED',
                        'authorizes_next_phase': False, 'physical_robot_motion': False},
    }
    create(receipt, brief)
    print(f'{receipt} {sha(receipt)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    seal(parser.parse_args().attempt.resolve())
