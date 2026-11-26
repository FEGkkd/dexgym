"""Publish a small, hashed Git evidence set from an already verified E0 trial."""
import json
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return sha256_file(stream)


def sha256_file(stream):
    h = sha256()
    for block in iter(lambda: stream.read(2**20), b''):
        h.update(block)
    return h.hexdigest()


def write(path, record):
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(record, indent=2, sort_keys=True) + '\n')


def seal(step):
    result = json.loads((step / 'verification.json').read_bytes())
    run = step.parent
    assert result['scope'] == 'E0_TRANSIENT_NON_PRODUCTION'
    assert result['E1'] == 'NOT_RUN' and not result['approved_for_full']
    receipts = {}
    for attempt in sorted(run.glob('attempt-*')):
        if not attempt.is_dir():
            continue
        commands = {}
        for exit_file in sorted(attempt.glob('*.exit')):
            name = exit_file.stem
            commands[name] = {
                'exit': int(exit_file.read_text()),
                'argv_shell': (attempt / (name+'.command')).read_text().strip(),
                'stdout_sha256': digest(attempt / (name+'.stdout')),
                'stderr_sha256': digest(attempt / (name+'.stderr')),
            }
        receipts[attempt.name] = commands
    expected = dict(unit=0, prepare=0, interrupted=1, resume=0, idempotent=0, integration=0)
    assert {k:v['exit'] for k,v in receipts[step.name].items()} == expected
    index = json.loads((step/'trial_products/build_index.json').read_bytes())
    assert index['execution_candidate_sha256'] == result['candidate_config_sha256']
    files = ('candidate/candidate_config.json', 'candidate/execution_lock.json', 'verification.json',
             'trial_products/build_index.json', 'trial_products/build_log.jsonl',
             'trial_products/.session-2.json', 'trial_products/.session-3.json')
    evidence = {}
    for path in [step/name for name in files] + [run/name for name in
            ('fixture_plan.json', 'freeze.sha256', 'frozen_at.txt', 'prerequisite_verification.json')]:
        dst = HERE / path.name
        assert not dst.exists()
        shutil.copyfile(path, dst)
        evidence[dst.name] = digest(dst)
    assert evidence['candidate_config.json'] == result['candidate_config_sha256']
    assert evidence['execution_lock.json'] == result['implementation_lock_sha256']
    write(HERE / 'test_results.json', dict(schema='s05.p01e.e0_commands.v1',
        successful_attempt=step.name, controlled_expected_exit={'interrupted': 1},
        commands=receipts))
    artifacts = {str(f.relative_to(run)):digest(f) for f in sorted(run.rglob('*'))
                 if f.is_file() and f.name != '.run.lock' and not f.name.endswith('.lock')}
    write(HERE/'output_sha256.json', dict(schema='s05.p01e.e0_outputs.v1', root=str(run), files=artifacts))
    for name in ('test_results.json', 'output_sha256.json'):
        evidence[name] = digest(HERE/name)
    config = json.loads((HERE/'candidate_config.json').read_bytes())
    lock = json.loads((HERE/'execution_lock.json').read_bytes())
    receipt = dict(schema='s05.p01e.e0_receipt.v1', scope=result['scope'],
        run_id=run.name, attempt=step.name, output=str(step/'trial_products'),
        executed_git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        bound_implementation_sha256=lock['source_sha256'],
        original_d_config_sha256=config['parent_d_config_sha256'],
        original_d_receipt_sha256=config['parent_d_receipt_sha256'],
        candidate_config_sha256=result['candidate_config_sha256'],
        execution_lock_sha256=result['implementation_lock_sha256'],
        technical_gates={'E0':'PASS','E1':'NOT_RUN'}, approval='AWAITING_HUMAN',
        repository_evidence_sha256=evidence)
    write(HERE/'review_receipt.json',receipt)


if __name__ == '__main__':
    seal(Path(sys.argv[1]).resolve())
