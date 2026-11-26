"""Replay the P01-C receipt and exercise phase state rejection paths."""
import argparse
from copy import deepcopy
from hashlib import file_digest
import json
from pathlib import Path
import runpy


EVIDENCE = Path('docs/s0-5/evidence')


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f'SHA mismatch: {path}')


def verify(attempt):
    receipt_path = EVIDENCE / 'P01_C/review_receipt.json'
    check_hash(attempt / 'review_receipt.json', sha(receipt_path))
    receipt = json.loads(receipt_path.read_bytes())
    products = attempt / 'products'
    manifest = products / 'output_sha256.json'
    check_hash(manifest, receipt['products_manifest_sha256'])
    hashes = json.loads(manifest.read_bytes())
    if len(hashes) != receipt['products_total'] or set(hashes) != {
            item.name for item in products.iterdir() if item.is_file() and item != manifest}:
        raise ValueError('Product files differ from the sealed manifest')
    for name, digest in hashes.items():
        check_hash(products / name, digest)
    for name, digest in receipt['execution_source_sha256'].items():
        check_hash(Path(name), digest)
    for process in receipt['processes'].values():
        for name, digest in process['evidence_sha256'].items():
            check_hash(attempt / name, digest)
    for figure in (receipt_path.parent).glob('manus-*.png'):
        check_hash(figure, hashes[figure.name])
    print(f'PASS {len(hashes)} sealed products; execution code, process logs and figures reverified')

    checker = runpy.run_path(str(EVIDENCE / 'P00/check_documents.py'))
    document = json.loads(Path('docs/s0-5/phase_state.json').read_bytes())
    checker['phase_states'](document)
    for broken_sha, expected in ((True, 'technical receipt hash mismatch'), (False, 'C2 technical PASS')):
        corrupt = deepcopy(document)
        end = corrupt['p01_subphase_events'][-1]
        end['to'] = 'READY_FOR_REVIEW'
        corrupt['p01_subphases']['P01-C'] = 'READY_FOR_REVIEW'
        if broken_sha:
            end['evidence_sha256'] = '0' * 64
        try:
            checker['phase_states'](corrupt)
        except (AssertionError, ValueError, SystemExit) as exc:
            if expected not in str(exc):
                raise
        else:
            raise ValueError(f'Phase checker accepted invalid state: {expected}')
        print(f'PASS negative control: {expected}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    verify(parser.parse_args().attempt)
