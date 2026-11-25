"""Review checker tests. No production P01 code or runtime dependencies."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest


module_path = Path(__file__).resolve().parents[1] / 'P00/check_documents.py'
spec = importlib.util.spec_from_file_location('s05_document_checks', module_path)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def log():
    states = dict.fromkeys(checks.PHASES, 'NOT_STARTED')
    states['P00'] = 'READY_FOR_REVIEW'
    return dict(schema='s05.phase-state.v1', current=states, events=[
        dict(phase='P00', at='2026-01-01T00:00:00+00:00', evidence_ref='p00-start',
             authorization_ref='explicit-p00-start', **{'from':'NOT_STARTED', 'to':'IN_PROGRESS'}),
        dict(phase='P00', at='2026-01-01T00:00:01+00:00', evidence_ref='p00-evidence',
             **{'from':'IN_PROGRESS','to':'READY_FOR_REVIEW'}),
    ])


def next_event(document, phase, status, **fields):
    source = document['current'][phase]
    document['events'].append(dict(phase=phase, at='2026-01-01T00:00:02+00:00',
                                   evidence_ref='example-review-artifact', **{'from':source,'to':status}, **fields))
    document['current'][phase] = status


class PhaseLogTests(unittest.TestCase):
    def test_initial_review_state(self):
        self.assertEqual(checks.phase_states(log())['P00'], 'READY_FOR_REVIEW')

    def test_human_approval_then_authorized_p01(self):
        document = log()
        next_event(document,'P00','ACCEPTED',approval={'type':'human','by':'reviewer','ref':'approval-record'})
        next_event(document,'P01','IN_PROGRESS',authorization_ref='approval-p01',spec_sha256='1'*64)
        self.assertEqual(checks.phase_states(document)['P01'], 'IN_PROGRESS')

    def test_cannot_self_approve_or_start_p01_early(self):
        document = log()
        next_event(document,'P00','ACCEPTED',approval={'type':'script','by':'tool','ref':'tool-record'})
        with self.assertRaisesRegex(ValueError, 'Human approval'):
            checks.phase_states(document)
        document = log()
        next_event(document,'P01','IN_PROGRESS',authorization_ref='approval-p01',spec_sha256='1'*64)
        with self.assertRaisesRegex(ValueError, 'Previous phase'):
            checks.phase_states(document)

    def test_p09_gate_requires_sealed_final_test(self):
        document = log()
        next_event(document,'P00','ACCEPTED',approval={'type':'human','by':'reviewer','ref':'approval-record'})
        for phase in checks.PHASES[1:9]:
            next_event(document,phase,'IN_PROGRESS',authorization_ref='authorized',spec_sha256='2'*64)
            next_event(document,phase,'READY_FOR_REVIEW')
            next_event(document,phase,'ACCEPTED',approval={'type':'human','by':'reviewer','ref':'approval-record'})
        next_event(document,'P09','IN_PROGRESS',authorization_ref='authorized',spec_sha256='2'*64)
        with self.assertRaisesRegex(ValueError, 'sealed final test'):
            checks.phase_states(document)
        document['events'][-1]['final_test'] = {'status':'FINAL_TEST_SEALED','manifest_sha256':'3'*64,
                                                'access_log_ref':'sealed-access-ledger'}
        self.assertEqual(checks.phase_states(document)['P09'], 'IN_PROGRESS')


class ManifestTests(unittest.TestCase):
    def test_hash_and_missing_reference_negative_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'meta.json'
            path.write_bytes(b'{}')
            manifest = dict(schema='s05.review-evidence.v1',files=[
                dict(path='meta.json',size=2,sha256=checks.digest(b'{}'))])
            self.assertIn('meta.json',checks.checked_manifest(root,manifest))
            wrong_sha = copy.deepcopy(manifest)
            wrong_sha['files'][0]['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'changed evidence'):
                checks.checked_manifest(root,wrong_sha)
            manifest['files'][0]['path'] = 'missing.json'
            with self.assertRaisesRegex(ValueError, 'Missing or external file'):
                checks.checked_manifest(root,manifest)


if __name__ == '__main__':
    unittest.main()
