"""Matched M1 seed identities, static audit and denominator controls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel


SCRIPT = Path(__file__).resolve().parents[3]/'docs/s0-5/evidence/P02/final_handoff/m1_pairing.py'
spec = importlib.util.spec_from_file_location('p02_m1_pairing', SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class M1PairingTests(unittest.TestCase):
    def test_seed_arms_match_and_hand_clips_are_enumerated(self):
        bounds = np.tile([-.2, .2], (54, 1))
        fake = SimpleNamespace(bounds=bounds, names=[f'q{i}' for i in range(54)])
        sdk = np.zeros(40)
        sdk[[0, 7]] = [.5, -.3]
        fixed, hold = study.hold_sdk_hands(fake, sdk)
        self.assertEqual([x['index'] for x in hold['clips']], [0, 7])
        self.assertEqual([x['bound'] for x in hold['clips']], ['upper', 'lower'])
        raw = [np.r_[np.arange(14)*step, hand] for step, hand in
               ((.001, fixed), (.002, np.ones(40)*.01), (.003, np.ones(40)*.02))]
        attempts = [dict(label=label, input_initial=q.tolist()) for label, q in zip(study.START_LABELS, raw, strict=True)]
        original = {'whole54': {'attempts': attempts}}
        m1 = {'multitask': {'attempts': attempts}}
        locked, open_ = study.paired_starts(original, m1, fixed)
        for (_, a), (_, b), q in zip(locked, open_, raw, strict=True):
            self.assertTrue(study.bitwise_match(a[:14], b[:14]))
            self.assertTrue(study.bitwise_match(a[14:], fixed))
            self.assertTrue(study.bitwise_match(b, q))
        m1['multitask']['attempts'] = [*attempts[:2], attempts[2] | {'input_initial': np.zeros(54).tolist()}]
        with self.assertRaisesRegex(ValueError, 'Perturbed-home'):
            study.paired_starts(original, m1, fixed)

    def test_signed_zero_change_is_rejected_during_solver_state_access(self):
        guard = study.GuardedRobot.__new__(study.GuardedRobot)
        fixed = np.zeros(40)
        state = np.zeros(54)
        changed = state.copy()
        changed[14] = -0.
        self.assertFalse(study.bitwise_match(state[14:], changed[14:]))
        with patch.object(RobotModel, 'set', autospec=True) as base:
            with guard.fixed(fixed):
                guard.set(state)
                self.assertEqual(guard.checked_states, 1)
                with self.assertRaisesRegex(ValueError, 'changed a bit'):
                    guard.set(changed)
                base.assert_called_once()
            guard.set(changed)
            self.assertEqual(base.call_count, 2)

    def test_recorded_witness_hand_change_is_rejected(self):
        q = np.zeros(54)
        start = [(study.START_LABELS[0], q)]
        attempt = dict(label=study.START_LABELS[0], input_initial=q.tolist(),
                       initial=q.tolist(), best_q=q.tolist())
        solution = dict(q=q.tolist(), attempts=[attempt])
        study.verify_frozen_attempts(solution, start, q[14:])
        solution['attempts'][0]['best_q'][14] = .001
        with self.assertRaisesRegex(ValueError, 'best_q'):
            study.verify_frozen_attempts(solution, start, q[14:])

    def test_unknown_counts_in_full_threshold_denominator_only(self):
        row = dict(identity=dict(split='Train', cohort='ROOT'),
                   paired_delta_54_minus_14={'max_tip_m': -.004},
                   results={'arms14': dict(status='LEGAL', scalar_metrics=dict(max_tip_m=.007)),
                            'whole54': dict(status='LEGAL', scalar_metrics=dict(max_tip_m=.003))})
        failed = dict(identity=dict(split='Validation', cohort='TAIL'),
                      results={mode: dict(status='UNKNOWN_INPUT', q=None) for mode in study.MODES})
        all_ = study.summarize([row, failed])['All']
        self.assertEqual((all_['planned'], all_['strictly_paired']), (2, 1))
        self.assertEqual(all_['modes']['whole54']['qualified_denominator'], 2)
        self.assertEqual(all_['modes']['whole54']['hits_mm']['3'], 1)
        self.assertEqual(all_['modes']['whole54']['metrics']['max_tip_m']['n'], 1)
        self.assertEqual(all_['paired_metrics']['max_tip_m']['improves'], 1)


if __name__ == '__main__':
    unittest.main()
