"""Review regressions for functional statistics, mapping, and minimax."""
import unittest

from seal_p02 import mapping_and_solver


class PinchStatisticsTests(unittest.TestCase):
    def test_active_indices_are_counts_and_positions(self):
        cases = [([0], [10], [1]), ([1], [20], [2]),
                 ([0, 2], [10, 30], [1, 3]), ([], [], [])]
        for indices, before, after in cases:
            with self.subTest(indices=indices):
                hand = {'near_pinch_pairs': indices, 'before_error_m': [10, 20, 30, 40],
                        'after_error_m': [1, 2, 3, 4], 'bone_length_delta_m': [0.]}
                record = dict(identity={'split':'Train','cohort':'ROOT'}, mapped={},
                              mapping={'hands':[hand],'relative_wrist_translation_delta_m':0.})
                summary = mapping_and_solver([record],[])['All']
                self.assertEqual(summary['near_pinch_pairs'], len(indices))
                self.assertEqual(summary['active_pinch_before_m'] is None, not before)
                self.assertEqual(summary['active_pinch_after_m'] is None, not after)
                if before:
                    self.assertEqual(summary['active_pinch_before_m'][-1], max(before))
                    self.assertEqual(summary['active_pinch_after_m'][-1], max(after))


if __name__ == '__main__':
    unittest.main()
