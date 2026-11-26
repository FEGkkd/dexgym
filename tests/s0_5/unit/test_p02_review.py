"""Review regressions for functional statistics, mapping, and minimax."""
import unittest

from seal_p02 import active_pinch_errors


class PinchStatisticsTests(unittest.TestCase):
    def test_active_indices_are_counts_and_positions(self):
        cases = [([0], [10], [1]), ([1], [20], [2]),
                 ([0, 2], [10, 30], [1, 3]), ([], [], [])]
        for indices, before, after in cases:
            with self.subTest(indices=indices):
                hand = {'near_pinch_pairs': indices, 'before_error_m': [10, 20, 30, 40],
                        'after_error_m': [1, 2, 3, 4]}
                self.assertEqual(active_pinch_errors(hand), (len(indices), before, after))


if __name__ == '__main__':
    unittest.main()
