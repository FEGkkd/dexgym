from copy import deepcopy
import json
import unittest

import numpy as np

from dexgym.tasks.direct.wholebody_s05.sim0.scoring import causal_mask, radial_lower, score_ticks, validate_rows


def fixture(n=1):
    w = np.zeros((n, 2, 3))
    w[:, 1, 0] = .20
    local = np.zeros((n, 2, 5, 2, 3))
    local[:, :, :, -1, 0] = [0., .010, .070, .090, .100]
    return {
        'wrist': w, 'rotation': np.tile(np.eye(3), (n, 2, 1, 1)), 'local': local,
        'world': local + w[:, :, None, None],
    }


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class Sim0ScoringTests(unittest.TestCase):
    def test_shared_translation_preserves_relative_function(self):
        intent = fixture()
        estimate = fixture()
        estimate['wrist'] += [0., .04, 0.]
        estimate['world'] += [0., .04, 0.]
        result = score_ticks(intent, estimate)
        per = result['per_tick']
        np.testing.assert_allclose(per['wrist_midpoint_m'], [.04], atol=1e-15)
        np.testing.assert_allclose(per['wrist_relative_displacement_m'], 0., atol=1e-15)
        np.testing.assert_allclose(per['wrist_span_signed_m'], 0., atol=1e-15)
        np.testing.assert_allclose(per['relative_se3_translation_m'], 0., atol=1e-15)
        np.testing.assert_allclose(per['relative_se3_rotation_rad'], 0., atol=1e-15)
        np.testing.assert_allclose(per['world_tip_max_m'], .04, atol=1e-15)
        np.testing.assert_allclose(per['cross_hand_m'], 0., atol=1e-15)
        np.testing.assert_allclose(per['pinch_vector_m'], 0., atol=1e-15)
        self.assertNotIn('execution_tip_m', per)

    def test_shared_yaw_world_change_is_not_relative_se3_distortion(self):
        intent = fixture()
        estimate = fixture()
        r = rotation_z(.50)
        estimate['wrist'] = intent['wrist'] @ r.T
        estimate['world'] = intent['world'] @ r.T
        estimate['rotation'] = r @ intent['rotation']
        per = score_ticks(intent, estimate)['per_tick']
        np.testing.assert_allclose(per['wrist_left_rotation_rad'], .50, atol=1e-15)
        np.testing.assert_allclose(per['relative_se3_translation_m'], 0., atol=1e-15)
        np.testing.assert_allclose(per['relative_se3_rotation_rad'], 0., atol=1e-15)
        np.testing.assert_allclose(per['wrist_span_signed_m'], 0., atol=1e-15)
        self.assertGreater(per['wrist_relative_displacement_m'][0], 0.)

    def test_rotation_geodesic_near_pi_and_bad_rotation(self):
        a, b = fixture(), fixture()
        b['rotation'][:, 1] = rotation_z(np.pi - 1e-8)
        result = score_ticks(a, b)
        np.testing.assert_allclose(result['per_tick']['relative_se3_rotation_rad'], np.pi - 1e-8, atol=1e-14)
        b['rotation'][:, 1] = np.diag([1., 1., -1.])
        with self.assertRaisesRegex(ValueError, 'proper rotations'):
            score_ticks(a, b)

    def test_near_pinch_uses_signed_input_hand_and_preserves_missing(self):
        intent, estimate = fixture(), fixture()
        estimate['local'][0, 1, 1, -1, 0] += .005
        estimate['local'][0, 0, 2, -1, 0] += .009
        p = score_ticks(intent, estimate)['per_tick']
        self.assertAlmostEqual(p['active_pinch_vector_m'][0, 1, 0], .005, places=15)
        self.assertAlmostEqual(p['active_pinch_distance_m'][0, 1, 0], .005, places=15)
        self.assertTrue(np.isnan(p['active_pinch_vector_m'][0, 0, 1]))
        self.assertAlmostEqual(p['pinch_distance_m'][0, 0, 1], .009, places=15)

    def test_decomposition_is_signed_and_reference_is_immutable(self):
        mapped, projected, fk = fixture(), fixture(), fixture()
        backup = deepcopy(mapped)
        projected['world'] += [0., .010, 0.]
        fk['world'] += [0., .007, 0.]
        result = score_ticks(mapped, fk, projected=projected)
        v = result['vectors']
        np.testing.assert_allclose(v['projection_tip_m'][..., 1], .010, atol=1e-15)
        np.testing.assert_allclose(v['execution_tip_m'][..., 1], -.003, atol=1e-15)
        np.testing.assert_allclose(v['end_to_end_tip_m'], v['projection_tip_m'] + v['execution_tip_m'], atol=1e-15)
        for key in mapped:
            np.testing.assert_array_equal(mapped[key], backup[key])
        json.dumps(result['summary'], allow_nan=False)

    def test_planned_denominator_keeps_failed_fingertips(self):
        intent, estimate = fixture(3), fixture(3)
        estimate['world'][1, 0, 0, -1] = np.nan
        estimate['valid'] = np.array([True, True, False])
        res = score_ticks(intent, estimate)
        self.assertEqual(res['summary']['tip_hits_denominator'], 3)
        self.assertEqual(res['summary']['tip_hits']['3'], 1)
        self.assertEqual(res['summary']['metrics']['world_tip_max_m']['scored_ticks'], 1)
        self.assertEqual(res['summary']['metrics']['world_tip_m']['value_count'], 19)
        self.assertTrue(np.isnan(res['per_tick']['world_tip_max_m'][1:]).all())

    def test_radial_bound_is_independent_of_missing_fk(self):
        intent, estimate = fixture(3), fixture(3)
        intent['world'][0, 0, 0, -1] = [.10, 0., 0.]
        estimate['valid'] = np.zeros(3, dtype=bool)
        r, centers = np.ones((2, 5)) * .09, intent['wrist'][0, :, None].repeat(5, axis=1)
        tips = intent['world'][..., -1, :]
        low = radial_lower(tips, centers, r)
        self.assertAlmostEqual(low[0, 0, 0], .01 - 1e-9, places=15)
        self.assertEqual(low[1, 0, 0], 0.)
        res = score_ticks(intent, estimate, radial_radius_m=r, radial_origin_m=centers)
        self.assertEqual(res['summary']['strictly_excluded_denominator'], 3)
        self.assertGreater(res['summary']['strictly_excluded']['3'], 0)
        self.assertEqual(res['summary']['tip_hits']['3'], 0)

    def test_velocity_support_no_future_or_crossing_windows(self):
        a, b = fixture(6), fixture(6)
        b['wrist'] += np.arange(6)[:, None, None] * np.array([0., .002, 0.])
        b['valid'] = np.array([True, True, True, False, True, True])
        t = np.array([100, 120, 140, 160, 180, 220]) * 1_000_000
        windows = ['a', 'a', 'b', 'b', 'b', 'b']
        m = score_ticks(a, b, tick_ns=t, window_ids=windows, max_step_ns=20_000_000)['per_tick']
        np.testing.assert_allclose(m['wrist_velocity_m_s'][1], [.1, .1], atol=1e-15)
        self.assertTrue(np.isnan(m['wrist_velocity_m_s'][[0, 2, 3, 4, 5]]).all())
        prefix = score_ticks({k: v[:3] for k, v in a.items()}, {k: v[:3] for k, v in b.items()},
                             tick_ns=t[:3], window_ids=windows[:3], max_step_ns=20_000_000)['per_tick']
        for key, value in prefix.items():
            np.testing.assert_allclose(value, m[key][:3], atol=0., rtol=0., equal_nan=True)

    def test_clock_support_keeps_unknown_and_rejects_future_causal_use(self):
        tick = np.array([100, 120])
        src = np.array([[90, -1], [121, 120]])
        status = causal_mask(tick, src)
        np.testing.assert_array_equal(status['known'], [[True, False], [True, True]])
        np.testing.assert_array_equal(status['available'], [[True, False], [False, True]])
        np.testing.assert_array_equal(status['future'], [[False, False], [True, False]])
        with self.assertRaisesRegex(ValueError, 'future sample'):
            score_ticks(fixture(2), fixture(2), tick_ns=tick, source_ns=src)

    def test_full_identity_coverage_and_frozen_split(self):
        rows = [{'root': 'a', 'partition': 'cc', 'tick': 0, 'split': 'Train'},
                {'root': 'b', 'partition': 'dd', 'tick': 8, 'split': 'Validation'}]
        observed = [{'identity': item, 'status': s} for item, s in zip(rows, ('PASS', 'UNKNOWN'), strict=True)]
        checked = validate_rows(rows, observed, {'a': 'Train', 'b': 'Validation'})
        self.assertEqual(checked['planned_rows'], 2)
        self.assertEqual(checked['split_root_overlap'], 0)
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_rows(rows, observed[:1])
        with self.assertRaisesRegex(ValueError, 'crosses'):
            validate_rows(rows, observed, {'a': 'Train', 'b': 'Train'})
        with self.assertRaisesRegex(ValueError, 'integer'):
            validate_rows(rows, [rows[0] | {'tick': 0.1}, rows[1]])


if __name__ == '__main__':
    unittest.main()
