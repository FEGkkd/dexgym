"""Read-only bimanual geometry scoring for the P03-SIM0 offline experiment."""
from collections import Counter

import numpy as np


TIP_THRESHOLDS = (.003, .005, .01, .015)
PINCH_THRESHOLD = .020
DEFAULT_CROSS_PAIRS = tuple((i, j) for i in range(5) for j in range(5))
ROTATION_TOL = 2e-7
RADIUS_ROUNDOFF_M = 1e-9


def _identity(row):
    r = row.get('identity', row)
    tick = r['tick']
    if isinstance(tick, (bool, np.bool_)) or not isinstance(tick, (int, np.integer)) or tick < 0:
        raise ValueError('Scoring identity requires an exact non-negative integer source tick')
    return r['root'], r['partition'], int(tick), r['split']


def validate_rows(expected, actual, root_splits=None):
    """Verify ordered one-to-one planned coverage, including unresolved failure rows."""
    planned, observed = list(map(_identity, expected)), list(map(_identity, actual))
    if len(set(planned)) != len(planned) or len(set(observed)) != len(observed):
        raise ValueError('Expected one record per immutable parent root, partition, and tick')
    groups = {}
    for root, _, _, split in planned:
        if split not in ('Train', 'Validation'):
            raise ValueError('Only reviewed development roots may appear in SIM0')
        if groups.setdefault(root, split) != split or (root_splits is not None and root_splits.get(root) != split):
            raise ValueError('Parent root crosses development splits or is outside the frozen split')
    if planned != observed:
        raise ValueError('Lost, reordered, inserted or changed a planned scoring record')
    return {
        'planned_rows': len(planned),
        'planned_rows_by_split': dict(Counter(v[-1] for v in planned)),
        'distinct_roots_by_split': dict(Counter(groups.values())),
        'split_root_overlap': 0,
    }


def causal_mask(tick_ns, source_ns):
    """Return (known, available, future) masks for source use on logged clock support."""
    t, s = np.asarray(tick_ns), np.asarray(source_ns)
    if t.ndim != 1 or s.ndim == 0 or s.shape[0] != len(t):
        raise ValueError('Query clock must have one row for each supplied source record')
    t = t.reshape((len(t),) + (1,) * (s.ndim - 1))
    known = np.isfinite(t) & np.isfinite(s) & (s >= 0) & (t >= 0)
    available = known & (s <= t)
    return {'known': known, 'available': available, 'future': known & (s > t)}


def radial_lower(tips, origins, radii):
    """A conservative necessary per-tip error bound, independent of any estimated q."""
    x, origin, radius = map(lambda a: np.asarray(a, dtype=float), (tips, origins, radii))
    if (x.shape[-3:] != (2, 5, 3) or origin.shape != (2, 5, 3) or radius.shape != (2, 5)
            or not np.isfinite(origin).all() or not np.isfinite(radius).all() or np.any(radius < 0)):
        raise ValueError('Expected named two-hand ten-tip radii and fixed pivot coordinates')
    return np.maximum(np.linalg.norm(x - origin, axis=-1) - radius - RADIUS_ROUNDOFF_M, 0.)


def radial_parameters(robot):
    """Recover fixed hinge pivots and sum segment lengths without mutating robot state."""
    import mujoco

    model = robot.model
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    origins, lengths = [], []
    for tip_id in robot.tip_ids:
        site = model.site(tip_id)
        points = [data.site(tip_id).xpos.copy()]
        body = int(site.bodyid[0])
        while body:
            n = int(model.body_jntnum[body])
            if n not in (0, 1):
                raise ValueError('Bound requires at most one hinge per body in every tip chain')
            if n:
                j = model.joint(int(model.body_jntadr[body]))
                if int(j.type[0]) != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('Bound requires a fixed-base revolute chain')
                points.append(data.joint(j.id).xanchor.copy())
            body = int(model.body_parentid[body])
        if len(points) < 2:
            raise ValueError('Expected a root pivot for every tip')
        origins.append(points[-1])
        lengths.append(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())
    return np.asarray(origins).reshape(2, 5, 3), np.asarray(lengths).reshape(2, 5)


def _array(source, name, shape):
    value = source.get(name)
    if value is None:
        return None
    a = np.asarray(value, dtype=float)
    if a.ndim != len(shape) or any(size is not None and a.shape[i] != size for i, size in enumerate(shape)):
        raise ValueError(f'Unexpected {name!r} shape {a.shape}; expected {shape}')
    return a


def _tips(source, named, landmarks, n):
    x = _array(source, named, (n, 2, 5, 3))
    if landmarks is not None:
        if landmarks.shape[3] <= 0:
            raise ValueError('Landmark axis must contain at least a named fingertip')
        if x is not None and not np.allclose(x, landmarks[:, :, :, -1], equal_nan=True, atol=0., rtol=0.):
            raise ValueError('Conflicting explicit fingertip and last named landmark values')
        return landmarks[:, :, :, -1]
    return x


def _geometry(source, n):
    wrist = _array(source, 'wrist', (n, 2, 3))
    rotation = _array(source, 'rotation', (n, 2, 3, 3))
    world = _array(source, 'world', (n, 2, 5, None, 3))
    local = _array(source, 'local', (n, 2, 5, None, 3))
    return {
        'wrist': wrist, 'rotation': rotation, 'local': local,
        'tips_world': _tips(source, 'tips_world', world, n),
        'tips_local': _tips(source, 'tips_local', local, n),
    }


def _mask(value, n, name):
    if value is None:
        return np.ones(n, dtype=bool)
    a = np.asarray(value)
    if a.shape != (n,) or a.dtype.kind != 'b':
        raise ValueError(f'{name} must be a boolean mask with one item per planned tick')
    return a


def _optional(a, b):
    return a is not None and b is not None


def _relative(wrist, rotation):
    d = wrist[:, 1] - wrist[:, 0]
    r = rotation[:, 0].transpose(0, 2, 1)
    return np.einsum('nij,nj->ni', r, d), r @ rotation[:, 1]


def _rotation_angle(relative):
    s = .5 * np.linalg.norm(np.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
    ), axis=-1), axis=-1)
    c = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.) * .5, -1., 1.)
    return np.arctan2(s, c)


def _check_rotations(x, valid):
    if x is None:
        return
    candidates = x[valid]
    finite = candidates[np.isfinite(candidates).all(axis=(-1, -2))]
    if (not np.allclose(finite.transpose(0, 2, 1) @ finite, np.eye(3), rtol=0., atol=ROTATION_TOL)
            or np.any(np.abs(np.linalg.det(finite) - 1.) > ROTATION_TOL)):
        raise ValueError('Expected proper rotations in the supplied, valid scoring records')


def _stats(array, planned):
    x = np.asarray(array)
    finite = np.isfinite(x) & planned.reshape((len(planned),) + (1,) * (x.ndim - 1))
    y = x[finite]
    return {
        'scored_ticks': int(np.count_nonzero(finite.reshape((len(planned), -1)).any(axis=1))),
        'value_count': len(y),
        'P50_P95_P99_max': [float(v) for v in np.percentile(y, (50, 95, 99, 100))] if len(y) else None,
    }


def _mark(values, valid):
    x = np.asarray(values, dtype=float)
    return np.where(valid.reshape((len(valid),) + (1,) * (x.ndim - 1)), x, np.nan)


def _cross(tips, pairs):
    return np.stack([tips[:, 1, j] - tips[:, 0, i] for i, j in pairs], axis=1)


def _geometry_scores(intent, estimate, active, pairs, pinch_threshold):
    """Every Cartesian signed comparison is estimate minus the immutable input reference."""
    a, b = intent, estimate
    m, v = {}, {}
    distance = lambda x: np.linalg.norm(x, axis=-1)
    if _optional(a['wrist'], b['wrist']):
        p, q = a['wrist'], b['wrist']
        d, e = p[:, 1] - p[:, 0], q[:, 1] - q[:, 0]
        v['wrist_error_m'] = q - p
        v['wrist_midpoint_error_m'] = (q - p).mean(axis=1)
        v['wrist_relative_vector_error_m'] = e - d
        m['wrist_left_m'], m['wrist_right_m'] = distance(q - p).T
        m['wrist_midpoint_m'] = distance(v['wrist_midpoint_error_m'])
        m['wrist_relative_displacement_m'] = distance(v['wrist_relative_vector_error_m'])
        m['wrist_span_signed_m'] = distance(e) - distance(d)
        m['wrist_span_absolute_m'] = np.abs(m['wrist_span_signed_m'])
    if _optional(a['rotation'], b['rotation']):
        ra, rb = a['rotation'], b['rotation']
        m['wrist_left_rotation_rad'], m['wrist_right_rotation_rad'] = _rotation_angle(
            ra.transpose(0, 1, 3, 2) @ rb).T
        if _optional(a['wrist'], b['wrist']):
            pa, rra = _relative(a['wrist'], ra)
            pb, rrb = _relative(b['wrist'], rb)
            v['relative_se3_translation_error_m'] = pb - pa
            m['relative_se3_translation_m'] = distance(pb - pa)
            m['relative_se3_rotation_rad'] = _rotation_angle(rra.transpose(0, 2, 1) @ rrb)
    if _optional(a['tips_world'], b['tips_world']):
        p, q = a['tips_world'], b['tips_world']
        v['world_tip_error_m'] = q - p
        m['world_tip_m'] = distance(q - p)
        m['world_tip_max_m'] = m['world_tip_m'].max(axis=(1, 2))
        m['world_tip_mean_m'] = m['world_tip_m'].mean(axis=(1, 2))
        v['cross_hand_error_m'] = _cross(q, pairs) - _cross(p, pairs)
        m['cross_hand_m'] = distance(v['cross_hand_error_m'])
    if _optional(a['local'], b['local']):
        if a['local'].shape != b['local'].shape:
            raise ValueError('Different named local landmark topology')
        v['local_landmark_error_m'] = b['local'] - a['local']
        m['local_landmarks_m'] = distance(v['local_landmark_error_m'])
        m['local_hand_rmse_m'] = np.sqrt(np.mean(m['local_landmarks_m'] ** 2, axis=(2, 3)))
    if _optional(a['tips_local'], b['tips_local']):
        p, q = a['tips_local'], b['tips_local']
        m['local_tip_m'] = distance(q - p)
        pin_a, pin_b = p[:, :, 1:] - p[:, :, :1], q[:, :, 1:] - q[:, :, :1]
        v['pinch_error_m'] = pin_b - pin_a
        m['pinch_vector_m'] = distance(pin_b - pin_a)
        m['pinch_distance_m'] = np.abs(distance(pin_b) - distance(pin_a))
        near = distance(pin_a) <= pinch_threshold
        m['active_pinch_vector_m'] = np.where(near, m['pinch_vector_m'], np.nan)
        m['active_pinch_distance_m'] = np.where(near, m['pinch_distance_m'], np.nan)
    return ({k: _mark(x, active) for k, x in m.items()}, {k: _mark(x, active) for k, x in v.items()})


def _velocity_scores(intent, estimate, shared_valid, tick_ns, window_ids, max_step_ns, cross_pairs):
    t = np.asarray(tick_ns)
    n = len(shared_valid)
    windows = np.asarray(window_ids) if window_ids is not None else np.zeros(n)
    if t.shape != (n,) or windows.shape != (n,) or not np.isfinite(t).all():
        raise ValueError('Expected finite time and one independent window ID per scoring row')
    same_window = windows[1:] == windows[:-1]
    step = np.diff(t)
    if np.any(same_window & (step <= 0)):
        raise ValueError('Query ticks must increase within each chronological window')
    adjacent = shared_valid[1:] & shared_valid[:-1] & same_window
    if max_step_ns is not None:
        if max_step_ns <= 0:
            raise ValueError('Expected a positive maximum derivative support gap')
        adjacent &= step <= max_step_ns
    dt = np.where(adjacent, step * 1e-9, np.nan)
    out = {}
    trajectories = {name: (intent[key], estimate[key]) for name, key in (
        ('wrist', 'wrist'), ('tip', 'tips_world'), ('local_tip', 'tips_local'),
    )}
    if all(x is not None for x in trajectories['wrist']):
        trajectories['relative_wrist'] = tuple(x[:, 1] - x[:, 0] for x in trajectories['wrist'])
    if all(x is not None for x in trajectories['tip']):
        trajectories['cross_hand'] = tuple(_cross(x, cross_pairs) for x in trajectories['tip'])
    if all(x is not None for x in trajectories['local_tip']):
        trajectories['pinch'] = tuple(x[:, :, 1:] - x[:, :, :1] for x in trajectories['local_tip'])
    for name, (a, b) in trajectories.items():
        if _optional(a, b):
            seconds = dt.reshape((len(dt),) + (1,) * (a.ndim - 1))
            dv = (np.diff(b, axis=0) - np.diff(a, axis=0)) / seconds
            for k, x in ((f'{name}_velocity_m_s', dv), (f'{name}_geometry_step_m', np.diff(b, axis=0)),
                         (f'{name}_geometry_step_error_m', np.diff(b - a, axis=0))):
                y = np.linalg.norm(x, axis=-1)
                new = np.full((n,) + y.shape[1:], np.nan)
                new[1:] = _mark(y, adjacent)
                out[k] = new
    if _optional(intent['rotation'], estimate['rotation']):
        changes = []
        for x in (intent['rotation'], estimate['rotation']):
            changes.append(_rotation_angle(x[:-1].transpose(0, 1, 3, 2) @ x[1:]))
        res = np.full((n, 2), np.nan)
        res[1:] = _mark(np.abs(changes[1] - changes[0]) / dt[:, None], adjacent)
        out['wrist_angular_speed_difference_rad_s'] = res
    return out


def score_ticks(intent, estimate, *, projected=None, tick_ns=None, window_ids=None, planned=None,
                thresholds_m=TIP_THRESHOLDS, radial_origin_m=None, radial_radius_m=None,
                source_ns=None, max_step_ns=None, cross_pairs=DEFAULT_CROSS_PAIRS, pinch_threshold_m=PINCH_THRESHOLD):
    """Score the immutable reference without fitting it to FK or command samples.

    Pair raw intent to candidate mapping for mapping fidelity, or mapped intent to
    observed simulated FK for achieved geometry. Only passing an independently frozen
    projected geometry produces execution/projection decompositions. Scored values
    preserve unavailable entries as NaN. Threshold counts always use planned ticks
    as the denominator, and percentiles use finite measured values.
    """
    n = len(intent['wrist'])
    if n <= 0:
        raise ValueError('Scoring requires a non-empty, preregistered interval')
    eval_mask = _mask(planned, n, 'planned')
    intent_mask = _mask(intent.get('valid'), n, 'intent.valid') & eval_mask
    observe_mask = _mask(estimate.get('valid'), n, 'estimate.valid') & eval_mask
    a, b = _geometry(intent, n), _geometry(estimate, n)
    combined = intent_mask & observe_mask
    _check_rotations(a['rotation'], intent_mask)
    _check_rotations(b['rotation'], observe_mask)
    pairs = tuple(tuple(pair) for pair in cross_pairs)
    if not pairs or len(set(pairs)) != len(pairs) or any(not (0 <= i < 5 and 0 <= j < 5) for i, j in pairs):
        raise ValueError('Expected a unique, named fixed set of cross-hand fingertip pairs')
    if pinch_threshold_m < 0 or not np.isfinite(pinch_threshold_m):
        raise ValueError('Invalid fixed reference pinch threshold')
    if any(v < 0 or not np.isfinite(v) for v in thresholds_m):
        raise ValueError('Invalid frozen fingertip thresholds')
    metrics, vectors = _geometry_scores(a, b, combined, pairs, pinch_threshold_m)
    if tick_ns is not None:
        metrics.update(_velocity_scores(a, b, combined, tick_ns, window_ids, max_step_ns, pairs))
    elif source_ns is not None:
        raise ValueError('Source timing audit requires supplied query ticks')
    time = {} if source_ns is None else causal_mask(tick_ns, source_ns)
    if time and np.any(time['future'] & intent_mask.reshape((n,) + (1,) * (time['future'].ndim - 1))):
        raise ValueError('A valid claimed causal record consumes a future sample')
    if radial_origin_m is not None or radial_radius_m is not None:
        if radial_origin_m is None or radial_radius_m is None or a['tips_world'] is None:
            raise ValueError('Radial bounds need both frozen model coefficients and immutable reference tips')
        low = radial_lower(a['tips_world'], radial_origin_m, radial_radius_m)
        metrics['strict_radial_lower_by_tip_m'] = _mark(low, intent_mask)
        metrics['strict_radial_lower_max_m'] = _mark(low.max(axis=(1, 2)), intent_mask)
    if projected is not None:
        f = _geometry(projected, n)
        projection_mask = _mask(projected.get('valid'), n, 'projected.valid') & eval_mask
        if any(x['tips_world'] is None for x in (a, b, f)):
            raise ValueError('Fingertip decomposition needs all three separately supplied geometries')
        signals = {
            'projection_tip_m': (f['tips_world'] - a['tips_world'], projection_mask & intent_mask),
            'execution_tip_m': (b['tips_world'] - f['tips_world'], observe_mask & projection_mask),
            'end_to_end_tip_m': (b['tips_world'] - a['tips_world'], combined),
        }
        for k, (signal, active) in signals.items():
            vectors[k] = _mark(signal, active)
            metrics[k] = np.linalg.norm(vectors[k], axis=-1)
            metrics[k.replace('_tip_m', '_tip_max_m')] = metrics[k].max(axis=(1, 2))
    coverage = {
        'planned_ticks': int(eval_mask.sum()),
        'reference_valid_ticks': int(intent_mask.sum()),
        'estimate_valid_ticks': int(observe_mask.sum()),
        'joint_valid_ticks': int(combined.sum()),
    }
    summary = {'coverage': coverage, 'metrics': {k: _stats(v, eval_mask) for k, v in metrics.items()}}
    for key, metric, less in (('tip_hits', 'world_tip_max_m', True),
                              ('strictly_excluded', 'strict_radial_lower_max_m', False)):
        if metric in metrics:
            x = metrics[metric]
            summary[key] = {
                f'{v * 1000:g}': int(np.count_nonzero(np.isfinite(x) & ((x <= v) if less else (x > v))))
                for v in thresholds_m
            }
            summary[f'{key}_denominator'] = coverage['planned_ticks']
    return {
        'summary': summary,
        'per_tick': metrics,
        'vectors': vectors,
        'masks': {'planned': eval_mask, 'intent': intent_mask, 'estimate': observe_mask, 'pair': combined} | time,
    }
