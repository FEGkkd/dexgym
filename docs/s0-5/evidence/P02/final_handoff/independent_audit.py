"""Read-only independent diagnostic for the signed P02 M1 14/54 handoff."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel


P02 = Path('docs/s0-5/evidence/P02')
REFINE = P02 / 'refinement'
BASELINE = 'ecd12ecd6657539e44830dde07c6055c6c507192'
SIGNED = {
    'execution_receipt.json': '0adfc8976ebfa7ee0845d42e5d291caa70f94cf172e72d4a7d2fcb3c16a15aa0',
    'fitted_candidates.json': '577195a8eb66699e11cf8bcf910383086672c384723d61fa90c13ed90652baa7',
    'scoring_freeze_candidate.json': '96aac52666d7b03d6a3c087cb845f824a2ed4403f1cf35ae87af0264dd5f4f94',
}
METRES_TOL = 2e-9
Q_TOL = 1e-9
MIN_DISTANCE = .0005
DISTANCE_TOL = 1e-7
TIP_CUTS = (.003, .005, .01, .015)


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def require(predicate, message):
    if not predicate:
        raise ValueError(message)


def read(path, expected=None):
    if expected is not None:
        require(digest(path) == expected, f'SHA-256 mismatch: {path}')
    return json.loads(path.read_bytes())


def same(actual, expected, name, atol=METRES_TOL):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.allclose(a, b, rtol=0., atol=atol), f'Mismatch: {name}')


def bits(actual, expected, name):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and a.tobytes() == b.tobytes(), f'Bitwise mismatch: {name}')


def signed_material(repo, original, matched):
    git_sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', BASELINE, git_sha], check=True)
    er, fits, request = (
        read(repo / REFINE / name, expected) for name, expected in SIGNED.items()
    )
    archive = read(repo / REFINE / 'archive_sha256.json', er['output_sha256']['archive_sha256.json'])
    prior_keys = read(repo / P02 / 'refinement_plan.json', er['plan_sha256'])
    old_receipt = read(repo / P02 / 'review_receipt.json', prior_keys['reference_review_receipt_sha256'])
    old_sha = read(repo / P02 / 'output_sha256.json', old_receipt['receipts']['artifact_manifest'])
    previous = read(original, old_sha['intent-02/study_records.json'])
    match = read(matched, archive['matched-01/matched_candidates.json'])
    require(match['candidates_sha256'] == SIGNED['fitted_candidates.json'] and fits['fit_scope'] == 'Train',
            'Matched candidates do not bind the Train fit')
    require(fits['optimizer']['M1']['train_roots'] == 24 and
            fits['plan_sha256'] == er['plan_sha256'] and request['fit_sha256'] == SIGNED['fitted_candidates.json'],
            'Changed Train fit root count, plan or research handoff fit identity')
    require(request['formal_scoring_freeze'] == 'PENDING_HUMAN_APPROVAL' and
            not request['p03_authorized'] and request['reviewed_original_target_retained'],
            'Existing handoff is proposed research only')
    require(er['phases'] == {'P01': 'ACCEPTED', 'P02': 'READY_FOR_REVIEW', 'P03': 'NOT_STARTED'} and
            er['authorized_p03'] is False, 'Unexpected signed phase state')
    base = read(repo / P02 / 'plan.json', old_receipt['plan_sha256'])
    audit_model = read(repo / P02 / 'model_report.json', er['model_audit_sha256'])
    assets = read(repo / REFINE / 'model_assets_sha256.json', er['output_sha256']['model_assets_sha256.json'])
    require(assets == audit_model['model']['contract']['assets'], 'Reviewed model asset set differs')
    for name, h in assets.items():
        require(digest(Path(name)) == h, f'Reviewed model asset changed: {name}')
    return git_sha, base, fits, previous, match['rows'], len(assets)


def norm(a, axis=-1):
    return np.linalg.norm(a, axis=axis)


def target(data):
    result = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    require(all(np.isfinite(a).all() for a in result.values()), 'Nonfinite mapped target')
    for k, shape in (
        ('world', (2, 5, 4, 3)), ('local', (2, 5, 4, 3)), ('wrist', (2, 3)),
        ('rotation', (2, 3, 3)), ('pinch', (2, 4, 3)), ('relative_wrist', (3,)),
        ('relative_rotation', (3, 3)),
    ):
        require(result[k].shape == shape, f'Invalid target field shape: {k}')
    return result


def moved_target(old, cfg0, cfg, pairs):
    old_r, new_r = np.asarray(cfg0['rotation']), np.asarray(cfg['rotation'])
    old_p = np.asarray(cfg0['translation'])
    new_p = np.asarray(cfg['translation'])
    centre = np.asarray(cfg['pivot'])
    require(np.allclose(new_r @ new_r.T, np.eye(3), rtol=0., atol=Q_TOL) and
            abs(np.linalg.det(new_r) - 1.) <= Q_TOL,
            'Shared registration must be proper')
    same(cfg['mount'], cfg0['mount'], 'Frozen side wrist mount')
    # Row vectors are used here only for storage and the Cartesian calculations.
    raw_wrist = (old['wrist'] - old_p) @ old_r
    w = (centre + (raw_wrist - centre) * cfg['scale']) @ new_r.T + new_p
    common_rotation = new_r @ old_r.T
    local_offset = old['world'] - old['wrist'][:, None, None]
    world = w[:, None, None] + local_offset @ common_rotation.T
    rr = common_rotation @ old['rotation']
    cross = np.stack([world[1, b, -1] - world[0, a, -1] for a, b in pairs])
    return {
        'wrist': w, 'rotation': rr, 'world': world, 'local': old['local'], 'pinch': old['pinch'],
        'cross': cross, 'relative_wrist': rr[0].T @ (w[1] - w[0]),
        'relative_rotation': rr[0].T @ rr[1],
    }


def distances(m, d, pairs):
    cap = .0015
    return min([cap] + [mujoco.mj_geomDistance(m, d, a, b, cap, None) for a, b in pairs])


def legal(robot, q):
    require(q.shape == (54,) and np.isfinite(q).all(), 'Invalid 54-axis witness state')
    bounds = robot.bounds
    range_slack = min(float(np.min(q - bounds[:, 0])), float(np.min(bounds[:, 1] - q)))
    coupling_slack = min(np.deg2rad(110.5) - 1.025 * abs(q[i]) - abs(q[j])
                         for i, j in [(5, 6), (12, 13)])
    require(range_slack >= -Q_TOL and coupling_slack >= -Q_TOL, 'Witness violates joint/drive or J6/J7')
    robot.set(q, contacts=True)
    m, d = robot.model, robot.data
    gap = distances(m, d, robot.collision_pairs)
    penetrating = any(contact.dist < -DISTANCE_TOL for contact in d.contact)
    require(gap >= MIN_DISTANCE - DISTANCE_TOL and not penetrating, 'Witness violates 0.5 mm collision clearance')
    # Capped signed geometry distance is sufficient for testing the local safety inequality.
    return {'range_slack_rad': range_slack, 'coupling_slack_rad': coupling_slack,
            'geometry_distance_capped_m': float(gap)}


def radius(robot):
    """Independent polygonal chain length calculation for 54 revolute axes."""
    robot.set(np.zeros(54))
    m, d = robot.model, robot.data
    origins, totals = [], []
    for id_ in robot.tip_ids:
        site = m.site(id_)
        endpoints = [d.site(id_).xpos.copy()]
        body = int(site.bodyid[0])
        while body:
            count = int(m.body_jntnum[body])
            require(count in (0, 1), 'Radial proof requires at most one joint per body')
            if count:
                j = m.joint(int(m.body_jntadr[body]))
                require(int(j.type[0]) == mujoco.mjtJoint.mjJNT_HINGE, 'Nonrevolute axis breaks radial proof')
                endpoints.append(d.joint(j.id).xanchor.copy())
            body = int(m.body_parentid[body])
        require(len(endpoints) > 1, 'Radial proof requires a fixed pivot')
        origins.append(endpoints[-1])
        totals.append(sum(float(norm(b - a)) for a, b in zip(endpoints, endpoints[1:])))
    return np.asarray(origins).reshape(2, 5, 3), np.asarray(totals).reshape(2, 5)


def radial(geometry, chain):
    centers, lengths = chain
    return max(0., float(np.max(norm(geometry['world'][:, :, -1] - centers) - lengths)) - 1e-9)


def measure(robot, q, r, scales, pairs):
    """Compute task residuals separately from the production oracle's metrics and cost."""
    g = robot.geometry(q)
    p, R = g.wrists, g.rotations
    e_tip = g.tips - r['world'][:, :, -1]
    p_rel = R[0].T @ (p[1] - p[0])
    R_rel = R[0].T @ R[1]
    e_wrot = Rotation.from_matrix(r['rotation'].transpose(0, 2, 1) @ R).as_rotvec()
    e_rrot = Rotation.from_matrix(r['relative_rotation'].T @ R_rel).as_rotvec()
    e_local = g.local - r['local']
    pin = g.local[:, 1:, -1] - g.local[:, :1, -1]
    cross = np.stack([g.tips[1, right] - g.tips[0, left] for left, right in pairs])
    residuals = {
        'world_tip_m': e_tip, 'wrist_translation_m': p - r['wrist'], 'wrist_rotation_rad': e_wrot,
        'local_landmark_m': e_local, 'pinch_vector_m': pin - r['pinch'], 'cross_hand_vector_m': cross - r['cross'],
        'relative_wrist_translation_m': p_rel - r['relative_wrist'], 'relative_wrist_rotation_rad': e_rrot,
        'joint_posture_rad': q - robot.home,
    }
    tip = norm(e_tip)
    return {
        'max_tip_m': float(tip.max()), 'mean_tip_m': float(tip.mean()), 'tip_m': tip.tolist(),
        'tip_vectors_m': e_tip.tolist(), 'local_landmark_rmse_m': float(np.sqrt(np.mean(norm(e_local) ** 2))),
        'pinch_vector_m': norm(residuals['pinch_vector_m']).tolist(),
        'pinch_distance_m': abs(norm(pin) - norm(r['pinch'])).tolist(),
        'cross_hand_m': norm(residuals['cross_hand_vector_m']).tolist(),
        'wrist_translation_m': norm(residuals['wrist_translation_m']).tolist(),
        'wrist_rotation_rad': norm(e_wrot).tolist(),
        'relative_wrist_translation_m': float(norm(residuals['relative_wrist_translation_m'])),
        'relative_wrist_rotation_rad': float(norm(e_rrot)),
        'objective_value': sum(float(np.sum((v / scales[k]) ** 2)) / 2 for k, v in residuals.items()),
    }


def summarize(rows, group, which):
    planned = [x for x in rows if (group == 'All' or group in (x['split'], x['cohort'],
                                                            f"{x['split']}_{x['cohort']}"))]
    vals = [x['methods'][which]['max_tip_m'] for x in planned if which in x['methods']]
    return {
        'planned': len(planned), 'scored': len(vals), 'unknown_or_no_q': len(planned) - len(vals),
        'max_tip_m_P50_P95_P99_max': np.percentile(vals, [50, 95, 99, 100]).tolist() if vals else None,
        'within_threshold_mm': {str(int(v * 1000)): sum(x <= v for x in vals) for v in TIP_CUTS},
    }


def check_attempts(method, seeds, fixed, robot, goal, base, reference):
    trials = method['attempts']
    require(len(trials) == len(seeds) == 3, 'Expected the three frozen M1 start slots')
    for trial, seed in zip(trials, seeds, strict=True):
        require(trial['label'] == seed['label'], 'Seed label or start order changed')
        expected = np.asarray(seed['input_initial'], dtype=float)
        if fixed is not None:
            expected[14:] = fixed
        bits(trial['input_initial'], expected, 'Preclip initialization')
        bits(trial['initial'], np.clip(expected, *robot.bounds.T), 'Clipped initialization')
        require(0 <= trial['iterations'] <= 300 and np.isfinite(trial['elapsed_s']) and trial['elapsed_s'] >= 0.,
                'Invalid recorded solver work')
        if trial['feasible_best']:
            q = np.asarray(trial['best_q'], dtype=float)
            if fixed is not None:
                bits(q[14:], fixed, 'Held SDK hands in feasible attempt')
            res = measure(robot, q, goal, base['scales'], base['cross_hand_pairs'])
            same(res['objective_value'], trial['best_value'], 'Attempt objective residual', atol=1e-8)
            require(robot.legality(q)['feasible'], 'Recorded attempt best candidate is illegal')
        else:
            require('best_q' not in trial and 'best_value' not in trial, 'Inconsistent absent attempt candidate')
    q = method['q']
    candidates = [t for t in trials if t['feasible_best']]
    if q is None:
        require(not candidates and method['status'] in ('SOLVER_TIMEOUT', 'UNKNOWN'), 'No-q status mismatch')
        return None
    q = np.asarray(q, dtype=float)
    if fixed is not None:
        bits(q[14:], fixed, 'Held SDK hands in published witness')
    require(bool(candidates), 'Returned q has no archived legal start result')
    winner = min(candidates, key=lambda v: v['best_value'])
    bits(q, winner['best_q'], 'Returned q must win ordinary solve budget')
    check = legal(robot, q)
    score = measure(robot, q, goal, base['scales'], base['cross_hand_pairs'])
    same(method['objective_value'], score['objective_value'], 'Returned objective', atol=1e-8)
    for key, value in method['metrics'].items():
        same(value, score[key], f'Returned metric {key}')
    against_original = measure(robot, q, reference, base['scales'], base['cross_hand_pairs'])
    return score | {'measured_constraints': check, 'vs_immutable_M0': against_original}


def review(repo, original, matched, pair):
    git_sha, base, fits, prior, previous, assets = signed_material(repo, original, matched)
    require(pair['schema'] == 's05.p02.m1_pairing.v1', 'Unexpected A experiment schema')
    assert_source = {'refinement_receipt': SIGNED['execution_receipt.json'],
                     'fitted_candidates': SIGNED['fitted_candidates.json'],
                     'refinement_scoring_candidate': SIGNED['scoring_freeze_candidate.json'],
                     'original': digest(original), 'matched': digest(matched), 'model': base['model_sha256'],
                     'pairing_script': digest(repo / P02 / 'final_handoff/m1_pairing.py')}
    require(all(pair['source_sha256'][k] == v for k, v in assert_source.items()), 'A output source SHA differs')
    require(pair['approved_input_commit'] == BASELINE and pair['p03_authorized'] is False and
            pair['formal_score_authorized'] is False, 'A output exceeded the static research scope')
    protocol = pair['seed_protocol']
    require(protocol['iterations_per_start'] == 300 and protocol['wall_seconds_per_start'] == 8. and
            protocol['clearance_m'] == MIN_DISTANCE and protocol['source_seed_not_regenerated'] is True and
            protocol['labels'] == base['solver']['starts'], 'Different seed, budget or safety protocol')
    rows = pair['rows']
    require(len(rows) == len(prior) == len(previous) == 38, 'Study must retain 32+6 planned identities')
    require(Counter(x['identity']['cohort'] for x in rows) == {'ROOT': 32, 'TAIL': 6}, 'Changed cohort')
    require(Counter(x['identity']['split'] for x in rows if x['identity']['cohort'] == 'ROOT') ==
            {'Train': 24, 'Validation': 8}, 'Changed frozen root split')
    ids = [x['identity']['key'] for x in rows]
    require(len(set(ids)) == 38, 'Repeated planned source identity')
    require(base['clearance_m'] == MIN_DISTANCE, 'Changed reviewed collision margin')
    model_xml = repo / base['model']
    require(digest(model_xml) == base['model_sha256'], 'Changed base MJCF')
    robot = RobotModel(model_xml, MIN_DISTANCE)
    chain = radius(robot)
    configs, pairs = fits['configs'], base['cross_hand_pairs']
    require(configs['M0']['scale'] == configs['M1']['scale'] == 1., 'Rigid M0/M1 scale changed')
    require(.75 <= configs['M2']['scale'] < 1., 'M2 is a separately tested bounded compression')
    signed_local = read(repo / REFINE / 'quantitative_summary.json',
                        read(repo / REFINE / 'execution_receipt.json')['output_sha256']['quantitative_summary.json'])
    comparisons = dict(m1_max_span_diff_m=0., m1_max_shape_cross_diff_m=0.,
                       m2_max_span_compression_m=0., m2_max_shape_cross_diff_m=0., strict36={})
    table = []
    for new, old, previous_case in zip(rows, prior, previous, strict=True):
        identity = new['identity']
        require(identity == old['identity'] == previous_case['identity'], 'Planned case identity drift')
        case = {'key': identity['key'], 'split': identity['split'], 'cohort': identity['cohort'], 'methods': {}}
        table.append(case)
        if 'mapped' not in old:
            require('candidates' not in previous_case and new.get('target_M1') is None,
                    'Original unknown intent must stay unknown')
            require(new['status'] == 'UNKNOWN_INPUT' and new['original_disposition'] == old['disposition'],
                    'Original unknown disposition changed')
            require(all(x is None or x.get('q') is None for x in (new.get('results') or {}).values()),
                    'Previously unqualified intent yielded a pose')
            case['status'] = new['status']
            continue
        original_target = target(old['mapped'])
        for label in ('M0', 'M1', 'M2'):
            stored = target(previous_case['candidates'][label]['mapped'])
            recomputed = moved_target(original_target, configs['M0'], configs[label], pairs)
            for field, values in recomputed.items():
                same(stored[field], values, f'{label} frozen morphology/workspace field {field}')
            lb = radial(stored, chain)
            same(lb, previous_case['candidates'][label]['strict_radial_bound_m'], f'{label} strict radius bound')
            if lb > .003:
                comparisons['strict36'][label] = 1 + comparisons['strict36'].get(label, 0)
            if label == 'M0':
                for field, values in stored.items():
                    bits(values, original_target[field], f'Original M0 field {field}')
            if label in ('M1', 'M2'):
                old_span = norm(original_target['wrist'][1] - original_target['wrist'][0])
                new_span = norm(stored['wrist'][1] - stored['wrist'][0])
                back = np.asarray(configs['M0']['rotation']) @ np.asarray(configs[label]['rotation']).T
                cross_change = float(norm(stored['cross'] @ back.T - original_target['cross']).max())
                span_key = {'M1': 'm1_max_span_diff_m', 'M2': 'm2_max_span_compression_m'}[label]
                comparisons[span_key] = max(comparisons[span_key], abs(float(new_span - old_span)))
                shape_key = label.lower() + '_max_shape_cross_diff_m'
                comparisons[shape_key] = max(comparisons[shape_key], cross_change)
                same(new_span, old_span * configs[label]['scale'], f'{label} wrist span ratio')
        require(comparisons['m1_max_shape_cross_diff_m'] < METRES_TOL and
                comparisons['m1_max_span_diff_m'] < METRES_TOL, 'Shared rigid M1 changed relative shape')
        archived = previous_case['candidates']['M1']
        m1 = target(archived['mapped'])
        published_target = target(new['target_M1'])
        for key, val in m1.items():
            bits(val, published_target[key], f'New paired runs must retain exact M1 {key}')
        fixed = np.clip(np.asarray(old['sdk_wrist_ik']['sdk_unclipped_q']), *robot.bounds[14:].T)
        bits(new['sdk_hands_hold']['frozen_q_rad'], fixed, 'Frozen source SDK angles')
        bits(new['sdk_hands_hold']['unclipped_q_rad'], old['sdk_wrist_ik']['sdk_unclipped_q'], 'SDK hand input')
        require(new['status'] == 'ATTEMPTED' and new['constrained_geometry_evaluations'] > 0,
                'No measured proof of a fixed-hand solve')
        seeds = archived['multitask']['attempts']
        radial_m = radial(m1, chain)
        case['strict_radial_lower_bound_m'] = radial_m
        for mode in ('arms14', 'whole54'):
            valid = check_attempts(new['results'][mode], seeds, fixed if mode == 'arms14' else None,
                                   robot, m1, base, original_target)
            if valid is not None:
                require(valid['max_tip_m'] + METRES_TOL >= radial_m,
                        'A legal witness cannot beat the analytic lower bound')
                case['methods'][mode] = valid
        case['status'] = new['status']
        if set(case['methods']) == {'arms14', 'whole54'}:
            case['delta_54_minus_14_m'] = (case['methods']['whole54']['max_tip_m'] -
                                          case['methods']['arms14']['max_tip_m'])
            same(case['delta_54_minus_14_m'], new['paired_delta_54_minus_14']['max_tip_m'],
                 'Paired ten-tip delta')
        if 'arms14' in case['methods']:
            witness = new['embedded_14_witness_no_extra_solve']
            require(witness['legal_in_54'] is True, 'Optional embedded 14 witness is classified as illegal')
            same(witness['objective_value_54'], case['methods']['arms14']['objective_value'],
                 'Embedded theoretical 54 witness cost', atol=1e-8)
    require(sum('strict_radial_lower_bound_m' in x for x in table) == 36 and
            sum('strict_radial_lower_bound_m' in x for x in table if x['cohort'] == 'ROOT') == 32,
            'Changed M1 target-qualified denominator')
    groups = ('All', 'ROOT', 'TAIL', 'Train', 'Validation', 'Train_ROOT', 'Validation_ROOT',
              'Train_TAIL', 'Validation_TAIL')
    summary = {g: {m: summarize(table, g, m) for m in ('arms14', 'whole54')} for g in groups}
    for g in groups:
        for mode, row in summary[g].items():
            reference = pair['summary'][g]['modes'][mode]
            require(row['planned'] == reference['qualified_denominator'] and row['scored'] == reference['solved'] and
                    row['within_threshold_mm'] == reference['hits_mm'], f'Independent aggregate mismatch in {g}/{mode}')
    for group in ('ROOT', 'All'):
        selected = [r['delta_54_minus_14_m'] for r in table
                    if (group == 'All' or r['cohort'] == group) and 'delta_54_minus_14_m' in r]
        summary[group]['paired'] = {
            'count': len(selected), 'improved': sum(x < -Q_TOL for x in selected),
            'worsened': sum(x > Q_TOL for x in selected), 'tied': sum(abs(x) <= Q_TOL for x in selected),
            'delta_m_P50_P95_P99_max': np.percentile(selected, [50, 95, 99, 100]).tolist() if selected else None,
        }
    result = {
        'schema': 's05.p02.independent_static_audit.v1',
        'status': 'PASS_STATIC_RESEARCH_ONLY', 'reviewed_baseline': BASELINE, 'executed_head': git_sha,
        'authorized_p03': False, 'formal_scoring_freeze': 'PENDING_HUMAN_APPROVAL',
        'signed_artifacts_sha256': SIGNED, 'verified_model_assets': assets,
        'original_archive_sha256': digest(original), 'matched_archive_sha256': digest(matched),
        'mass_geometry_distortion_checks': comparisons, 'source_cohorts': {'ROOT': 32, 'TAIL': 6},
        'pairedscored': sum('delta_54_minus_14_m' in r for r in table),
        'strict_lower_bound_is_necessary_condition_only': True,
        'legal_q_is_constructive_upper_bound_only': True, 'summary': summary, 'rows': table,
        'baseline_research_only_m1_multitask_max_tip_m': signed_local['matched']['M1']['ROOT']['multitask'][
            'distribution']['max_tip_m'],
    }
    return result


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--matched', type=Path, required=True)
    p.add_argument('--pairing', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = review(args.repo, args.original, args.matched, read(args.pairing))
    result['agent_a_result_sha256'] = digest(args.pairing)
    with args.output.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(result['status'], 'independently rechecked pairs:', result['pairedscored'])


if __name__ == '__main__':
    main()
