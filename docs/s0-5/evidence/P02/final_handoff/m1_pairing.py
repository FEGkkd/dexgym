"""Replay the signed M1 static task under matched 14/54 DoF search budgets."""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import mujoco
import numpy as np
import scipy

from dexgym.tasks.direct.wholebody_s05.p02.intent import Target
from dexgym.tasks.direct.wholebody_s05.p02.model import PALMS, WRISTS, RobotModel, asset_files
from dexgym.tasks.direct.wholebody_s05.p02.model import digest, independent_frames
from dexgym.tasks.direct.wholebody_s05.p02.oracle import Objective, metrics, solve


P02 = Path('docs/s0-5/evidence/P02')
WORK_PLAN = P02/'final_handoff/work_plan.json'
REVIEWED_SHA = 'ecd12ecd6657539e44830dde07c6055c6c507192'
REFINEMENT_RECEIPT_SHA256 = '0adfc8976ebfa7ee0845d42e5d291caa70f94cf172e72d4a7d2fcb3c16a15aa0'
FITTED_SHA256 = '577195a8eb66699e11cf8bcf910383086672c384723d61fa90c13ed90652baa7'
SCORING_CANDIDATE_SHA256 = '96aac52666d7b03d6a3c087cb845f824a2ed4403f1cf35ae87af0264dd5f4f94'
MATCHED_KEY = 'matched-01/matched_candidates.json'
ORIGINAL_KEY = 'intent-02/study_records.json'
THRESHOLDS_MM = (3, 5, 10, 15)
MODES = ('arms14', 'whole54')
SCOPES = ('All', 'ROOT', 'TAIL', 'Train', 'Validation', 'Train_ROOT', 'Validation_ROOT',
          'Train_TAIL', 'Validation_TAIL')
START_LABELS = ('SDK_AND_WRIST_IK', 'JOINT_RESIDUAL_SEED', 'PERTURBED_HOME')
ITERATIONS = 300
WALL_SECONDS = 8.0
FK_TOLERANCE = 2e-9
DELTA_TIE = 1e-9
JSON_SCHEMA = 's05.p02.m1_pairing.v1'


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def read(path, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError(f'Signed fingerprint differs: {path}')
    return json.loads(Path(path).read_bytes())


def signed_inputs(receipt_path, original_path, matched_path):
    work_plan = read(WORK_PLAN)
    pairing = work_plan['M1_pair']
    if (work_plan['reviewed_baseline'] != REVIEWED_SHA or pairing['iterations_per_start'] != ITERATIONS or
            pairing['seconds_per_start'] != WALL_SECONDS or pairing['starts_per_mode'] != len(START_LABELS) or
            pairing['clearance_m'] != .0005 or pairing['matched_M2'] != 'NOT_RUN' or
            any(work_plan['execution_authorizations'].values())):
        raise ValueError('The final P02-only work plan and the audited pair protocol differ')
    receipt = read(receipt_path, REFINEMENT_RECEIPT_SHA256)
    if receipt['schema'] != 's05.p02.independent_refinement.receipt.v1' or receipt['authorized_p03']:
        raise ValueError('Invalid offline refinement receipt')
    base = read(P02/'plan.json')
    refine = read(P02/'refinement_plan.json', receipt['plan_sha256'])
    original_receipt = read(P02/'review_receipt.json', refine['reference_review_receipt_sha256'])
    if original_receipt['plan_sha256'] != digest(P02/'plan.json'):
        raise ValueError('Original P02 solver plan is not the reviewed plan')
    original_archive = read(P02/'output_sha256.json', original_receipt['receipts']['artifact_manifest'])
    original = read(original_path, original_archive[ORIGINAL_KEY])
    outputs = receipt['output_sha256']
    archive = read(receipt_path.with_name('archive_sha256.json'), outputs['archive_sha256.json'])
    fitted = read(receipt_path.with_name('fitted_candidates.json'), FITTED_SHA256)
    scoring = read(receipt_path.with_name('scoring_freeze_candidate.json'), SCORING_CANDIDATE_SHA256)
    assets = read(receipt_path.with_name('model_assets_sha256.json'), outputs['model_assets_sha256.json'])
    matched = read(matched_path, archive[MATCHED_KEY])
    if (outputs['fitted_candidates.json'] != FITTED_SHA256 or
            outputs['scoring_freeze_candidate.json'] != SCORING_CANDIDATE_SHA256 or
            receipt['train_frozen_candidates_sha256'] != FITTED_SHA256 or
            matched['candidates_sha256'] != FITTED_SHA256 or scoring['fit_sha256'] != FITTED_SHA256 or
            fitted['plan_sha256'] != receipt['plan_sha256']):
        raise ValueError('The M1 target rows and fitted Train candidate identities differ')
    if scoring['p03_authorized'] or scoring['formal_scoring_freeze'] != 'PENDING_HUMAN_APPROVAL':
        raise ValueError('Expected the immutable prior research candidate, with no P03 authorization')
    if fitted['fit_scope'] != 'Train' or fitted['optimizer']['M1']['train_roots'] != 24:
        raise ValueError('Expected the frozen M1 fit on 24 Train roots')
    configs = fitted['configs']
    m0, m1 = configs['M0'], configs['M1']
    if m0['scale'] != 1 or m1['scale'] != 1 or m0['mount'] != m1['mount']:
        raise ValueError('M1 must retain shared unit scale and the two signed fixed wrist mounts')
    if (base['clearance_m'] != .0005 or refine['solve']['static_clearance_m'] != .0005 or
            base['solver']['wall_seconds_per_start'] != WALL_SECONDS or
            refine['solve']['max_seconds_per_start'] != WALL_SECONDS or
            refine['solve']['matched_candidates_iterations'] != ITERATIONS or tuple(base['solver']['starts']) != START_LABELS):
        raise ValueError('The fixed model margin, three start rules or matched budgets differ')
    if not (base['model_sha256'] == receipt['model_sha256'] == scoring['model_sha256'] ==
            fitted['model_hash'] == digest(Path(base['model']))):
        raise ValueError('The frozen model identity differs')
    if asset_files(base['model']) != assets:
        raise ValueError('The compiled MJCF asset cascade differs from the signed model')
    paired = frozen_pairs(original, matched['rows'])
    shas = {
        'refinement_receipt': REFINEMENT_RECEIPT_SHA256, 'fitted_candidates': FITTED_SHA256,
        'refinement_scoring_candidate': SCORING_CANDIDATE_SHA256, 'refinement_archive': outputs['archive_sha256.json'],
        'original_receipt': refine['reference_review_receipt_sha256'], 'original': original_archive[ORIGINAL_KEY],
        'matched': archive[MATCHED_KEY], 'model': base['model_sha256'], 'plan': digest(P02/'plan.json'),
        'refinement_plan': receipt['plan_sha256'], 'model_assets': outputs['model_assets_sha256.json'],
        'final_handoff_work_plan': digest(WORK_PLAN)}
    return base, paired, (m0, m1), shas


def frozen_pairs(original, matched):
    ids = [row['identity'] for row in original]
    if ids != [row['identity'] for row in matched]:
        raise ValueError('The full ordered original identity records are not preserved')
    expected = {('ROOT', 'Train'): 24, ('ROOT', 'Validation'): 8}
    observed = Counter((x['cohort'], x['split']) for x in ids)
    if (len(ids) != 38 or len({x['key'] for x in ids}) != 38 or
            any(observed[key] != n for key, n in expected.items()) or
            sum(x['cohort'] == 'TAIL' and x['split'] in ('Train', 'Validation') for x in ids) != 6):
        raise ValueError('Expected the signed 32 ROOT (24 Train/8 Validation) plus six TAIL cases')
    return list(zip(original, matched, strict=True))


def target(d):
    result = Target(**{field.name: np.asarray(d[field.name], dtype=float) for field in fields(Target)})
    if not all(np.isfinite(x).all() for x in vars(result).values()):
        raise ValueError('Nonfinite signed target')
    return result


def validate_targets(original, candidate, mapping_configs):
    """Independently check the signed M1 target as the frozen rigid transform of M0."""
    prev, new = mapping_configs
    r0, r1 = np.asarray(prev['rotation']), np.asarray(new['rotation'])
    r = r1 @ r0.T
    t = np.asarray(new['translation']) - r @ np.asarray(prev['translation'])
    if (not np.allclose(r.T @ r, np.eye(3), atol=1e-10, rtol=0) or
            abs(np.linalg.det(r) - 1.) > 1e-10):
        raise ValueError('The signed M0 to M1 transform is not a proper rigid rotation')
    expected = dict(world=original.world @ r.T + t, wrist=original.wrist @ r.T + t,
                    rotation=r @ original.rotation, local=original.local, pinch=original.pinch,
                    cross=original.cross @ r.T, relative_wrist=original.relative_wrist,
                    relative_rotation=original.relative_rotation)
    deltas = {name: float(np.max(np.abs(getattr(candidate, name) - array))) for name, array in expected.items()}
    if max(deltas.values()) > FK_TOLERANCE:
        raise ValueError(f'The signed M1 row disagrees with the shared rigid transform: {deltas}')
    return deltas


def bitwise_match(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')


class GuardedRobot(RobotModel):
    """Observe every state used by the existing solver's 14-coordinate subspace."""
    fixed_hands = None
    checked_states = 0

    @contextmanager
    def fixed(self, hands):
        self.fixed_hands = np.asarray(hands, dtype=np.float64).copy()
        self.checked_states = 0
        try:
            yield
        finally:
            self.fixed_hands = None

    def set(self, q, contacts=False):
        if self.fixed_hands is not None:
            if not bitwise_match(np.asarray(q)[14:], self.fixed_hands):
                raise ValueError('Solver changed a bit of the fixed 40-DoF SDK vector')
            self.checked_states += 1
        super().set(q, contacts=contacts)


def hold_sdk_hands(robot, original_sdk):
    sdk = np.asarray(original_sdk, dtype=np.float64)
    if sdk.shape != (40,) or not np.isfinite(sdk).all():
        raise ValueError('Expected 40 finite, signed SDK hand coordinates')
    limits = robot.bounds[14:]
    fixed = np.clip(sdk, limits[:, 0], limits[:, 1])
    clips = [dict(index=int(j), joint=robot.names[14+j], unclipped_rad=float(sdk[j]), frozen_rad=float(fixed[j]),
                  delta_rad=float(fixed[j] - sdk[j]), bound='lower' if sdk[j] < limits[j, 0] else 'upper',
                  legal_intersection_rad=limits[j].tolist())
             for j in np.flatnonzero(sdk != fixed)]
    return fixed, dict(unclipped_q_rad=sdk.tolist(), frozen_q_rad=fixed.tolist(), clips=clips)


def paired_starts(original, variant, fixed):
    attempts = variant['multitask']['attempts']
    if tuple(a['label'] for a in attempts) != START_LABELS:
        raise ValueError('Expected the three signed matched M1 multitask initialization labels')
    starts54 = [(a['label'], np.asarray(a['input_initial'], dtype=np.float64)) for a in attempts]
    if any(q.shape != (54,) or not np.isfinite(q).all() for _, q in starts54):
        raise ValueError('Invalid signed M1 initialization')
    orig_pert = original['whole54']['attempts'][2]
    if (orig_pert['label'] != START_LABELS[2] or not bitwise_match(starts54[2][1], orig_pert['input_initial']) or
            not bitwise_match(starts54[0][1][14:], fixed)):
        raise ValueError('Perturbed-home identity or clipped SDK hands in the signed wrist seed differs')
    starts14 = [(label, np.concatenate((q[:14], fixed))) for label, q in starts54]
    return starts14, starts54


def verify_frozen_attempts(result, starts, fixed):
    """Audit the exact recorded inputs, clipped starts, attempt winners and final 14 q."""
    if len(result['attempts']) != len(starts):
        raise ValueError('Expected exactly the prescribed attempts')
    for attempt, (label, q) in zip(result['attempts'], starts, strict=True):
        if attempt['label'] != label or not bitwise_match(attempt['input_initial'], q):
            raise ValueError('Fixed-hand attempt source seed differs')
        for name in ('input_initial', 'initial', 'best_q'):
            if name in attempt and not bitwise_match(attempt[name][14:], fixed):
                raise ValueError(f'Fixed-hand vector did not survive {name} for {label}')
    if result['q'] is not None and not bitwise_match(result['q'][14:], fixed):
        raise ValueError('Fixed-hand vector did not survive final witness selection')


def fk_witness(reference, goal, result, config, lower_bound):
    if result['q'] is None:
        return None
    q = np.asarray(result['q'], dtype=float)
    g = reference.geometry(q)
    frames = independent_frames(reference, q)
    sites = np.asarray([frames['site', name][0] for name in WRISTS])
    turns = np.asarray([frames['site', name][1] for name in WRISTS])
    palms = np.asarray([frames['body', name][0] for name in PALMS])
    palm_r = np.asarray([frames['body', name][1] for name in PALMS])
    landmarks = np.asarray([
        [[frames['site' if i == 3 else 'joint', name][0] for i, name in enumerate(finger)] for finger in hand]
        for hand in reference.landmark_names])
    local = (landmarks - palms[:, None, None]) @ palm_r[:, None]
    tip_max = float(np.linalg.norm(landmarks[:, :, -1] - goal.world[:, :, -1], axis=-1).max())
    errors = dict(wrist_position_m=float(np.max(np.abs(g.wrists - sites))),
        wrist_rotation_component=float(np.max(np.abs(g.rotations - turns))),
        world_landmark_m=float(np.max(np.abs(g.landmarks - landmarks))),
        local_landmark_m=float(np.max(np.abs(g.local - local))),
        tip_residual_m=abs(tip_max - result['metrics']['max_tip_m']))
    legal = reference.legality(q)
    cost = Objective(reference, goal, config).cost(q)
    if (max(errors.values()) > FK_TOLERANCE or not legal['feasible'] or tip_max + FK_TOLERANCE < lower_bound or
            not np.isclose(result['objective_value'], cost, atol=1e-9, rtol=1e-10)):
        raise ValueError(f'Independent witness verification failed: {errors}, legal={legal}, cost={cost}')
    return dict(fk_component_errors=errors, legality=legal, objective_value=cost,
                max_tip_m=tip_max, signed_radial_lower_bound_m=lower_bound)


def scalar_metrics(result, close_pairs):
    if not result or result['q'] is None:
        return {}
    data = result['metrics']
    values = {k: float(data[k]) for k in ('max_tip_m', 'mean_tip_m', 'local_landmark_rmse_m',
              'relative_wrist_translation_m', 'relative_wrist_rotation_rad')}
    values['multitask_cost'] = float(result['objective_value'])
    for k in ('pinch_vector_m', 'pinch_distance_m', 'cross_hand_m', 'wrist_translation_m', 'wrist_rotation_rad'):
        values[k.replace('_m', '_max_m') if k.endswith('_m') else 'wrist_rotation_max_rad'] = float(np.max(data[k]))
    if close_pairs:
        for k in ('pinch_vector_m', 'pinch_distance_m'):
            values['active_' + k.replace('_m', '_max_m')] = max(float(data[k][s][f]) for s, f in close_pairs)
    return values


def within_scope(row, scope):
    x = row['identity']
    return scope in ('All', x['split'], x['cohort'], x['split'] + '_' + x['cohort'])


def percentiles(values):
    return dict(zip(('P50', 'P95', 'P99'), map(float, np.percentile(values, [50, 95, 99])), strict=True)) if values else None


def summarize(rows):
    out = {}
    for scope in SCOPES:
        cases = [r for r in rows if within_scope(r, scope)]
        both = [r for r in cases if r.get('paired_delta_54_minus_14')]
        report = dict(planned=len(cases), strictly_paired=len(both), modes={})
        for mode in MODES:
            solutions = [r['results'][mode] for r in cases]
            values = [a.get('scalar_metrics', {}) for a in solutions]
            keys = sorted({k for v in values for k in v})
            attempts = [x for s in solutions for x in s.get('attempts', [])]
            report['modes'][mode] = dict(
                solved=sum(bool(x) for x in values),
                statuses=dict(Counter(r['status'] for r in solutions)),
                qualified_denominator=len(cases),
                hits_mm={str(t): sum(v.get('max_tip_m', float('inf')) <= t/1000 for v in values) for t in THRESHOLDS_MM},
                metrics={k: dict(n=sum(k in v for v in values), percentiles=percentiles([v[k] for v in values if k in v]))
                         for k in keys},
                attempts=dict(n=len(attempts), converged=sum(a['solver_success'] for a in attempts),
                              budget_exhausted=sum(a['budget_exhausted'] for a in attempts),
                              feasible=sum(a['feasible_best'] for a in attempts),
                              seconds=percentiles([a['elapsed_s'] for a in attempts]),
                              iterations=percentiles([a['iterations'] for a in attempts]),
                              termination_codes=dict(Counter(a['termination'] for a in attempts))))
        deltas = [r['paired_delta_54_minus_14'] for r in both]
        keys = sorted({k for d in deltas for k in d})
        report['paired_metrics'] = {}
        for key in keys:
            diff = [d[key] for d in deltas if key in d]
            report['paired_metrics'][key] = dict(n=len(diff), percentiles=percentiles(diff),
                improves=sum(x < -DELTA_TIE for x in diff), worsens=sum(x > DELTA_TIE for x in diff),
                stable=sum(abs(x) <= DELTA_TIE for x in diff))
        report['sdk_clips'] = dict(
            coordinates=sum(len(r.get('sdk_hands_hold', {}).get('clips', [])) for r in cases),
            cases=sum(bool(r.get('sdk_hands_hold', {}).get('clips', [])) for r in cases))
        out[scope] = report
    return out


def real_pair(robot, reference, config, original, variant, registration):
    goal = target(variant['mapped'])
    validation = validate_targets(target(original['mapped']), goal, registration)
    fixed, hold = hold_sdk_hands(robot, original['sdk_wrist_ik']['sdk_unclipped_q'])
    starts14, starts54 = paired_starts(original, variant, fixed)
    near = [(s, f) for s, row in enumerate(original['mapping']['hands']) for f in row['near_pinch_pairs']]
    lower = variant['strict_radial_bound_m']
    with robot.fixed(fixed):
        a = solve(robot, goal, config, starts14, indices=np.arange(14))
    checked = robot.checked_states
    verify_frozen_attempts(a, starts14, fixed)
    b = solve(robot, goal, config, starts54)
    result = dict(identity=original['identity'], status='ATTEMPTED', target_M1=variant['mapped'],
                  rigid_transform_error=validation, sdk_hands_hold=hold, active_pinch_pairs=near,
                  constrained_geometry_evaluations=checked, signed_radial_lower_bound_m=lower,
                  results={'arms14': a, 'whole54': b})
    for mode in MODES:
        solution = result['results'][mode]
        solution['verification'] = fk_witness(reference, goal, solution, config, lower)
        solution['scalar_metrics'] = scalar_metrics(solution, near)
    ma, mb = (result['results'][mode]['scalar_metrics'] for mode in MODES)
    result['paired_delta_54_minus_14'] = {k: mb[k] - ma[k] for k in ma.keys() & mb.keys()}
    if a['q'] is not None:
        result['embedded_14_witness_no_extra_solve'] = dict(
            legal_in_54=a['verification']['legality']['feasible'],
            objective_value_54=Objective(reference, goal, config).cost(np.asarray(a['q'])),
            relation='The 14 witness is admissible to the 54 problem. This cost-only check leaves the fixed three attempts unchanged.')
    return result


def run(args):
    start = perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    baseline = git('rev-parse', '--verify', f'{REVIEWED_SHA}^{{commit}}')
    subprocess.run(['git', 'merge-base', '--is-ancestor', baseline, 'HEAD'], check=True)
    config, signed, registration, shas = signed_inputs(args.refinement_receipt, args.original, args.matched)
    config['solver'] |= dict(max_iterations_per_start=ITERATIONS, wall_seconds_per_start=WALL_SECONDS)
    robot = GuardedRobot(config['model'], config['clearance_m'])
    reference = RobotModel(config['model'], config['clearance_m'])
    rows = []
    for original, matched in signed:
        ident = original['identity']
        if 'M1' not in matched.get('candidates', {}):
            if 'mapped' in original:
                raise ValueError(f'A previously mapped case lost its signed M1 target: {ident["key"]}')
            status = 'UNKNOWN_INPUT'
            row = dict(identity=ident, status=status, original_disposition=original['disposition'],
                       results={mode: dict(status=status, q=None, attempts=[]) for mode in MODES})
        else:
            row = real_pair(robot, reference, config, original, matched['candidates']['M1'], registration)
        rows.append(row)
        print(ident['key'], {mode: r.get('metrics', {}).get('max_tip_m', r['status'])
                             for mode, r in row['results'].items()}, flush=True)
    report = dict(
        schema=JSON_SCHEMA, approved_input_commit=baseline, execution_git=git('rev-parse', 'HEAD'),
        started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=perf_counter() - start,
        dataset_scope='HISTORICAL_HMB_OFFLINE_STATIC_RESEARCH', formal_score_authorized=False, p03_authorized=False,
        input_paths={k: str(getattr(args, k)) for k in ('refinement_receipt', 'matched', 'original')},
        source_sha256=shas | {'pairing_script': digest(Path(__file__))}, command=sys.argv,
        runtime=dict(python=platform.python_version(), python_path=sys.executable, numpy=np.__version__,
                     scipy=scipy.__version__, mujoco=mujoco.__version__, platform=platform.platform()),
        seed_protocol=dict(labels=list(START_LABELS), identity='Exact archived signed M1 multitask input seeds',
                           arms14='Use the same 14 arm coordinates and hold clipped SDK hand coordinates in all evaluations',
                           whole54='Use the full native archived 54 seed for each matched label',
                           arm_ranges='Full legal compiled arm joint / actuator range intersections with J6/J7 coupling',
                           iterations_per_start=ITERATIONS, wall_seconds_per_start=WALL_SECONDS,
                           clearance_m=config['clearance_m'], selected_by='Frozen multi-task objective',
                           source_seed_not_regenerated=True),
        thresholds_mm=list(THRESHOLDS_MM), thresholds_denominator='All planned cases in each stratum',
        percentile_denominator='Successful legal witnesses per metric. Paired deltas require both witnesses.',
        paired_delta='54 minus 14. Negative means lower error or cost.',
        paired_delta_tie=DELTA_TIE, rows=rows, summary=summarize(rows))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('M1 pairing archive', args.output, digest(args.output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('refinement-receipt', 'matched', 'original', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
