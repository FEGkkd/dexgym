"""Recheck static witnesses and seal the P02 evidence without modifying completed runs."""
from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, digest, independent_frames
from dexgym.tasks.direct.wholebody_s05.p02.scoring import position_errors
from run_study import independent_q, load, samples, verify_approval, write


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def pt(values):
    return np.percentile(values, [50, 95, 99, 100]).tolist() if len(values) else None


def rot_gap(first, second):
    return Rotation.from_matrix(first.reshape(-1, 3, 3).transpose(0, 2, 1)
                                @ second.reshape(-1, 3, 3)).magnitude()


def reach_envelope(robot):
    """Use the triangle inequality along each chain, independent of solver and limits."""
    model = robot.model
    robot.set(np.zeros(54))
    bases, radii = [], []
    for site_id in robot.tip_ids:
        site = model.site(site_id)
        nodes = [robot.data.site(site_id).xpos.copy()]
        body = int(site.bodyid[0])
        while body:
            count = int(model.body_jntnum[body])
            if count > 1:
                raise ValueError('Reach bound requires a serial tree of single hinges')
            if count:
                nodes.append(robot.data.joint(int(model.body_jntadr[body])).xanchor.copy())
            body = int(model.body_parentid[body])
        if len(nodes) < 2:
            raise ValueError('Tip has no associated chain')
        bases.append(nodes[-1])
        radii.append(float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()))
    return np.array(bases).reshape(2, 5, 3), np.array(radii).reshape(2, 5)


def check_targets(config, robot, out, records):
    archives = {name: read_arrays(out / (name + '.npz')) for name in
                ('mapped_intent', 'feasible_scoring_target', 'projection_error', 'q_oracle_offline')}
    mapped, feasible, projection, joints = archives.values()
    view = load(out / 'view_manifest.json')
    for name, checksum in view['files'].items():
        if digest(out / name) != checksum:
            raise ValueError(f'View manifest disagrees with archived bytes: {name}')
    root = [row for row in records if row['identity']['cohort'] == 'ROOT']
    counts = Counter(row['identity']['split'] for row in root)
    roots = {row['identity']['root'] for row in root}
    keys = {row['identity']['key'] for row in records}
    if counts != {'Train': 24, 'Validation': 8} or len(roots) != 32 or len(keys) != 38:
        raise ValueError('Source denominators differ')
    intended = [row for row in records if 'mapped' in row]
    witnesses = [row for row in records if 'feasible' in row]
    if list(mapped['case_key']) != [row['identity']['key'] for row in intended]:
        raise ValueError('Unpaired mapped cases')
    ordered = [row['identity']['key'] for row in witnesses]
    if any(list(x['case_key']) != ordered for x in (feasible, projection, joints)):
        raise ValueError('Unpaired feasible cases')
    if list(joints['joint_names']) != list(robot.names):
        raise ValueError('Offline joint names differ')
    mapped_order = [list(mapped['case_key']).index(k) for k in ordered]
    for name in ('world', 'local', 'wrist', 'pinch', 'cross', 'relative_wrist'):
        expected = position_errors(mapped[name][mapped_order], feasible[name])['projection_error']
        if not np.array_equal(expected, projection[name]):
            raise ValueError(f'Incorrect signed projection for {name}')
    r_summary = {name: pt(rot_gap(mapped[name][mapped_order], feasible[name])) for name in
                 ('rotation', 'relative_rotation')}
    for archive in (mapped, feasible):
        for name in r_summary:
            rotations = archive[name].reshape(-1, 3, 3)
            if not (np.isfinite(rotations).all() and np.allclose(np.linalg.det(rotations), 1., atol=1e-10)
                    and np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-10)):
                raise ValueError(f'Invalid SO(3) in {name}')
    bound_base, radii = reach_envelope(robot)
    audited = []
    for i, row in enumerate(witnesses):
        q = joints['q_rad'][i]
        tree = independent_frames(robot, q)
        tips = np.array([tree['site', robot.model.site(k).name][0]
                         for k in robot.tip_ids]).reshape(2, 5, 3)
        if not np.allclose(tips, feasible['world'][i, :, :, -1], atol=2e-9, rtol=0):
            raise ValueError('Archived target differs from independent tip FK')
        constraints = robot.legality(q)
        if not constraints['feasible']:
            raise ValueError(f'Archived q violates static constraints: {constraints}')
        distance_from_pivot = np.linalg.norm(tips - bound_base, axis=-1)
        if (distance_from_pivot > radii + 1e-9).any():
            raise ValueError('Analytic reach envelope does not enclose a static witness')
        goals = mapped['world'][mapped_order[i], :, :, -1]
        excess = np.maximum(0, np.linalg.norm(goals - bound_base, axis=-1) - radii - 1e-9)
        audited.append(dict(case_key=ordered[i], root=row['identity']['root'], split=row['identity']['split'],
                            lower_bound_tip_m=excess.tolist(), lower_bound_max_m=float(excess.max()),
                            active_model_ranges=np.asarray(robot.names)[
                                (np.abs(q[:, None] - robot.bounds) <= 1e-5).any(axis=1)].tolist(),
                            coupling_active=constraints['coupling_margin_rad'] <= 1e-5,
                            collision_near_margin=constraints['collision_clearance_m'] <= robot.clearance_m + 1e-5,
                            margin=constraints))
    return dict(root_counts=counts, all_case_count=len(records), full_q_witnesses=len(witnesses),
                tip_fk_recheck='PASS', vector_decomposition='PASS', rotations_rad=r_summary,
                reach_formula='Triangle inequality over first hinge pivot, successive fixed link lengths and functional tip. Positive result bounds the error even before other constraints.',
                reach_radius_m=radii.tolist(), rechecked=audited)


def mapping_and_solver(records, selected):
    result = {}
    for label in ('All', 'Train', 'Validation', 'ROOT', 'TAIL'):
        planned = [r for r in records if label in ('All', r['identity']['split'], r['identity']['cohort'])]
        mapped = [r for r in planned if 'mapped' in r]
        eligible = [r for r in mapped if 'feasible' in r]
        hands = [h for r in mapped for h in r['mapping']['hands']]
        paired = [r for r in eligible if 'metrics' in r.get('arms14', {})]
        arms = [r['arms14']['metrics']['max_tip_m'] for r in paired]
        full = [r['whole54']['metrics']['max_tip_m'] for r in paired]
        details = {}
        for key in ('max_tip_m', 'mean_tip_m', 'tip_m', 'pinch_vector_m', 'pinch_distance_m',
                    'wrist_translation_m', 'wrist_rotation_rad', 'relative_wrist_translation_m',
                    'relative_wrist_rotation_rad', 'cross_hand_m', 'local_landmark_rmse_m'):
            values = np.array([r['whole54']['metrics'][key] for r in eligible])
            details[key] = pt(values.reshape(-1))
        result[label] = dict(mapped=len(mapped), near_pinch_pairs=sum(len(h['near_pinch_pairs']) for h in hands),
                             active_pinch_before_m=pt([h['before_error_m'][i] for h in hands for i in h['near_pinch_pairs']]),
                             active_pinch_after_m=pt([h['after_error_m'][i] for h in hands for i in h['near_pinch_pairs']]),
                             bone_length_abs_delta_m=pt(np.abs(np.array([h['bone_length_delta_m'] for h in hands])).ravel()),
                             wrist_baseline_distortion_m=pt([r['mapping']['relative_wrist_translation_delta_m'] for r in mapped]),
                             geometric_projection=details,
                             paired_count=len(paired), paired_arms_m=pt(arms), paired_full_m=pt(full),
                             paired_delta_54_minus_14_m=pt(np.asarray(full)-arms),
                             improved_cases=sum(b<a for a,b in zip(arms,full)),
                             all_status=Counter(r.get('whole54',{}).get('status', r.get('disposition','UNKNOWN'))
                                                for r in planned))
    attempts = [a for r in records for mode in ('arms14','whole54') for a in r.get(mode, {}).get('attempts', ())]
    result['optimization'] = dict(attempts=len(attempts), durations_seconds=pt([a['elapsed_s'] for a in attempts]),
                                  iterations=pt([a['iterations'] for a in attempts]),
                                  budget_exhausted=sum(bool(a['budget_exhausted']) for a in attempts),
                                  solver_success=sum(bool(a['solver_success']) for a in attempts),
                                  finite_feasible_witness_attempts=sum(bool(a['feasible_best']) for a in attempts))
    result['sample_motion'] = [
        dict(case_key=ident['key'], split=ident['split'], task=ident['task'],
             sum_four_stream_peak_proxy_m_s=row['proxy_motion'] if row and row['proxy_motion'] != 1e9 else None)
        for ident, row in selected
    ]
    return result


def world_plot(records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    case = next(r for r in records if r['identity']['task']=='bi_hand' and 'feasible' in r)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection='3d')
    for side, color, label in ((0, '#2563a6', 'Left'), (1, '#bd551f', 'Right')):
        for kind, style in (('mapped', '-'), ('feasible', '--')):
            target = case[kind]
            points = np.array(target['world'])[side]
            wrist = np.array(target['wrist'])[side]
            for j, line in enumerate(points):
                ax.plot(*line.T, style, color=color, marker='o', markersize=2,
                        label=f'{label} {kind}' if j==0 else None)
            ax.scatter(*wrist, marker='s', color=color)
    both = np.concatenate([np.array(case[role]['world']).reshape(-1,3) for role in ('mapped','feasible')])
    ax.set_box_aspect(np.ptp(both,axis=0))
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title='Bimanual target and static projection · simulation world')
    ax.legend()
    fig.text(.05,.04,f"{case['identity']['partition']} · tick {case['identity']['tick']} · historical HMB")
    fig.savefig(output,dpi=145)
    plt.close(fig)


def main(plan_path, run, sealed):
    config = load(plan_path)
    handoff = verify_approval(config)
    subprocess.run(['sha256sum','-c',str(run/'frozen.sha256')],check=True)
    accepted = ('model-02','intent-02','history-02','tests-02','pinch-01','constraints-01')
    names = ('model-01','intent-01','history-01','tests-01') + accepted
    evidence = {}
    for name in names:
        p = run/name
        rc = int((p/'exit').read_text())
        if name in accepted and rc:
            raise ValueError(f'Required experiment failed: {name}')
        logs = ('stdout','stderr','runtime.json','resources.txt')
        missing = [x for x in logs if not (p/x).is_file()]
        if name in accepted and missing:
            raise ValueError(f'Missing required runtime evidence for {name}: {missing}')
        evidence[name] = dict(exit=rc, git=(p/'git_head').read_text().strip(), command=(p/'command').read_text(),
                             logs={x:digest(p/x) for x in logs if x not in missing}, missing_logs=missing,
                             started=(p/'started_at').read_text().strip(), ended=(p/'ended_at').read_text().strip())
    model_report = load(run/'model-02/model_report.json')
    if not all(report['valid'] for report in model_report.values()):
        raise ValueError('Model audit did not pass')
    history = load(run/'history-02/historical_hard_case.json')
    if len(history['cases']) != config['historical_hold']['max_cases'] or not history['no_final_test_access']:
        raise ValueError('Unexpected historical diagnostic scope')
    robot = RobotModel(config['model'], config['clearance_m'])
    records = load(run/'intent-02/study_records.json')
    audit = check_targets(config, robot, run/'intent-02', records)
    selected = samples(config, handoff)
    if [ident for ident,_ in selected] != [r['identity'] for r in records]:
        raise ValueError('Approved HMB sample binding has changed')
    summary = mapping_and_solver(records, selected)
    witness_plan = plan_path.with_name('pinch_witness_plan.json')
    pinch_report = load(run/'pinch-01/near_pinch_witness.json')
    if pinch_report['plan_sha256'] != digest(witness_plan):
        raise ValueError('Supplemental plan fingerprint differs')
    near_pinch = {
        split: {side: record['close_pose_ticks'] for side,record in sides.items()}
        for split,sides in pinch_report['strata'].items()
    }
    mapping_ok = all(sum(counts.values()) for counts in near_pinch.values())
    if summary['All']['active_pinch_after_m'] is not None and summary['All']['active_pinch_after_m'][-1] > 1e-10:
        raise ValueError('Near-pinch mapping did not preserve its fixed functional vectors')
    probe = load(run/'constraints-01/constraint_probe.json')
    if probe['plan_sha256'] != digest(plan_path.with_name('constraint_probe_plan.json')) or len(probe['results']) != 2:
        raise ValueError('Constraint counterfactual provenance differs')
    sealed.mkdir(mode=0o700)
    write(sealed/'score_integrity.json',audit)
    write(sealed/'mapping_and_solver_summary.json',summary)
    write(sealed/'execution_records.json',evidence)
    world_plot(records,sealed/'bimanual_projection.png')
    view = load(run/'intent-02/view_manifest.json')
    contracts = {
        'schema':'s05.p02.scoring_handoff.v1', 'run_id':config['run_id'],
        'E1_manifest_sha256':config['e1_manifest_sha256'], 'F1_handoff_sha256':config['handoff_sha256'],
        'scope':'HISTORICAL_HMB_STATIC_SIMULATION_DEVELOPMENT',
        'robot_model_sha256':config['model_sha256'], 'plan_sha256':digest(plan_path),
        'intent_generator_sha256':digest(Path('source/dexgym/tasks/direct/wholebody_s05/p02/intent.py')),
        'oracle_sha256':digest(Path('source/dexgym/tasks/direct/wholebody_s05/p02/oracle.py')),
        'scorer_sha256':digest(Path('source/dexgym/tasks/direct/wholebody_s05/p02/scoring.py')),
        'views':view, 'control_state_used_by_generator':False, 'offline_q_access':'OFFLINE_DIAGNOSTIC_ONLY',
        'online_actor_and_nominal_forbidden':['q_oracle_offline','future geometry','offline interpolation',
                                               'diagnostic tail labels','complete-trajectory metadata'],
        'retain_mapped_and_projection_in_scoring':True, 'dynamic_execution_measured':False,
        'P03_start_authorized':False, 'P03_required_independent_acceptance':['P02','P01-T','C2-RT','robot-base'],
        'operational_target_freeze':'PENDING_HUMAN_REVIEW_OF_P02_AND_WORKSPACE_DCR',
    }
    write(sealed/'P03_scoring_handoff.json',contracts)
    paths = [p for name in names for p in (run/name).iterdir() if p.is_file()]
    paths += [run/'frozen_at.txt',run/'frozen.sha256',*sealed.iterdir()]
    tracked = subprocess.check_output(['git','ls-files','-z',
        'source/dexgym/tasks/direct/wholebody_s05/p02','tests/s0_5/unit/test_p02_static.py',
        'docs/s0-5/evidence/P02'], text=True).split('\0')
    write(sealed/'code_sha256.json',{rel:digest(Path(rel)) for rel in tracked if rel})
    paths.append(sealed/'code_sha256.json')
    output = {str(p.relative_to(run)):digest(p) for p in sorted(paths)}
    write(sealed/'output_sha256.json',output)
    receipt = dict(schema='s05.p02.review_receipt.v1',run_id=config['run_id'],
                   execution_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   E1_manifest_sha256=config['e1_manifest_sha256'], F1_handoff_sha256=config['handoff_sha256'],
                   plan_sha256=digest(plan_path), scope=contracts['scope'],
                   spec_sha256=digest(Path('docs/s0-5/specs/P02_intent_and_feasibility.md')),
                   receipts={'artifact_manifest':digest(sealed/'output_sha256.json'),
                             'score_integrity':digest(sealed/'score_integrity.json'),
                             'scoring_handoff':digest(sealed/'P03_scoring_handoff.json')},
                   gates={**dict.fromkeys(('P02-A','P02-C','P02-D','P02-E'),'PASS'),
                          'P02-B':'PASS' if mapping_ok else 'BLOCKED',
                          'P02':'PASS' if mapping_ok else 'BLOCKED'},
                   outcomes=dict(planned_cases=len(records), feasible_witnesses=audit['full_q_witnesses'],
                                 unknown=len(records)-audit['full_q_witnesses']),
                   engineering_blocks=dict.fromkeys(('C2-RT','robot-base','P01-T','Final Test','HMB_REALTIME_VERIFIED'),'BLOCKED'),
                   P03_start_authorized=False, review_only=True)
    write(sealed/'review_receipt.json',receipt)
    print(receipt['gates']['P02'],'workflow evidence',len(output),'checksummed files',digest(sealed/'review_receipt.json'))


if __name__ == '__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
