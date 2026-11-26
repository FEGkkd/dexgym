"""Read-only verification of P02 static collision, range and coupling activity."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT, COUPLING_WEIGHTS
from dexgym.tasks.direct.wholebody_s05.p02.model import FINGERS, PALMS, RobotModel, digest


PLAN = Path('docs/s0-5/evidence/P02/plan.json')
DISTANCE_CAP_M = .015
ACTIVITY_EPSILON_M = 1e-5
ACTIVITY_EPSILON_RAD = 1e-5
LEGAL_GAP_TOLERANCE_M = 1e-7
LEGAL_JOINT_TOLERANCE_RAD = 1e-9
DISTANCE_AGREEMENT_M = 2e-9


def load(path):
    return json.loads(path.read_bytes())


def body_name(model, body):
    return model.body(int(body)).name or f'body#{body}'


def geometry_inventory(robot):
    """Derive components and digits from named joint ownership and MJCF ancestry."""
    model = robot.model
    anchors = {0: ('environment', None, None)}
    for side, arm, hand, palm, mount in zip(
            ('left', 'right'), robot.groups[:2], robot.groups[2:], PALMS, ('l_mount', 'r_mount'), strict=True):
        for name in arm:
            anchors[int(model.joint(name).bodyid[0])] = ('arm', side, None)
        for finger, names in zip(FINGERS, np.asarray(hand).reshape(5, 4), strict=True):
            for name in names:
                anchors[int(model.joint(str(name)).bodyid[0])] = ('hand', side, finger)
        for name in (palm, mount):
            anchors[model.body(name).id] = ('hand', side, None)
    for name in ('pedestal', 'center_mount', 'Base_L', 'Base_R'):
        anchors[model.body(name).id] = ('base', None, None)
    geoms = {}
    for geom, raw_body in enumerate(model.geom_bodyid):
        body = anchor = int(raw_body)
        while anchor not in anchors:
            anchor = int(model.body_parentid[anchor])
        component, side, finger = anchors[anchor]
        if anchor == 0 and body != 0:
            component = 'unclassified'
        geoms[geom] = dict(name=model.geom(geom).name or f'geom#{geom}', id=geom, body=body_name(model, body),
                           body_id=body, anatomy_anchor=body_name(model, anchor), component=component, side=side,
                           finger=finger, contype=int(model.geom_contype[geom]),
                           conaffinity=int(model.geom_conaffinity[geom]), type=int(model.geom_type[geom]))
    return geoms


def category(a, b):
    components = {a['component'], b['component']}
    if 'unclassified' in components:
        return 'unclassified'
    if 'environment' in components:
        return 'environment'
    if 'base' in components:
        return 'base_vs_' + next(iter(components - {'base'}), 'base')
    same_side = a['side'] == b['side']
    if components == {'arm', 'hand'}:
        return 'same_side_arm_hand' if same_side else 'opposite_side_arm_hand'
    if components == {'arm'}:
        return 'same_side_arm_links' if same_side else 'inter_arm'
    if not same_side:
        return 'cross_hand'
    if a['finger'] is None or b['finger'] is None:
        return 'same_hand_palm_or_mount'
    return 'same_finger_links' if a['finger'] == b['finger'] else 'same_hand_distinct_fingers'


def collision_inventory(robot, geoms):
    """Reconstruct compiled masks and compare with the frozen Oracle filter."""
    model = robot.model
    i, j = np.triu_indices(model.ngeom, k=1)
    bodies = np.asarray(model.geom_bodyid, dtype=np.int64)
    a, b = bodies[i], bodies[j]
    welds = np.asarray(model.body_weldid)
    wa, wb = welds[a], welds[b]
    parents = welds[model.body_parentid]
    excludes = np.isin(np.minimum(a, b) * 65536 + np.maximum(a, b), model.exclude_signature)
    compatible = ((model.geom_contype[i] & model.geom_conaffinity[j]) != 0) | (
        (model.geom_contype[j] & model.geom_conaffinity[i]) != 0)
    filter_parent = not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    parent_pair = (wa != 0) & (wb != 0) & ((parents[wa] == wb) | (parents[wb] == wa))
    dynamic = compatible & ~excludes & (wa != wb)
    if filter_parent:
        dynamic &= ~parent_pair
    explicit = {tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2, strict=True)}
    allowed = {(int(a), int(b)) for a, b in zip(i[dynamic], j[dynamic], strict=True)} | explicit
    reference = {tuple(pair) for pair in robot.collision_pairs}
    classes = {pair: category(*(geoms[k] for k in pair)) for pair in sorted(allowed)}
    inventory = dict(compiled_geometries=model.ngeom, possible_geom_pairs=len(i), oracle_allowed=len(reference),
        reconstructed_allowed=len(allowed), allowed_by_category=dict(Counter(classes.values())),
        explicit_geom_pairs=sorted(explicit),
        explicit_body_excludes=[[body_name(model, int(s) >> 16), body_name(model, int(s) & 0xffff)]
                                for s in model.exclude_signature],
        filter_parent=filter_parent, global_contact_disabled=bool(
            model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT),
        raw_filter_counts_overlap=dict(incompatible_contact_bits=int((~compatible).sum()),
            explicit_body_exclusion=int(excludes.sum()), same_weld_component=int((wa == wb).sum()),
            parent_weld_component=int(parent_pair.sum()) if filter_parent else 0),
        absent_from_oracle=sorted(allowed - reference),
        absent_from_compiled_filter=sorted(reference - allowed))
    return classes, inventory


def limits(robot, q):
    physical = np.asarray([joint.range for joint in robot.joints])
    drive = np.asarray([motor.ctrlrange for motor in robot.actuators])
    low, high = np.maximum(physical[:, 0], drive[:, 0]), np.minimum(physical[:, 1], drive[:, 1])
    slack = np.column_stack((q - low, high - q))
    active = []
    for i, j in zip(*np.where(slack <= ACTIVITY_EPSILON_RAD), strict=True):
        bound = (low if j == 0 else high)[i]
        active.append(dict(joint=robot.names[i], boundary=('lower', 'upper')[j], q_rad=float(q[i]),
                           bound_rad=float(bound), slack_rad=float(slack[i, j]),
                           source=[name for name, ranges in (('joint', physical), ('actuator', drive))
                                   if np.isclose(bound, ranges[i, j], rtol=0., atol=1e-11)]))
    coupling = {}
    for side, suffix in (('left', 'L'), ('right', 'R')):
        col = [robot.names.index(f'Joint{joint}_{suffix}') for joint in (6, 7)]
        coupling[side] = float(COUPLING_LIMIT - np.dot(COUPLING_WEIGHTS, np.abs(q[col])))
    return float(slack.min()), coupling, active


def audit_state(robot, q, classes, archived):
    """Run all enabled geom pairs directly, bypassing the Oracle's sphere pruning."""
    q = np.asarray(q, dtype=float)
    robot.set(q, contacts=True)
    model, data = robot.model, robot.data
    closest = None
    gap = DISTANCE_CAP_M
    near = []
    category_ranges = dict.fromkeys(set(classes.values()), DISTANCE_CAP_M)
    for (i, j), label in classes.items():
        distance = float(mujoco.mj_geomDistance(model, data, i, j, DISTANCE_CAP_M, None))
        if not np.isfinite(distance):
            raise ValueError(f'Nonfinite distance at geom pair {i}/{j}')
        category_ranges[label] = min(category_ranges[label], distance)
        if distance < gap:
            gap, closest = distance, (i, j)
        if distance <= robot.clearance_m + ACTIVITY_EPSILON_M:
            near.append(dict(geom_ids=(i, j), category=label, distance_m=distance, slack_m=distance - robot.clearance_m))
    engine = {}
    for c in data.contact:
        pair = tuple(sorted(map(int, c.geom)))
        engine[pair] = min(engine.get(pair, float('inf')), float(c.dist))
    penetration = min(0., *engine.values()) if engine else 0.
    engine_extras = sorted(set(engine) - classes.keys())
    slack, coupling, active_axes = limits(robot, q)
    legal = (gap >= robot.clearance_m - LEGAL_GAP_TOLERANCE_M and
             slack >= -LEGAL_JOINT_TOLERANCE_RAD and min(coupling.values()) >= -LEGAL_JOINT_TOLERANCE_RAD and
             penetration >= -LEGAL_GAP_TOLERANCE_M)
    return dict(recomputed_legal=bool(legal), stored_legal=bool(archived['feasible']), closest_geom_pair=closest,
        closest_category=None if closest is None else classes[closest], min_signed_distance_m=gap,
        collision_slack_m=gap - robot.clearance_m, min_distance_m_by_category=category_ranges,
        active_pairs=sorted(near, key=lambda row: (row['distance_m'], row['geom_ids'])),
        independent_minus_stored_distance_m=gap - archived['collision_clearance_m'],
        independent_minus_stored_coupling_rad=min(coupling.values()) - archived['coupling_margin_rad'],
        independent_minus_stored_range_violation_rad=max(0., -slack) - archived['range_violation_rad'],
        min_engine_contact_distance_m=min(engine.values()) if engine else None,
        engine_contact_pairs_outside_filter=engine_extras, active_axis_limits=active_axes,
        minimum_joint_limit_slack_rad=slack, J6_J7_slack_rad=coupling,
        active_J6_J7_sides=[side for side, value in coupling.items() if value <= ACTIVITY_EPSILON_RAD])


def quantities(values):
    return dict(zip(('min', 'P50', 'P95', 'max'), map(float, np.percentile(values, [0, 50, 95, 100])), strict=True)
                ) if values else None


def aggregate(rows, classes):
    evidence = [row['audit'] for row in rows if 'audit' in row]
    by_pair = defaultdict(list)
    for audit in evidence:
        for pair in audit['active_pairs']:
            by_pair[tuple(pair['geom_ids'])].append(pair['distance_m'])
    return dict(
        planned=len(rows), measured=len(evidence), missing=len(rows) - len(evidence),
        missing_by_status=dict(Counter(row['solver_status'] for row in rows if 'audit' not in row)),
        closest_pair_by_category=dict(Counter(a['closest_category'] for a in evidence if a['closest_category'])),
        witnesses_with_activity=dict(collision=sum(bool(a['active_pairs']) for a in evidence),
            range=sum(bool(a['active_axis_limits']) for a in evidence), coupling=sum(bool(a['active_J6_J7_sides']) for a in evidence)),
        collision_activities_by_category=dict(Counter(pair['category'] for a in evidence for pair in a['active_pairs'])),
        active_pairs=[dict(geom_ids=key, category=classes[key], distance_m=quantities(by_pair[key]), witnesses=len(by_pair[key]))
                      for key in sorted(by_pair, key=lambda k: (-len(by_pair[k]), k))],
        active_range_by_axis_and_boundary=dict(Counter(
            f"{axis['joint']}/{axis['boundary']}" for a in evidence for axis in a['active_axis_limits'])),
        active_coupling_by_side=dict(Counter(side for a in evidence for side in a['active_J6_J7_sides'])),
        min_signed_distance_m=quantities([a['min_signed_distance_m'] for a in evidence]),
        collision_slack_m=quantities([a['collision_slack_m'] for a in evidence]),
        min_joint_limit_slack_rad=quantities([a['minimum_joint_limit_slack_rad'] for a in evidence]),
        min_coupling_slack_rad=quantities([min(a['J6_J7_slack_rad'].values()) for a in evidence]))


def witnesses(original, matched):
    original_rows, matched_rows = load(original), load(matched)['rows']
    keys = [row['identity']['key'] for row in original_rows]
    if len(set(keys)) != len(keys) or keys != [row['identity']['key'] for row in matched_rows]:
        raise ValueError('Matched witnesses do not retain the ordered original case identities')
    configs = sorted({name for row in matched_rows for name in row.get('candidates', {})})
    if 'M0' not in configs:
        raise ValueError('Missing M0 witness group')
    yield 'original/90', [(r['identity'], r.get('whole54'), r.get('disposition')) for r in original_rows]
    for mapping in configs:
        for mode in ('multitask', 'minimax'):
            yield f'{mapping}/{mode}', [
                (r['identity'], r.get('candidates', {}).get(mapping, {}).get(mode), r.get('original_status'))
                for r in matched_rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('matched', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plan = load(PLAN)
    robot = RobotModel(plan['model'], plan['clearance_m'])
    if digest(robot.path) != plan['model_sha256'] or robot.clearance_m != .0005:
        raise ValueError('Signed MJCF identity or clearance changed')
    geoms = geometry_inventory(robot)
    classes, inventory = collision_inventory(robot, geoms)
    discrepancies = []
    if inventory['absent_from_oracle'] or inventory['absent_from_compiled_filter'] or inventory['global_contact_disabled']:
        discrepancies.append('compiled_collision_inventory')
    states = {}
    for label, group in witnesses(args.original, args.matched):
        states[label] = []
        for ident, solution, fallback in group:
            status = solution['status'] if solution else fallback
            record = dict(case_key=ident['key'], split=ident['split'], cohort=ident['cohort'], solver_status=status)
            states[label].append(record)
            if not solution or solution.get('q') is None:
                continue
            audit = audit_state(robot, solution['q'], classes, solution['legality'])
            record['audit'] = audit
            if (not audit['recomputed_legal'] or not audit['stored_legal'] or audit['engine_contact_pairs_outside_filter']
                    or abs(audit['independent_minus_stored_distance_m']) > DISTANCE_AGREEMENT_M
                    or abs(audit['independent_minus_stored_coupling_rad']) > LEGAL_JOINT_TOLERANCE_RAD
                    or abs(audit['independent_minus_stored_range_violation_rad']) > LEGAL_JOINT_TOLERANCE_RAD):
                discrepancies.append(f"{label}/{record['case_key']}")
    summary = {label: aggregate(records, classes) for label, records in states.items()}
    result = dict(schema='s05.p02.constraint_activity.v1', technical_status='FAIL' if discrepancies else 'PASS',
        execution_git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        model_sha256=digest(robot.path), mujoco_version=mujoco.__version__, original_sha256=digest(args.original),
        matched_sha256=digest(args.matched), model_clearance_m=robot.clearance_m, distance_cap_m=DISTANCE_CAP_M,
        activity_above_margin_m=ACTIVITY_EPSILON_M, joint_activity_rad=ACTIVITY_EPSILON_RAD,
        distance_agreement_tolerance_m=DISTANCE_AGREEMENT_M, collision_model=inventory,
        method='Independent compiled static masks and all-pairs MuJoCo narrow phase over archived static q',
        collision_reference='https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection',
        geometry=[geoms[i] for i in sorted(geoms)], per_witness=states, summary=summary, discrepancies=discrepancies)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(result['technical_status'], 'R-P02-06', {label: data['measured'] for label, data in summary.items()}, flush=True)
    if discrepancies:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
