"""Bounded near-pinch witnesses from the same 32 approved root partitions."""
from pathlib import Path
import sys

import numpy as np

from dexgym.tasks.direct.wholebody_s05.hmb_reader import GEOMETRY, HMBReader
from dexgym.tasks.direct.wholebody_s05.p02.intent import Mapper, semantic_order
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, digest
from run_study import E, E0, load, samples, verify_approval, write


def main(plan, supplement, out):
    config = load(plan)
    prereg = load(supplement)
    handoff = verify_approval(config)
    selected = samples(config, handoff)
    robot = RobotModel(config['model'], config['clearance_m'])
    mapper = Mapper(robot, [row for ident,row in selected if row is not None
                            and ident['split']=='Train' and ident['cohort']=='ROOT'], config)
    splits = ('Train','Validation')
    readers = {split:HMBReader(E/'hmb_manifest.json', E/'production_plan.json', E0/'candidate_config.json',
                              handoff['ubuntu_products'], split, GEOMETRY) for split in splits}
    counts = {split:{side:dict(pose_ticks=0,close_pose_ticks=0,min_distance_m=None,first=None)
                     for side in ('left','right')} for split in splits}
    partitions = total_ticks = 0
    for ident, _ in selected:
        if ident['cohort']!='ROOT':
            continue
        split = ident['split']
        a, meta, source_table, _ = readers[split].read(ident['partition'])
        partitions += 1
        total_ticks += meta['ticks']
        valid_ticks, = np.nonzero(a['joint_valid'] & a['pose_valid'][:,:4].all(axis=1))
        def topology(t,s,side):
            ver = meta['dictionary']['topology_id'][int(a['topology_id'][t,s])]
            return next(x for x in meta['topologies'] if (x['stream'],x['version'])==(f'manus_{side}',ver))
        for t in valid_ticks:
            topologies = [topology(t,s,side) for s,side in enumerate(('left','right'))]
            for s,side in enumerate(('left','right')):
                report = counts[split][side]
                _, order = semantic_order(topologies[s])
                xyz = a['pose'][t,s,order[:,-1],:3]
                minimum = float(np.linalg.norm(xyz[1:]-xyz[:1],axis=1).min())
                report['pose_ticks'] += 1
                if report['min_distance_m'] is None or minimum < report['min_distance_m']:
                    report['min_distance_m'] = minimum
                if minimum > config['pinch_raw_threshold_m']:
                    continue
                report['close_pose_ticks'] += 1
                if report['first'] is not None:
                    continue
                row = {'palm':a['pose'][t,:2], 'sdk_common':a['sdk_common_pose'][t,:2],
                       'topology':topologies, 'wrist':a['pose'][t,2:4,0]}
                _,_, result = mapper.map(row)
                witness = result['hands'][s]
                if not witness['tol_active'] or max(witness['tol_active']) > 1e-10:
                    raise ValueError('Raw near-pinch witness failed the frozen mapping audit')
                report['first'] = dict(root=ident['root'],partition=ident['partition'],tick=int(t),
                                       source=source_table[int(a['source_index'][t,s,0])],
                                       topology=int(a['topology_id'][t,s]),distance_m=minimum, mapping=witness)
    if partitions!=prereg['budget']['max_partitions'] or total_ticks>prereg['budget']['max_ticks']:
        raise ValueError('Supplemental partition budget is not conserved')
    write(out,dict(plan_sha256=digest(supplement), exam_partitions=partitions,total_ticks=total_ticks,
                   scope=prereg['scope'],strata=counts))
    print({split:{s:r['close_pose_ticks'] for s,r in part.items()} for split,part in counts.items()})


if __name__=='__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
