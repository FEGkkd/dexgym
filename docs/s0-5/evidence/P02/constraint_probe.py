"""Matched static counterfactual for the diagnostic contribution of the collision margin."""
from pathlib import Path
import sys

import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.intent import Target
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, digest
from dexgym.tasks.direct.wholebody_s05.p02.oracle import solve
from run_study import load, write


def main(plan, probe, run, output):
    config = load(plan)
    scored = {r['identity']['key']:r for r in load(run/'intent-02/study_records.json') if 'feasible' in r}
    integrity = load(run/'sealed-01/score_integrity.json')
    results = []
    for positive in (False, True):
        eligible = [r for r in integrity['rechecked'] if r['collision_near_margin']
                    and (r['lower_bound_max_m'] > 0) == positive]
        witness = max(eligible, key=lambda r:(scored[r['case_key']]['whole54']['metrics']['max_tip_m'],
                                              r['case_key']))
        case = scored[witness['case_key']]
        target = Target(**{key:np.asarray(value) for key,value in case['mapped'].items()})
        seeds = [(x['label'],np.asarray(x['input_initial'])) for x in case['whole54']['attempts']]
        ablation = {}
        for name, margin in (('nominal',config['clearance_m']),('zero_margin',0.)):
            robot = RobotModel(config['model'],margin)
            result = solve(robot,target,config,seeds)
            ablation[name] = result
        results.append(dict(case_key=witness['case_key'],strict_radial_lower_bound_m=witness['lower_bound_max_m'],
                            paired=ablation))
    write(output,dict(plan_sha256=digest(probe),scope=load(probe)['scope'],results=results))
    for row in results:
        print(row['case_key'],[(k,v['status'],v.get('metrics',{}).get('max_tip_m'))
                              for k,v in row['paired'].items()])


if __name__=='__main__':
    main(*(Path(arg) for arg in sys.argv[1:]))
