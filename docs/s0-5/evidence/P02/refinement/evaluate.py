"""Matched static Oracle budgets, epigraph diagnosis and signed functional witnesses."""
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import mujoco
import numpy as np

from dexgym.tasks.direct.wholebody_s05.hmb_reader import GEOMETRY,HMBReader
from dexgym.tasks.direct.wholebody_s05.p02.intent import Target,target_geometry,robot_target
from dexgym.tasks.direct.wholebody_s05.p02.minimax import solve_minimax
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel,digest,independent_frames
from dexgym.tasks.direct.wholebody_s05.p02.oracle import Objective,joint_seed,solve,wrist_seed
from run_study import E,E0,load,write
from seal_p02 import mapping_and_solver,reach_envelope
from workspace import DEF,ORIGINAL,apply,from_frame,norm,source,stats


def corrected_pinch(out,records):
    before=load(ORIGINAL/'sealed-02/mapping_and_solver_summary.json')
    now=mapping_and_solver(records,[])
    fields=('near_pinch_pairs','active_pinch_before_m','active_pinch_after_m')
    delta={k:{s:{'original': before[k][s],'corrected':now[k][s]} for s in fields}
           for k in ('All','ROOT','TAIL','Train','Validation')}
    write(out/'pinch_statistics_correction.json',dict(record_sha256=digest(ORIGINAL/'intent-02/study_records.json'),
         original_summary_sha256=digest(ORIGINAL/'sealed-02/mapping_and_solver_summary.json'),changes=delta,
         units='count and m; metrics P50/P95/P99/max; only active indices',geometry_or_solver_changed=False))


def recorded_target(data):
    return Target(**{k:np.asarray(v,dtype=float) for k,v in data.items()})


def transform_target(old,row,cfg,cal,pairs):
    if cfg['scale']==1. and cfg['yaw_offset_rad']==0. and cfg['translation']==cal['world_translation_m']:
        return old
    w,r,_=apply(cfg,row['wrist'][None],old.local[None],cal)
    return target_geometry(w[0],r[0],old.local,
                 (np.asarray(cal['wrist_to_palm_translation_m']),np.asarray(cal['wrist_to_palm_rotation'])),pairs)


def initial_seeds(robot,config,row,target,recorded):
    start=perf_counter()
    old=row['whole54']['attempts']
    if len(old)!=3:
        raise ValueError('Expected the original three initializations')
    if recorded:
        result=[(x['label'],np.asarray(x['input_initial'])) for x in old]
    else:
        anchor=np.r_[robot.home[:14],row['sdk_wrist_ik']['sdk_unclipped_q']]
        q=wrist_seed(robot,target,anchor)
        result=[(old[0]['label'],q),(old[1]['label'],joint_seed(Objective(robot,target,config),q)),
                (old[2]['label'],np.asarray(old[2]['input_initial']))]
    return result,perf_counter()-start


def solve_matched(robot,config,target,seeds,it):
    return solve(robot,target,config|{'solver':config['solver']|{'max_iterations_per_start':it}},seeds)


def check_witness(robot,goal,result,bound):
    if result['q'] is None:
        return
    q=np.array(result['q'])
    if not robot.legality(q)['feasible']:
        raise ValueError('Returned witness on second model violates hard constraints')
    tree=independent_frames(robot,q)
    tips=np.asarray([tree['site',robot.model.site(idx).name][0] for idx in robot.tip_ids]).reshape(2,5,3)
    residual=norm(tips-goal.world[:,:,-1]).max()
    if abs(residual-result['metrics']['max_tip_m'])>2e-9 or residual+2e-9<bound:
        raise ValueError(f'Independent tip geometry or analytic bound violation: error={residual} bound={bound}')


def analytic_bound(robot,goal,radius):
    origin,r=radius
    return float(np.maximum(0,norm(goal.world[:,:,-1]-origin)-r-1e-9).max())


def do_budget(out,base,plan,rows,robot,reference):
    corrected_pinch(out,rows)
    results=[]
    radius=reach_envelope(robot)
    for i,row in enumerate(rows):
        ident=row['identity']
        frame=dict(identity=ident,status=row.get('disposition','MAPPED'))
        results.append(frame)
        if 'mapped' not in row:
            continue
        old=recorded_target(row['mapped'])
        bound=analytic_bound(robot,old,radius)
        seeds,seconds=initial_seeds(robot,base,row,old,True)
        tests=dict()
        for budget in plan['solve']['multitask_iteration_budgets']:
            result=solve_matched(robot,base,old,seeds,budget)
            check_witness(reference,old,result,bound)
            tests[str(budget)]=result
        frame.update(radial_lower_bound_m=bound,seed_loading_s=seconds,
                     original_90_tip_m=row['whole54']['metrics']['max_tip_m'],
                     budgets=tests)
        print('budget',i+1,ident['key'],[round(a.get('metrics',{}).get('max_tip_m',-1)*1000,2) for a in tests.values()],flush=True)
    write(out/'budget_results.json',results)


def build_reader(bank):
    return {s:HMBReader(E/'hmb_manifest.json',E/'production_plan.json',E0/'candidate_config.json',
                        bank['ubuntu_products'],s,GEOMETRY) for s in ('Train','Validation')}


def read_frame(readers,id_):
    a,meta,_,_=readers[id_['split']].read(id_['partition'])
    t=id_['tick']
    if not a['joint_valid'][t]:
        raise ValueError('Joint-mask-invalid score target')
    return a,meta,t


def task(out,geo,bud,base,plan,rows,robot,reference,bank):
    configs_path=geo/'fitted_candidates.json'
    if (geo/'fitted_candidates.sha256').read_text().strip()!=digest(configs_path):
        raise ValueError('Candidates differ from the frozen Train bytes')
    fixed=load(configs_path)
    if fixed['fit_scope']!='Train':
        raise ValueError('Candidates not fitted on approved Train roots')
    cal=load(DEF/'mapping_contract.json')
    readers=build_reader(bank)
    radius=reach_envelope(robot)
    budget_rows=load(bud/'budget_results.json')
    run=[]
    for i,(record,br) in enumerate(zip(rows,budget_rows,strict=True)):
        ident=record['identity']
        row=dict(identity=ident,original_status=record.get('disposition','MAPPED'))
        run.append(row)
        if 'mapped' not in record:
            continue
        a,meta,t=read_frame(readers,ident)
        old=recorded_target(record['mapped'])
        replay,_=from_frame(a,meta,t,robot,cal,fixed['configs']['M0'],base['cross_hand_pairs'])
        if max(np.max(np.abs(getattr(replay,f)-getattr(old,f))) for f in vars(old))>2e-9:
            raise ValueError('Vectorized M0 differs from original frozen study target')
        variants={}
        for name,cfg in fixed['configs'].items():
            target=transform_target(old,{'wrist':a['pose'][t,2:4,0]},cfg,cal,base['cross_hand_pairs'])
            lower=analytic_bound(robot,target,radius)
            seed,seconds=initial_seeds(robot,base,record,target,name=='M0')
            mt=br['budgets'][str(plan['solve']['matched_candidates_iterations'])] if name=='M0' else solve_matched(
                robot,base,target,seed,plan['solve']['matched_candidates_iterations'])
            locked=record['arms14'].get('q')
            candidates=[('MATCHED_MULTITASK',np.asarray(mt['q']) if mt['q'] is not None else seed[1][1]),
                        ('LEGAL_FIXED_HAND_OR_WRIST',np.asarray(locked) if locked is not None else seed[0][1]),
                        ('HOME',robot.home.copy())]
            mm=solve_minimax(robot,target,base,candidates,plan['solve']['minimax_iterations'])
            check_witness(reference,target,mt,lower)
            check_witness(reference,target,mm,lower)
            if mt['q'] is not None and mm['q'] is not None and mm['epigraph_m'] > mt['metrics']['max_tip_m']+1e-10:
                raise ValueError('Minimax discarded its legal multitask candidate')
            if name=='M0' and locked is not None and mm['epigraph_m']>record['arms14']['metrics']['max_tip_m']+1e-10:
                raise ValueError('Minimax discarded its legal fixed-hand witness')
            variants[name]=dict(mapped=target.json(),strict_radial_bound_m=lower,
                necessary_domain='EXCLUDED_AT_3MM' if lower>.003 else 'NOT_EXCLUDED_AT_3MM',
                max_tip_change_from_M0_m=float(norm(target.world[:,:,-1]-old.world[:,:,-1]).max()),
                wrist_span_raw_m=float(norm(a['pose'][t,3,0,:3]-a['pose'][t,2,0,:3])),
                wrist_span_mapped_m=float(norm(target.wrist[1]-target.wrist[0])),
                cross_delta_from_M0_m=norm(target.cross-old.cross).tolist(),
                seed_generation_s=seconds, multitask=mt, minimax=mm)
        row['candidates']=variants
        print('matched',i+1,ident['key'],{k:(round(v['strict_radial_bound_m']*1000,2),
              round(v['minimax'].get('epigraph_m',-1)*1000,2)) for k,v in variants.items()},flush=True)
    write(out/'matched_candidates.json',dict(candidates_sha256=digest(configs_path),rows=run))


def controls(out,base,plan,robot,reference):
    rng=np.random.default_rng(base['seed'])
    models=load(DEF/'model_report.json')['model']['results']
    safe=[np.asarray(row['q']) for row in models if row['legality']['feasible']
          and row['scenario'] in ('random_legal','coupling_boundary')]
    bound=next(np.asarray(row['q']) for row in models if row['scenario']=='coupling_boundary' and row['legality']['feasible'])
    q=safe[len(safe)//2]
    radius=reach_envelope(robot)
    cases=[]
    for name,goal in [
        ('known_home',robot_target(robot,robot.geometry(robot.home),base['cross_hand_pairs'])),
        ('known_nonhome',robot_target(robot,robot.geometry(q),base['cross_hand_pairs'])),
        ('known_coupling_boundary',robot_target(robot,robot.geometry(bound),base['cross_hand_pairs']))]:
        q_target={'known_home':robot.home,'known_nonhome':q,'known_coupling_boundary':bound}[name]
        seed=[('PERTURBED_TARGET',q_target+rng.normal(0,.025,54)),
              ('PERTURBED_HOME',robot.home+rng.normal(0,.018,54)),('HOME',robot.home)]
        result=solve_minimax(robot,goal,base,seed,300)
        nominal=solve_minimax(robot,goal,base,[('KNOWN_LEGAL_WITNESS',q_target)],90)
        check_witness(reference,goal,result,analytic_bound(robot,goal,radius))
        check_witness(reference,goal,nominal,0.)
        if nominal['epigraph_m']>1e-10:
            raise ValueError('Known legal exact target lost its same-q witness')
        cases.append(dict(name=name,result=result,known_witness_tip_m=nominal['epigraph_m']))
    goal=robot_target(robot,robot.geometry(robot.home),base['cross_hand_pairs'])
    goal.world[:]+=np.array([0.,0.,2.])
    seed=[('HOME',robot.home),('BOUNDARY',bound),('LEGAL_OTHER',q)]
    low=analytic_bound(robot,goal,radius)
    result=solve_minimax(robot,goal,base,seed,300)
    other=solve_minimax(robot,goal,base,seed,300)
    check_witness(reference,goal,result,low)
    status=abs(result['epigraph_m']-other['epigraph_m'])<1e-10
    if not status or low<=1.:
        raise ValueError('Determinism or synthetic analytic bound control failed')
    cases.append(dict(name='known_unreachable_translation',radial_lower_bound_m=low,result=result,
                      deterministic_goal_value_within_1e_10_m=status,repeated_q_max_delta_rad=float(
                          np.abs(np.asarray(result['q'])-other['q']).max())))
    write(out/'minimax_controls.json',cases)


def functional(out,geo,base,plan,robot,reference,bank):
    cal=load(DEF/'mapping_contract.json')
    cfg=load(geo/'fitted_candidates.json')['configs']['M0']
    readers=build_reader(bank)
    signed=load(DEF/'near_pinch_witness.json')
    witnesses=[]
    for split in ('Train','Validation'):
        for side,entry in signed['strata'][split].items():
            first=entry['first']
            if first is None:
                continue
            identity={k:first[k] for k in ('partition','tick','root')}|{'split':split,'side':side}
            a,meta,t=read_frame(readers,identity)
            target,geom=from_frame(a,meta,t,robot,cal,cfg,base['cross_hand_pairs'])
            s=0 if side=='left' else 1
            raw=geom['raw_pinch'][0,s]
            active=np.flatnonzero(norm(raw)<=.020)
            errors=np.abs(target.pinch[s]-np.asarray(first['mapping']['after_vectors']))
            if list(active)!=first['mapping']['near_pinch_pairs'] or float(errors.max())>2e-9:
                raise ValueError('Independent near-pinch reconstruction differs from signed data')
            witnesses.append(dict(identity=identity,active=active.tolist(),raw_m=raw.tolist(),
                    before_m=(geom['unadjusted'][0,s,1:,-1]-geom['unadjusted'][0,s,:1,-1]).tolist(),
                    hard_m=target.pinch[s].tolist(),reconstruction_max_component_m=float(errors.max())))
    probes=[]
    for split in ('Train','Validation'):
        func=load(geo/split.lower()/'functional_summary.json')
        for record in func['fk_witnesses']:
            id_=record|{'split':split}
            a,meta,t=read_frame(readers,id_)
            for index in (t,t-1):
                row=id_|{'tick':index}
                for method in ('hard','smooth'):
                    goal,geom=from_frame(a,meta,index,robot,cal,cfg,base['cross_hand_pairs'],smooth=method=='smooth')
                    q0=wrist_seed(robot,goal,robot.home)
                    q1=joint_seed(Objective(robot,goal,base),q0)
                    result=solve_matched(robot,base,goal,[('HOME',robot.home),('WRIST_IK',q0),('JOINT_IK',q1)],300)
                    check_witness(reference,goal,result,0.)
                    probes.append(dict(identity=row,method=method,raw_pinch_m=geom['raw_pinch'][0].tolist(),
                         mapped_pinch_m=goal.pinch.tolist(),solver=result))
    write(out/'functional_fk_checks.json',dict(historical_reconstruction=witnesses,paired_continuity_fk=probes))


def main(mode,out,geo=None,bud=None):
    base,plan,_,bank=source()
    robot=RobotModel(base['model'],base['clearance_m'])
    witness=RobotModel(base['model'],base['clearance_m'])
    rows=load(ORIGINAL/'intent-02/study_records.json')
    if len(rows)!=38 or sum(x['identity']['cohort']=='ROOT' for x in rows)!=32:
        raise ValueError('Signed cohort plan differs')
    out.mkdir(mode=0o700)
    if mode=='budget':
        do_budget(out,base,plan,rows,robot,witness)
    if mode=='matched':
        task(out,geo,bud,base,plan,rows,robot,witness,bank)
    if mode=='functional':
        functional(out,geo,base,plan,robot,witness,bank)
        controls(out,base,plan,robot,witness)


if __name__=='__main__':
    main(sys.argv[1],*(Path(x) for x in sys.argv[2:]))
