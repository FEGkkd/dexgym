"""Exact archived S0-4 target diagnostics on its original MJCF, without rollout."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.intent import cross_hand, pinch, target_geometry
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, digest
from dexgym.tasks.direct.wholebody_s05.p02.oracle import metrics, solve, wrist_seed


ROOT=Path('/home/ubuntu/dexgym/data/processed/wholebody-s04')
RUN=ROOT/'world_hand20_aligned_from_scratch_20261125'
OLD=ROOT/'world_hand20_gravity_20000_20261125/gravity_cases.json'


def load(p):
    return json.loads(Path(p).read_bytes())


def numpy_scalar(value):
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'Unexpected value in historical evidence: {type(value).__name__}')


def verify(path,expected):
    if digest(path)!=expected:
        raise ValueError(f'Historical identity mismatch: {path}')


def reference(data,id_,robot,manifest,plan):
    group=manifest['frame']
    if (group['wrist'],group['fingers'],group['landmarks'])!=(
        'fixed_robot_world_simulated_wuji_wrist','same_side_simulated_wuji_wrist',['joint_2','joint_3','tip']):
        raise ValueError('The bank does not use archived physical-wrist geometry')
    positions=np.asarray(data['wrist'][id_],dtype=float)
    rotations=np.asarray(data['rotation'][id_],dtype=float)
    monitor=manifest['wrist_calibration']
    if monitor['direction']!='marvin_native_tcp_from_wuji_wrist':
        raise ValueError('Historical wrist calibration direction differs')
    mount_t=np.asarray(monitor['translation_m'])
    mount_r=Rotation.from_quat(monitor['quaternion_xyzw']).as_matrix()
    home=robot.geometry(robot.home)
    native_t=np.einsum('sji,sj->si',home.rotations,home.palm-home.wrists)
    native_r=home.rotations.transpose(0,2,1)@home.palm_rotation
    if not (np.allclose(mount_t,native_t,atol=1e-10) and np.allclose(mount_r,native_r,atol=1e-10)):
        raise ValueError('Historical bank calibration does not agree with its MJCF')
    tcp_r=rotations@mount_r.transpose(0,2,1)
    tcp_p=positions-np.einsum('sij,sj->si',tcp_r,mount_t)
    angles=np.asarray(data['rsi_q'][id_],dtype=float)
    names=manifest['joint_names']
    if set(names)!=set(robot.names):
        raise ValueError('Historical bank joint mapping differs')
    q=angles[[names.index(n) for n in robot.names]]
    stored=np.asarray(data['hand'][id_],dtype=float)
    local=np.concatenate((home.local[:,:,:1],stored),axis=2)
    goal=target_geometry(tcp_p,tcp_r,local,(mount_t,mount_r),plan['cross_hand_pairs'])
    direct=stored@rotations.transpose(0,2,1)[:,None]+positions[:,None,None]
    if float(np.max(np.abs(direct-goal.world[:,:,1:])))>2e-7:
        raise ValueError('Archived S0-4 fingertip world target changed in conversion')
    g=robot.geometry(q)
    sdk_geometry=float(np.linalg.norm(g.local[:,:,1:]-stored,axis=-1).max())
    return q,goal,sdk_geometry


def historical_clip(sources,frame):
    for item in sources:
        lo=item.get('start',item.get('start_frame',item.get('row_start')))
        hi=item.get('stop',item.get('end_frame',item.get('row_stop',item.get('end'))))
        if isinstance(lo,int) and isinstance(hi,int) and lo<=frame<hi:
            return {k:item[k] for k in ('id','start','stop','source','source_type','episode','recording','source_episode') if k in item}
    return {'status':'UNKNOWN_CLIP_METADATA'}


def grouping(rows,plan):
    m={}
    for key in ('A_archived','B_fixed_hand_14','C_bounded_arm_54','D_full_54'):
        e=[r[key]['metrics']['max_tip_m'] for r in rows if 'metrics' in r[key]]
        m[key]=dict(scored=len(e),planned=len(rows),median_m=float(np.median(e)) if e else None,
             maximum_m=max(e) if e else None,
             hit_counts={str(h):sum(v<=h for v in e) for h in plan['tolerance']['hit_thresholds_m']})
    return m


def run(plan,out,model):
    if not load(model)['historical_model']['valid']:
        raise ValueError('Historical model requires independent P02-A audit')
    prov=load(RUN/'train/provenance.json')
    if prov['model_sha256']!=plan['historical_model_sha256'] or prov['action_dim']!=14 or prov['residual_bound_deg']!=8.:
        raise ValueError('Historical action/geometry identity differs')
    verify(plan['historical_model'],plan['historical_model_sha256'])
    cp=RUN/'train'/plan['historical_hold']['checkpoint']
    verify(cp,plan['historical_hold']['checkpoint_sha256'])
    bank=Path(prov['bank'])
    verify(bank,prov['bank_sha256'])
    manifest=load(bank)
    if manifest['teleop_model_sha256']!=plan['historical_model_sha256']:
        raise ValueError('Historical bank model digest differs')
    split=manifest['splits']['validation']
    samples=(bank.parent/split['file']).resolve()
    verify(samples,split['sha256'])
    review=RUN/'train/review_10000.json'
    record=load(review)
    if record['iteration']!=10000 or Path(record['actor_file']).name!=cp.name:
        raise ValueError('Historical hold case report belongs to a different checkpoint')
    old=load(OLD)
    frames=[10148,38378,27552]
    frames.extend(x['frame'] for x in old if x.get('hard_tail') and x['frame'] not in frames)
    frames=frames[:plan['historical_hold']['max_cases']]
    rng=np.random.default_rng(plan['seed'])
    robot=RobotModel(plan['historical_model'],plan['clearance_m'])
    check_robot=RobotModel(plan['historical_model'],plan['clearance_m'])
    entries=[]
    with np.load(samples,allow_pickle=False) as data:
        values={k:data[k] for k in ('rsi_q','hand','wrist','rotation','time_ns')}
        for id_ in frames:
            q,goal,shape=reference(values,id_,robot,manifest,plan)
            arm_seed=wrist_seed(robot,goal,q)
            noise=q+rng.normal(0,.03,54)
            radius=np.deg2rad(plan['solver']['paired_arm_residual_deg'])
            limits=robot.bounds.copy()
            limits[:14,0]=np.maximum(limits[:14,0],q[:14]-radius)
            limits[:14,1]=np.minimum(limits[:14,1],q[:14]+radius)
            starts=[('ARCHIVE_REFERENCE',q),('WRIST_IK',arm_seed),('PERTURBED_REFERENCE',noise)]
            locked=[(label,np.r_[s[:14],q[14:]]) for label,s in starts]
            a=dict(metrics=metrics(goal,robot,q,plan['cross_hand_pairs']),legality=robot.legality(q),
                   status='ORIGINAL_ARCHIVED_SDK_REFERENCE',q=q.tolist())
            b=solve(robot,goal,plan,locked,indices=np.arange(14))
            c=solve(robot,goal,plan,starts,bounds=limits)
            d=solve(robot,goal,plan,starts)
            for candidate in (b,c,d):
                if candidate['q'] is not None and not check_robot.legality(np.asarray(candidate['q']))['feasible']:
                    raise ValueError('Historical best witness violates constraints on second model instance')
            measured=record['hold'].get(f'case_{id_}_mm')
            entries.append(dict(
                frame=id_,time_ns=int(values['time_ns'][id_]),clip=historical_clip(split['clips'],id_),
                ref_shape_error_m=shape,archived_target=goal.json(),historical_hold_median_mm=measured,
                historical_hold_scope='25 terminal dynamics ticks, historical repeated selection',
                A_archived=a,B_fixed_hand_14=b,C_bounded_arm_54=c,D_full_54=d))
            print(id_,[round(x.get('metrics',{}).get('max_tip_m',-1)*1000,4) for x in (a,b,c,d)],flush=True)
    result=dict(source_role='HISTORICAL_VALIDATION_AND_DIAGNOSTICS_ONLY',no_final_test_access=True,
                historical_model=plan['historical_model'],historical_model_sha256=plan['historical_model_sha256'],
                historical_bank=str(bank),bank_sha256=prov['bank_sha256'],validation_npz_sha256=split['sha256'],
                historical_checkpoint=str(cp),checkpoint_sha256=digest(cp),
                selection_table_sha256=digest(OLD),historical_hold_report_sha256=digest(review),
                historical_servo_profile=prov['servo_profile'],historical_servo_sha256=prov['servo_profile_sha256'],
                historical_review_hold=record['hold'],historical_review_hard=record['hard'],
                reference_lineage=dict(arm_refit_validation_published=manifest['arm_reference_refit']['splits']['validation']['published'],
                      original_angles_preserved=manifest['fixed_hand_reachability']['original_angles_preserved'],
                      sdk_reference=manifest['hand_reference_contract']),
                constraints=dict(margin_m=robot.clearance_m,all_modes_same_model=True,same_target_for_all=True,
                                 B_fixed_hands_full_legal_arm=True,C_bound_arm_about_original_deg=8),
                cases=entries,summary=grouping(entries,plan),
                execution_measurement_scope='HISTORICAL_HOLD_SUMMARY_ONLY_NO_P02_DYNAMIC_REPLAY')
    with (out/'historical_hard_case.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False,default=numpy_scalar)


if __name__=='__main__':
    plan,out,model=(Path(x) for x in sys.argv[1:])
    run(load(plan),out,model)
