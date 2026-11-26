"""Read-only raw Tracker jump and bounded wrist mount studies for DCR-003."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel,digest
from dexgym.tasks.direct.wholebody_s05.p02.oracle import metrics
from evaluate import (analytic_bound,build_reader,check_witness, initial_seeds,read_frame,
                      recorded_target,solve_matched,transform_target)
from run_study import E,load,samples,write
from workspace import DEF,ORIGINAL,SIDE,apply,from_frame,norm,prep,reach_envelope,semantic_order,source,stats


def raw_jumps(out,base,prod,bank,geo,robot,cal,fits):
    signed=load(E/'full_coverage.json')
    readers=build_reader(bank)
    thresholds=(.025,.050,.100)
    b,r=reach_envelope(robot)
    result={}
    for split,reader in readers.items():
        counts=Counter()
        extremes=[]
        for p in prod['partitions']:
            if p['split']!=split:
                continue
            a,meta,_,frames=reader.read(p['key'])
            counts['partitions']+=1
            counts['planned']+=len(a['joint_valid'])
            counts['joint_valid']+=int(a['joint_valid'].sum())
            if a['pose'].shape[1]!=6:
                raise ValueError('Expected MANUS, registered and raw Tracker in signed six-stream product')
            for s,side in enumerate(SIDE):
                raw=a['pose'][:,4+s,0,:3]
                valid=a['pose_valid'][:,4+s]
                counts[side,'raw_valid']+=int(valid.sum())
                src=a['source_index'][:,4+s,0]
                delta=norm(np.diff(raw,axis=0))
                good=valid[1:]&valid[:-1]&(np.diff(src)!=0)&(np.diff(a['tick_ns'])==20_000_000)
                counts[side,'raw_adjacent_distinct']+=int(good.sum())
                counts[side,'joint_comparable']+=int((good&a['joint_valid'][1:]).sum())
                # Classify source updates on the HMB tick grid using three predeclared step sizes.
                tick=np.flatnonzero(good&(delta>thresholds[0]))+1
                for cut in thresholds:
                    mask=tick[delta[tick-1]>cut]
                    counts[side,int(cut*1000),'all']+=len(mask)
                    counts[side,int(cut*1000),'joint_valid']+=int(a['joint_valid'][mask].sum())
                for t in tick:
                    mag=float(delta[t-1])
                    extremes.append(dict(root=p['parent'],partition=p['key'],tick=int(t),tick_ns=int(a['tick_ns'][t]),
                                   side=side,raw_step_m=mag,raw_source_indices=src[t-1:t+1].tolist(),
                                   joint_valid=bool(a['joint_valid'][t])))
                accepted=tick[a['joint_valid'][tick]]
                for cfg_name,cfg in fits.items():
                    bucket=[]
                    for t in accepted:
                        target,_=from_frame(a,meta,t,robot,cal,cfg,base['cross_hand_pairs'])
                        bucket.append(float(np.maximum(0,norm(target.world[:,:,-1]-b)-r-1e-9).max()))
                    for cut in thresholds:
                        prox=np.asarray(bucket)[delta[accepted-1]>cut]
                        counts[side,int(cut*1000),cfg_name,'strict3_joint']+=int((prox>.003).sum())
        declared=signed[split]['counts']
        measured=load(geo/split.lower()/'workspace_summary.json')
        if counts['joint_valid']!=declared['causal.joint_valid'] or counts['planned']!=declared['planned_ticks'] or (
               measured['signed_joint_valid']!=counts['joint_valid'] or measured['counts']['planned']!=counts['planned']):
            raise ValueError('Independent signed E1 counts did not reproduce')
        result[split]=dict(counts={str(k):v for k,v in sorted(counts.items(),key=lambda kv:str(kv[0]))},
                           top_raw_steps=sorted(extremes,key=lambda o:(-o['raw_step_m'],o['partition'],o['tick']))[:12],
                           signed_e1_agreement=True)
    write(out/'raw_tracker_step_overlap.json',dict(
        clock='Historical simulated causal tick, successive valid distinct raw sample indices within a partition',
        scope='A step threshold is a diagnostic observation, not a sensor-artifact classification.',
        thresholds_m=thresholds,splits=result,signed_coverage_sha256=digest(E/'full_coverage.json')))


def fit_mount(out,base,bank,robot,cal,m1):
    train=build_reader(bank)['Train']
    signed=load(DEF/'study_records.json') if (DEF/'study_records.json').is_file() else load(ORIGINAL/'intent-02/study_records.json')
    selected=[r for r in signed if r['identity']['split']=='Train' and r['identity']['cohort']=='ROOT' and 'mapped' in r]
    tracker=[]
    for case in selected:
        id_=case['identity']
        a,_,t=read_frame({'Train':train},id_)
        tracker.append(a['pose'][t,2:4,0,3:7])
    data=np.array(tracker)
    rr=Rotation.from_quat(data.reshape(-1,4)).as_matrix().reshape(-1,2,3,3)
    rs=np.asarray(m1['rotation'])
    home=robot.geometry(robot.home).rotations
    guess=((rs@rr).transpose(0,1,3,2)@home)
    v=np.stack([Rotation.from_matrix(guess[:,s]).mean().as_matrix() for s in range(2)])
    old=np.array(m1['mount'])
    diff=Rotation.from_matrix(old.transpose(0,2,1)@v).as_rotvec()
    clipped=diff*np.minimum(1,np.deg2rad(10)/np.maximum(norm(diff),1e-20))[:,None]
    cfg=m1|{'mount':(old@Rotation.from_rotvec(clipped).as_matrix()).tolist()}
    (out/'mount_plan.txt').write_text('Use all archived Train ROOT rows. Chordal wrist-to-home mount fit capped at 10 degrees per side. Frozen pair: first three eligible ROOT rows per split; same safety and 300 iterations; fit never uses Validation.\n')
    write(out/'mount_diagnostic_train_freeze.json',dict(config=cfg,train_count=len(data),train_ids=[x['identity']['key'] for x in selected],
          additional_rotation_rad=norm(clipped).tolist(),max_rotation_rad=float(np.deg2rad(10))))
    return cfg


def probe_mount(out,base,bank,robot,reference,cal,m1,alternative):
    records=load(ORIGINAL/'intent-02/study_records.json')
    readers=build_reader(bank)
    probe=[]
    for split in ('Train','Validation'):
        selected=[x for x in records if x['identity']['split']==split and x['identity']['cohort']=='ROOT' and 'mapped' in x][:3]
        for case in selected:
            identity=case['identity']
            a,_,t=read_frame(readers,identity)
            old=recorded_target(case['mapped'])
            paired={}
            for name,cfg in [('M1_immutable',m1),('M1_bounded_mount',alternative)]:
                target=transform_target(old,{'wrist':a['pose'][t,2:4,0]},cfg,cal,base['cross_hand_pairs'])
                seed,cost=initial_seeds(robot,base,case,target,False)
                result=solve_matched(robot,base,target,seed,300)
                bound=analytic_bound(robot,target,reach_envelope(robot))
                check_witness(reference,target,result,bound)
                paired[name]=dict(fk=result,lower_m=bound,seed_s=cost)
            probe.append(dict(identity=identity,modes=paired))
    write(out/'mount_ablation.json',probe)


def main(out,geo):
    base,plan,prod,bank=source()
    cal=load(DEF/'mapping_contract.json')
    robot=RobotModel(base['model'],base['clearance_m'])
    reference=RobotModel(base['model'],base['clearance_m'])
    frozen=geo/'fitted_candidates.json'
    if digest(frozen)!=(geo/'fitted_candidates.sha256').read_text().strip():
        raise ValueError('Fitted candidate identity failure')
    fits=load(frozen)['configs']
    out.mkdir(mode=0o700)
    alternate=fit_mount(out,base,bank,robot,cal,fits['M1'])
    probe_mount(out,base,bank,robot,reference,cal,fits['M1'],alternate)
    raw_jumps(out,base,prod,bank,geo,robot,cal,fits)
    print('Bounded wrist mount and independent raw Tracker diagnostic complete',flush=True)


if __name__=='__main__':
    main(*(Path(x) for x in sys.argv[1:]))
