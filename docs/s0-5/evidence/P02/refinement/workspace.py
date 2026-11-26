"""Train-only workspace registration, and complete signed HMB geometry audit."""
from collections import Counter
from pathlib import Path
import json
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.hmb_reader import GEOMETRY, HMBReader
from dexgym.tasks.direct.wholebody_s05.p02.intent import semantic_order, Target
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel, digest
from run_study import E, E0, F, load, write
from seal_p02 import reach_envelope


DEF = Path('docs/s0-5/evidence/P02')
ORIGINAL = Path('/home/ubuntu/dexgym_data/s0-5/p02/runs/P02-20261126-54c2e06')
SPLITS = ('Train','Validation')
SIDE = ('left','right')
THRESHOLDS = (.003,.005,.010,.015)
JUMP = (.025,.050,.100)


def source():
    base, plan = (load(DEF/f) for f in ('plan.json','refinement_plan.json'))
    reqs = [(DEF/'review_receipt.json',plan['reference_review_receipt_sha256']),
            (F/'p02_handoff_manifest.json',base['handoff_sha256']),
            (E/'hmb_manifest.json',base['e1_manifest_sha256']),
            (Path(base['model']),base['model_sha256'])]
    bank = load(F/'p02_handoff_manifest.json')
    for path,expected in reqs:
        if digest(path)!=expected:
            raise ValueError(f'Review source fingerprint mismatch: {path}')
    run_manifest=load(DEF/'output_sha256.json')
    for name in ('intent-02/study_records.json','intent-02/mapping_contract.json',
                 'intent-02/summary.json','sealed-02/mapping_and_solver_summary.json'):
        if digest(ORIGINAL/name)!=run_manifest[name]:
            raise ValueError(f'Original experiment archive differs: {name}')
    return base,plan,load(E/'production_plan.json'),bank


def norm(x):
    return np.linalg.norm(x,axis=-1)


def unit(x):
    lengths=norm(x)
    valid=(np.isfinite(lengths)&(lengths>=1e-7)).all(axis=tuple(range(1,lengths.ndim)))
    shape=np.expand_dims(lengths,-1)
    return np.divide(x,shape,out=np.zeros_like(x),where=shape>=1e-7),valid


def weight(distance,lo=.016,hi=.024):
    u=np.clip((hi-distance)/(hi-lo),0.,1.)
    return u*u*(3-2*u)


def apply_pinch(points,desired,w):
    """Minimum-energy endpoint deviations for per-pair soft functional constraints."""
    old=points[...,-1,:]
    raw=old[:,:,1:]-old[:,:,:1]
    r=desired-raw
    shift_thumb=-(w[...,None]*r).sum(axis=2)/(1+w.sum(axis=2,keepdims=True))
    shift_fingers=w[...,None]*(r+shift_thumb[:,:,None,:])
    shift=np.concatenate((shift_thumb[:,:,None,:],shift_fingers),axis=2)
    return points+shift[:,:,:,None,:]*np.array([0.,1/3,2/3,1.])[None,None,None,:,None]


def palm_geometry(palm,orders,robot):
    roots=np.asarray([x[0] for x in orders])
    human=np.stack([palm[:,s,ind,:3]-palm[:,s,roots[s]:roots[s]+1,:3][:,:,None,:]
                    for s,(_,ind) in enumerate(orders)],axis=1)
    home=robot.geometry(robot.home)
    from dexgym.tasks.direct.wholebody_s05.p02.intent import hand_basis
    basis=np.stack([hand_basis(x[1:]) for x in home.local[:,:,0]])
    offsets=home.local[:,:,0]
    lengths=norm(np.diff(home.local,axis=2))
    knuckles=human[:,:,1:,0]
    long,v1=unit(knuckles.mean(axis=2))
    across=knuckles[:,:,0]-knuckles[:,:,-1]
    lateral,v2=unit(across-long*np.sum(long*across,axis=-1)[...,None])
    local_basis=np.stack((long,lateral,np.cross(long,lateral)),axis=-1)
    transform=basis[None]@local_basis.swapaxes(-1,-2)
    direction,v3=unit(np.diff(human,axis=3))
    directions=np.einsum('nsij,nsfkj->nsfki',transform,direction)
    shape=np.concatenate((np.broadcast_to(offsets[None,:,:,None],(len(human),2,5,1,3)),
                         offsets[None,:,:,None]+np.cumsum(directions*lengths[None,...,None],axis=3)),axis=3)
    raw=human[:,:,1:,-1]-human[:,:,:1,-1]
    return shape,raw,transform,v1&v2&v3


def prep(palm,orders,robot,cal):
    shape,raw,transform,usable=palm_geometry(palm,orders,robot)
    desired=np.einsum('nsij,nsfj->nsfi',transform,raw)*np.array(cal['hand_scale'])[None,:,None,None]
    d=norm(raw)
    hard=apply_pinch(shape,desired,(d<=.020).astype(float))
    soft=apply_pinch(shape,desired,weight(d))
    return dict(local=hard,smooth=soft,unadjusted=shape,raw_pinch=raw,desired=desired,
                det=np.linalg.det(transform),usable=usable)


def collect(split,prod,bank,robot,cal):
    reader=HMBReader(E/'hmb_manifest.json',E/'production_plan.json',E0/'candidate_config.json',
                    bank['ubuntu_products'],split,GEOMETRY)
    parts=[p for p in prod['partitions'] if p['split']==split]
    rows=[]; counters=Counter(); failures=[]
    for part_id,p in enumerate(parts):
        a,meta,_,_=reader.read(p['key'])
        total=len(a['joint_valid'])
        valid=np.asarray(a['joint_valid'],dtype=bool)
        if not np.all(a['pose_valid'][valid,:4]):
            raise ValueError('Joint valid but a necessary P01 source is invalid')
        root=p['parent']
        counters[root,'planned']+=total
        counters[root,'invalid_hmb']+=total-int(valid.sum())
        if not valid.any():
            continue
        ticks=np.nonzero(valid)[0]
        topo_ids=a['topology_id'][ticks,:2]
        n=len(ticks)
        batch={}
        for codes in np.unique(topo_ids,axis=0):
            mask=(topo_ids==codes).all(axis=1)
            tops=[]
            for s,side in enumerate(SIDE):
                v=meta['dictionary']['topology_id'][int(codes[s])]
                tops.append(next(x for x in meta['topologies'] if (x['stream'],x['version'])==(f'manus_{side}',v)))
            result=prep(a['pose'][ticks[mask],:2],list(map(semantic_order,tops)),robot,cal)
            for k,v in result.items():
                if k not in batch:
                    batch[k]=np.empty((n,*v.shape[1:]),dtype=v.dtype)
                batch[k][mask]=v
        ok=batch.pop('usable') & np.isfinite(a['pose'][ticks,2:4,0]).all(axis=(1,2))
        codes=np.flatnonzero(~ok)
        failures.extend(dict(partition=p['key'],tick=int(ticks[i]),root=root,reason='DEGENERATE_OR_NONFINITE_GEOMETRY') for i in codes)
        counters[root,'unknown']+=len(codes)
        t=ticks[ok]
        counters[root,'scored']+=len(t)
        rows.append(dict(**{k:v[ok] for k,v in batch.items()},wrist=a['pose'][t,2:4,0,:7],
                         time_ns=a['tick_ns'][t],part=np.full(len(t),part_id),tick=t,
                         root=np.full(len(t),root),segment=a['segment_id'][t]))
        if (part_id+1)%32==0:
            print(split,part_id+1,len(parts),flush=True)
    return {k:np.concatenate([r[k] for r in rows],axis=0) for k in rows[0]},parts,counters,failures


def m0(cal):
    return dict(yaw_offset_rad=0.,rotation=cal['world_rotation'],translation=cal['world_translation_m'],
                scale=1.,pivot=[0.,0.,0.],mount=cal['tracker_to_tcp_orientation_mount'])


def apply(cfg,wrist,local,cal):
    r, t = np.asarray(cfg['rotation']),np.asarray(cfg['translation'])
    pivot=np.asarray(cfg['pivot'])
    w=(pivot+(wrist[...,:3]-pivot)*cfg['scale'])@r.T+t
    wrist_orientation=Rotation.from_quat(wrist[...,3:].reshape(-1,4)).as_matrix().reshape(-1,2,3,3)
    r_w=r@wrist_orientation@np.array(cfg['mount'])
    r_p=r_w@np.array(cal['wrist_to_palm_rotation'])
    p_p=w+np.einsum('nsij,sj->nsi',r_w,np.array(cal['wrist_to_palm_translation_m']))
    world=np.einsum('nsij,nsfkj->nsfki',r_p,local)+p_p[:,:,None,None,:]
    return w,r_w,world


def fit(train,base,plan,cal,robot,radii,max_scale=False,initial=None):
    if max_scale and initial is None:
        raise ValueError('M2 requires an M1 fit')
    stride=plan['M1']['sample_stride_per_partition_ticks']
    pick=np.sort(np.unique(np.r_[np.nonzero(train['tick']%stride==0)[0],
                                 np.r_[0,np.nonzero(np.diff(train['part']))[0]+1]]))
    wrist=train['wrist'][pick]
    local=train['local'][pick]
    roots, inverse, n=np.unique(train['root'][pick],return_inverse=True,return_counts=True)
    weights=1/(len(n)*n[inverse])
    cfg=m0(cal)
    pivot=np.einsum('n,nsi->i',weights,wrist[...,:3])/2
    cfg['pivot']=pivot.tolist()
    zero=np.array(cfg['rotation'])
    t0=np.array(cfg['translation'])
    home=robot.geometry(robot.home).wrists.copy()
    b,r=radii
    def transform(x):
        yaw, offset=x[0], x[1:4]
        scale=x[4] if max_scale else 1.
        return cfg|dict(rotation=(Rotation.from_rotvec([0.,0.,yaw]).as_matrix()@zero).tolist(),
                        yaw_offset_rad=float(yaw),translation=(offset+t0).tolist(),scale=float(scale))
    def loss(x):
        w,_,p=apply(transform(x),wrist,local,cal)
        excess=np.maximum(0,norm(p[:,:,:,-1]-b)-r)
        rw=np.square(norm(w-home)).mean(axis=1)
        regularization=.0003*x[0]**2+.001*np.square(x[1:4]).sum()
        if max_scale:
            regularization+=plan['M2']['scale_penalty_m2']*(x[4]-1.)**2
        return float(np.sum(weights*(np.square(excess).mean(axis=(1,2))+.018*rw))+regularization)
    yaw=np.deg2rad(plan['M1']['yaw_offset_bound_deg'])
    tr=plan['M1']['translation_offset_bound_m']
    bounds=[(-yaw,yaw)]+[(-tr,tr)]*3
    if max_scale:
        bounds += [tuple(plan['M2']['scale_limits'])]
        s=np.r_[initial['yaw_offset_rad'],np.array(initial['translation'])-t0,1.]
        starts=[s,s.copy(),s.copy()]
        starts[1][-1]=.86;starts[2][-1]=.75
    else:
        starts=[np.r_[np.deg2rad(y),np.zeros(3)] for y in plan['M1']['yaw_seed_offsets_deg']]
    outcomes=[]
    for x in starts:
        result=minimize(loss,x,method='L-BFGS-B',bounds=bounds,
                        options={'maxiter':plan['M1']['optimizer_max_iterations'],'ftol':1e-13})
        outcomes.append(dict(initial=x.tolist(),loss=float(result.fun),x=result.x.tolist(),
                             status=int(result.status),iterations=int(result.nit),message=str(result.message)))
    winning=min(outcomes,key=lambda y:y['loss'])
    return transform(np.array(winning['x'])),dict(train_samples=len(pick),train_roots=len(roots),restarts=outcomes)


def stats(x,axis=0):
    if not len(x):
        return None
    return np.quantile(x,[0.,.05,.50,.95,.99,1.],axis=axis).tolist()


def span(w,r):
    delta=w[:,1]-w[:,0]
    dv=np.einsum('nij,nj->ni',r[:,0].transpose(0,2,1),delta)
    so3=Rotation.from_matrix(r[:,0].transpose(0,2,1)@r[:,1]).as_rotvec()
    return dict(wrist_xyz_m=stats(w),midpoint_xyz_m=stats(w.mean(axis=1)),
                separation_m=stats(norm(delta)),relative_wrist_xyz_m=stats(dv),
                relative_unit_world=stats(delta/np.maximum(norm(delta),1e-15)[:,None]),
                relative_rotation_vector_rad=stats(so3),relative_angle_rad=stats(norm(so3)))


def summarize(group,w,r,world,raw,base_world,origin,radii,jump,counters,roots):
    w,r,world,raw,old,jump=[x[group] for x in (w,r,world,raw,base_world,jump)]
    tips=world[:,:,:,-1]
    delta=tips-origin
    low=np.maximum(0.,norm(delta)-radii-1e-9)
    largest=low.max(axis=(1,2))
    cross_pairs=((0,0),(1,1),(2,2),(3,3),(4,4),(0,1),(1,0),(0,2),(2,0))
    cross=np.stack([tips[:,1,b]-tips[:,0,a]-(old[:,1,b,-1]-old[:,0,a,-1]) for a,b in cross_pairs],axis=1)
    return dict(**span(w,r),raw_wrist_separation_m=stats(norm(raw[:,1,:3]-raw[:,0,:3])),
                planned=sum(counters[root,'planned'] for root in roots),
                invalid_hmb=sum(counters[root,'invalid_hmb'] for root in roots),
                unknown=sum(counters[root,'unknown'] for root in roots),scored=len(w),
                tip_relative_to_pivot_xyz_m=stats(delta),
                tip_radius_m=stats(norm(delta)),lower_bound_per_tip_m=stats(low),
                lower_bound_max_m=stats(largest),
                strict_above={str(int(th*1000)):dict(n=int((largest>th).sum()),per_side=(low.max(axis=2)>th).sum(axis=0).tolist())
                              for th in THRESHOLDS},
                necessary_domain_above_3mm=int((largest>.003).sum()),original_target_retained=len(w),
                world_landmark_distortion_from_M0_m=stats(norm(world-old)),
                world_tip_distortion_from_M0_m=stats(norm(world[:,:,:,-1]-old[:,:,:,-1])),
                cross_hand_distortion_from_M0_m=stats(norm(cross)),
                wrist_jump_cross={str(int(v*1000)):dict(jump_ticks=int((jump>v).sum()),
                    jump_and_outside_3mm=int(((jump>v)&(largest>.003)).sum()),
                    no_jump_and_outside_3mm=int(((jump<=v)&(largest>.003)).sum()),
                    comparable=int(np.isfinite(jump).sum())) for v in JUMP})


def timeline(d,parts,output):
    same=np.r_[False,(np.diff(d['part'])==0)&(np.diff(d['tick'])==1)&(np.diff(d['time_ns'])==20_000_000)]
    pair=same & (d['segment']>=0) & np.r_[False,np.diff(d['segment'])==0]
    jump=np.r_[np.nan,norm(np.diff(d['wrist'][...,:3],axis=0)).max(axis=1)]
    jump[~same]=np.nan
    raw=norm(d['raw_pinch'])
    old=d['local'][:,:,:,-1]
    new=d['smooth'][:,:,:,-1]
    active=raw<=.020
    change=active ^ np.concatenate((active[:1],active[:-1]),axis=0)
    pairs={}
    records=[]
    for s,side in enumerate(SIDE):
        for direction in ('entry','exit'):
            event=(change[:,s]&(active[:,s] if direction=='entry' else ~active[:,s])).any(axis=1)&pair
            seq=np.flatnonzero(event)
            if len(seq):
                k=int(seq[0])
                records.append(dict(side=side,direction=direction,root=d['root'][k],partition=parts[d['part'][k]]['key'],
                                    tick=int(d['tick'][k]),previous_tick=int(d['tick'][k-1])))
            pairs[side,direction]=seq
    out={}
    for name,tip,geom in [('hard',old,d['local']),('smooth',new,d['smooth'])]:
        step=np.concatenate((np.zeros_like(tip[:1]),norm(np.diff(tip,axis=0))[...,None]*np.ones(3)),axis=0)[...,0]
        vtip=tip[:,:,1:]-tip[:,:,:1]
        norms=norm(vtip)
        counts={f'{side}_{direct}':dict(n=len(idx),step_m=stats(step[idx,SIDE.index(side)].max(axis=1)),
                                       proxy_speed_m_s=stats(step[idx,SIDE.index(side)].max(axis=1)/.020))
                for (side,direct),idx in pairs.items()}
        geom_err=norm(vtip-d['desired'])
        bone=np.abs(norm(np.diff(geom,axis=3))-norm(np.diff(d['unadjusted'],axis=3)))
        out[name]=dict(all_tip_step_m=stats(step[pair]),threshold_crossings=counts,
                       active_vector_error_m=stats(geom_err[active]),
                       active_distance_error_m=stats(np.abs(norms-norm(d['desired']))[active]),
                       bone_abs_change_m=stats(bone.ravel()),tip_delta_from_hard_m=stats(norm(tip-old)))
    np.savez_compressed(output/'functional_traces.npz',raw_pinch_m=raw,hard_tip=old,soft_tip=new,
                        time_ns=d['time_ns'],root=d['root'],part=d['part'],tick=d['tick'],consecutive=pair)
    return jump,dict(proxy_velocity_period_s=.020,continuous_pairs=int(pair.sum()),
                     nearest_extremes_threshold_m=[.016,.020,.024],scenario=out,fk_witnesses=records)


def process(mode,run):
    base,plan,prod,bank=source()
    cal=load(DEF/'mapping_contract.json')
    robot=RobotModel(base['model'],base['clearance_m'])
    origin,r=reach_envelope(robot)
    split='Train' if mode=='train' else 'Validation'
    out=run/split.lower()
    out.mkdir(mode=0o700)
    frozen=run/'fitted_candidates.json'
    marker=run/'fitted_candidates.sha256'
    if mode!='train':
        if not marker.is_file() or marker.read_text().strip()!=digest(frozen):
            raise ValueError('Validation requires independently frozen Train candidate bytes')
        fits=load(frozen)
        if fits['fit_scope']!='Train' or fits['plan_sha256']!=digest(DEF/'refinement_plan.json'):
            raise ValueError('Candidate fit scope or plan differs')
    start=perf_counter()
    d,parts,counts,fail=collect(split,prod,bank,robot,cal)
    if split=='Train':
        candidate,m1=fit(d,base,plan,cal,robot,(origin,r))
        _,_,target=apply(candidate,d['wrist'],d['local'],cal)
        bound=np.maximum(0,norm(target[:,:,:,-1]-origin)-r-1e-9).max(axis=(1,2))
        fraction=float(np.mean(bound>.003))
        configs=dict(M0=m0(cal),M1=candidate)
        info=dict(M1=m1,M1_train_fraction_above_3mm=fraction,M2_triggered=fraction>.02)
        if fraction>.02:
            configs['M2'],info['M2']=fit(d,base,plan,cal,robot,(origin,r),max_scale=True,initial=candidate)
        fits=dict(fit_scope='Train',plan_sha256=digest(DEF/'refinement_plan.json'),
                  model_hash=base['model_sha256'],source_config_sha256=digest(DEF/'mapping_contract.json'),
                  configs=configs,optimizer=info)
        write(frozen,fits)
        marker.write_text(digest(frozen)+'\n')
    jump,func=timeline(d,parts,out)
    baseline=apply(fits['configs']['M0'],d['wrist'],d['local'],cal)[2]
    tasks={p['root']:p['task'] for p in prod['sources']}
    roots=sorted({p['parent'] for p in parts})
    if len(roots)!=(24 if split=='Train' else 8):
        raise ValueError('Unexpected signed split denominator')
    for root in roots:
        if counts[root,'planned']!=sum(counts[root,p] for p in ('scored','invalid_hmb','unknown')):
            raise ValueError(f'Denominator does not conserve for {root}')
    root_masks={root:d['root']==root for root in roots}
    groups={'all':roots,**{root:[root] for root in roots}}
    groups.update({f'task:{task}':[root for root in roots if tasks[root]==task] for task in sorted({tasks[r] for r in roots})})
    per={}
    for name,cfg in fits['configs'].items():
        w,rot,xyz=apply(cfg,d['wrist'],d['local'],cal)
        per[name]={label:summarize(np.logical_or.reduce([root_masks[r] for r in rr]),w,rot,xyz,d['wrist'],
                                    baseline,origin,r,jump,counts,rr) for label,rr in groups.items()}
    expected={'Train': 0,'Validation': 0}   # E1 identity, verified below from signed product summary.
    expected[split]=sum(counts[root,'scored']+counts[root,'unknown'] for root in roots)
    raw=wrist_report(d)
    write(out/'workspace_summary.json',dict(split=split,source_roots=len(roots),source_partition_count=len(parts),
          signed_joint_valid=expected[split],time_quality='HISTORICAL_HMB_SIMULATED',
          scope='Necessary radial bound only. A zero bound supplies no sufficiency or collision evidence.',
          radius_origin_m=origin.tolist(),radius_m=r.tolist(),geometry_failures=fail,
          counts={'planned':sum(counts[root,'planned'] for root in roots),
                  'HMB_invalid':sum(counts[root,'invalid_hmb'] for root in roots),
                  'UNKNOWN':len(fail),'qualified':len(d['tick'])},
          raw=raw,candidates=per,fits_sha256=digest(frozen),duration_s=perf_counter()-start))
    write(out/'functional_summary.json',func|dict(proper_human_to_model_rotation_det=stats(d['det'])))
    print(f'{split} qualified {len(d["tick"])}; M0/M1/M2 bounds',
          {k:v['all']['strict_above'] for k,v in per.items()},flush=True)


def wrist_report(d):
    w=d['wrist'][...,:3]
    r=Rotation.from_quat(d['wrist'][...,3:].reshape(-1,4)).as_matrix().reshape(-1,2,3,3)
    return span(w,r)


def from_frame(a,meta,t,robot,cal,cfg,pairs,smooth=False):
    ids=a['topology_id'][t,:2]
    top=[]
    for s,side in enumerate(SIDE):
        ver=meta['dictionary']['topology_id'][int(ids[s])]
        top.append(next(x for x in meta['topologies'] if (x['stream'],x['version'])==(f'manus_{side}',ver)))
    fine=prep(a['pose'][t:t+1,:2],list(map(semantic_order,top)),robot,cal)
    if not fine['usable'][0]:
        raise ValueError('Degenerate witness')
    local=fine['smooth' if smooth else 'local']
    w,r,p=apply(cfg,a['pose'][t:t+1,2:4,0],local,cal)
    from dexgym.tasks.direct.wholebody_s05.p02.intent import target_geometry
    return target_geometry(w[0],r[0],local[0],
             (np.array(cal['wrist_to_palm_translation_m']),np.array(cal['wrist_to_palm_rotation'])),pairs),fine


if __name__=='__main__':
    process(sys.argv[1],Path(sys.argv[2]))
