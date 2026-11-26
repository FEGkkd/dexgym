"""Independent tree FK and central-difference validation on both signed models."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.p02.model import (
    COUPLING_LIMIT, PALMS, WRISTS, RobotModel, digest, independent_frames,
)


def sample(robot,seed):
    rng=np.random.default_rng(seed)
    eligible=[]
    low,high=robot.bounds.T
    for _ in range(64):
        q=np.clip(robot.home+rng.uniform(-.22,.22,54)*(high-low),low+1e-6,high-1e-6)
        if robot.legality(q)['feasible']:
            eligible.append(q)
    need=12
    if len(eligible)<need:
        for _ in range(80):
            q=np.clip(robot.home+rng.uniform(-.06,.06,54)*(high-low),low+1e-6,high-1e-6)
            if robot.legality(q)['feasible']:
                eligible.append(q)
    if len(eligible)<need:
        raise ValueError(f'Only {len(eligible)} legal random states')
    robot.set(robot.home)
    base=[('home',robot.home)]
    for i in (0,5,6,7,12,13,14,33,34,53):
        q=robot.home.copy()
        q[i]+=0.015 if q[i]+.015<high[i] else -.015
        if robot.legality(q)['feasible']:
            base.append(('single_axis',q))
    accepted=[]
    for i in (0,2,3,4,7,9,10,11,14,19,34,39,53):
        for v in (low[i]+1e-5,high[i]-1e-5):
            q=robot.home.copy()
            q[i]=v
            if robot.legality(q)['feasible']:
                accepted.append(('near_limit',q))
                break
    base.extend(accepted[:6])
    for s in range(2):
        for sign in (-1,1):
            q=robot.home.copy()
            j,k=s*7+5,s*7+6
            q[j]=np.deg2rad(sign*49)
            q[k]=sign*(COUPLING_LIMIT-1.025*abs(q[j])-1e-5)
            if robot.legality(q)['feasible']:
                base.append(('coupling_boundary',q))
    def singular(q):
        robot.set(q)
        p,a=robot.point_jac('site',WRISTS[0])
        return np.linalg.svd(np.vstack((p[:,:7],a[:,:7])),compute_uv=False)[-1]
    base.extend([('random_legal',q) for q in eligible[:12]])
    base.extend([('lower_sigma_legal',q) for q in sorted(eligible,key=singular)[:2]])
    return base


def errors(robot,q,steps):
    robot.set(q)
    m,d=robot.model,robot.data
    ref=independent_frames(robot,q)
    pos,ori=0.,0.
    jac={}
    for (kind,name),(p,r) in ref.items():
        if kind=='joint':
            x=d.joint(name).xanchor
            a=d.body(int(m.joint(name).bodyid[0])).xmat.reshape(3,3)
        else:
            obj=getattr(d,kind)(name)
            x,a=obj.xpos,obj.xmat.reshape(3,3)
        pos=max(pos,float(np.linalg.norm(p-x)))
        ori=max(ori,float(Rotation.from_matrix(a@r.T).magnitude()))
        jac[kind,name]=robot.point_jac(kind,name)
    out={'fk_m':pos,'fk_rotation_rad':ori}
    for h in steps:
        ep,ea=0.,0.
        for j in range(54):
            delta=np.zeros(54)
            delta[j]=h
            a,b=independent_frames(robot,q+delta),independent_frames(robot,q-delta)
            for key in ref:
                a0,a1=a[key]; b0,b1=b[key]
                jp,ja=jac[key]
                ep=max(ep,float(np.linalg.norm((a0-b0)/(2*h)-jp[:,j])))
                va=Rotation.from_matrix(a1@b1.T).as_rotvec()/(2*h)
                ea=max(ea,float(np.linalg.norm(va-ja[:,j])))
        out[str(h)]={'position':ep,'angular':ea}
    return out


def check_collision_catalog(robot,seed):
    rng=np.random.default_rng(seed+901)
    actual=Counter()
    missing=set()
    low,high=robot.bounds.T
    catalog=set(robot.collision_pairs)
    for _ in range(36):
        q=rng.uniform(low,high)
        robot.set(q,contacts=True)
        for c in robot.data.contact:
            if c.dist>=0:
                continue
            a,b=sorted((int(c.geom1),int(c.geom2)))
            actual[(a,b)]+=1
            if (a,b) not in catalog:
                missing.add((a,b))
    return dict(observed_negative_contacts=sum(actual.values()),pairs_observed=len(actual),uncatalogued=sorted(missing))


def pad_geometry(robot):
    """Measure site separation from the model's distal body origin and collision bounds."""
    output=[]
    for group in robot.landmark_names:
        for finger in group:
            site=robot.model.site(finger[-1])
            body=int(site.bodyid[0])
            ancestor=int(robot.model.body_parentid[body])
            bodies={body,ancestor}
            colliders=[int(i) for i,b in enumerate(robot.model.geom_bodyid)
                       if b in bodies and robot.model.geom_contype[i]]
            p=robot.data.site(site.name).xpos
            output.append(dict(site=site.name,tip_to_parent_origin_m=float(
                np.linalg.norm(p-robot.data.body(ancestor).xpos)),colliders=[
                dict(id=i,name=robot.model.geom(i).name,geometry_type=int(robot.model.geom_type[i]),
                     center_to_functional_site_m=float(np.linalg.norm(robot.data.geom(i).xpos-p)),
                     bounding_radius_m=float(robot.model.geom_rbound[i])) for i in colliders],
                physical_pad_offset='UNKNOWN_NO_EXTERNAL_PAD_MEASUREMENT'))
    return output


def evaluate(path,expected,plan):
    if digest(path)!=expected:
        raise ValueError(f'Unapproved MJCF source: {path}')
    r=RobotModel(path,plan['clearance_m'])
    t=plan['tolerance']
    chosen=sample(r,plan['seed'])
    records=[]
    for i,(kind,q) in enumerate(chosen):
        # Step sensitivity on representative home/boundary cases, frozen middle step at remaining cases.
        steps=t['fd_steps_rad'] if i<=2 or kind=='coupling_boundary' else [1e-5]
        e=errors(r,q,steps)
        records.append(dict(scenario=kind,index=i,q=q.tolist(),errors=e,legality=r.legality(q)))
    catalog=check_collision_catalog(r,plan['seed'])
    contract=r.contract()
    pad=pad_geometry(r)
    good=not catalog['uncatalogued'] and all(
        x['errors']['fk_m']<=t['fk_m'] and x['errors']['fk_rotation_rad']<=t['fk_rotation_rad']
        and max(x['errors']['1e-05'].values())<=t['jacobian'] for x in records)
    groups=Counter(x['scenario'] for x in records)
    good=good and all(groups[k] for k in ('random_legal','near_limit','coupling_boundary','single_axis','lower_sigma_legal'))
    return dict(contract=contract,pad_proxy=pad,valid=good,groups=groups,results=records,contact_catalog_test=catalog,
                maximums={k:max(x['errors'][k] for x in records) for k in ('fk_m','fk_rotation_rad')}|
                {k:max(x['errors']['1e-05'][k] for x in records) for k in ('position','angular')})


if __name__=='__main__':
    plan=json.loads(Path(sys.argv[1]).read_bytes())
    result={}
    for label in ('model','historical_model'):
        result[label]=evaluate(plan[label],plan[label+'_sha256'],plan)
        m=result[label]
        print(label,{'valid':m['valid'],'groups':m['groups'],'maximums':m['maximums'],
                     'home':m['contract']['home_constraints'],'catalog':m['contact_catalog_test']},flush=True)
    with Path(sys.argv[2]).open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True)
    if not all(m['valid'] for m in result.values()):
        sys.exit(3)
