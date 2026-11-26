"""Finite-budget constrained static Oracle with analytic task Jacobians."""
from time import perf_counter

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from .intent import robot_target
from .model import COUPLING_LIMIT, PALMS, WRISTS


def skew(v):
    x,y,z=v
    return np.asarray([[0,-z,y],[z,0,-x],[-y,x,0]])


def so3(goal,actual,world_jac):
    phi=Rotation.from_matrix(goal.T@actual).as_rotvec()
    mag=np.linalg.norm(phi)
    a=1/12 if mag<1e-5 else (1-.5*mag/np.tan(.5*mag))/(mag*mag)
    w=skew(phi)
    return phi,(np.eye(3)-.5*w+a*w@w)@goal.T@world_jac


class Objective:
    def __init__(self,robot,target,config):
        self.robot=robot
        self.target=target
        self.scales=config['scales']
        self.pairs=config['cross_hand_pairs']
        self.keys=[]
        self.q=None

    def evaluate(self,q):
        if self.q is not None and np.array_equal(q,self.q):
            self.robot.set(q)
            return self.r,self.j
        r=self.robot
        g=r.geometry(q)
        goal=self.target
        d,js=[],[]
        def add(key,value,jac):
            s=self.scales[key]
            d.append(np.asarray(value).reshape(-1)/s)
            js.append(np.asarray(jac).reshape(-1,54)/s)
        wp,wa=map(np.asarray,zip(*(r.point_jac('site',n) for n in WRISTS),strict=True))
        pp,pa=map(np.asarray,zip(*(r.point_jac('body',n) for n in PALMS),strict=True))
        jp=np.asarray([[[r.point_jac('joint' if i<3 else 'site',n)[0] for i,n in enumerate(f)]
                         for f in group] for group in r.landmark_names])
        jlocal=np.empty_like(jp)
        for side in range(2):
            for finger in range(5):
                for n in range(4):
                    pos=g.landmarks[side,finger,n]
                    jlocal[side,finger,n]=g.palm_rotation[side].T@(
                        jp[side,finger,n]-pp[side]+skew(pos-g.palm[side])@pa[side])
        add('world_tip_m',g.tips-goal.world[:,:,-1],jp[:,:,-1])
        add('local_landmark_m',g.local-goal.local,jlocal)
        add('pinch_vector_m',g.local[:,1:,-1]-g.local[:,:1,-1]-goal.pinch,
            jlocal[:,1:,-1]-jlocal[:,:1,-1])
        add('cross_hand_vector_m',[g.tips[1,b]-g.tips[0,a]-goal.cross[i] for i,(a,b) in enumerate(self.pairs)],
            [jp[1,b,-1]-jp[0,a,-1] for a,b in self.pairs])
        add('wrist_translation_m',g.wrists-goal.wrist,wp)
        for s in range(2):
            w,j=so3(goal.rotation[s],g.rotations[s],wa[s])
            add('wrist_rotation_rad',w,j)
        delta=g.wrists[1]-g.wrists[0]
        rel=g.rotations[0].T
        add('relative_wrist_translation_m',rel@delta-goal.relative_wrist,
            rel@(wp[1]-wp[0]+skew(delta)@wa[0]))
        w,j=so3(goal.relative_rotation,rel@g.rotations[1],rel@(wa[1]-wa[0]))
        add('relative_wrist_rotation_rad',w,j)
        add('joint_posture_rad',q-r.home,np.eye(54))
        self.q=np.asarray(q).copy()
        self.r,self.j=np.concatenate(d),np.concatenate(js)
        return self.r,self.j

    def cost(self,q):
        value,_=self.evaluate(q)
        return float(.5*(value@value))


def metrics(target,robot,q,pairs):
    g=robot_target(robot,robot.geometry(q),pairs)
    error=g.world[:,:,-1]-target.world[:,:,-1]
    per_tip=np.linalg.norm(error,axis=-1)
    return dict(max_tip_m=float(per_tip.max()),mean_tip_m=float(per_tip.mean()),tip_m=per_tip.tolist(),
                tip_vectors_m=error.tolist(),
                local_landmark_rmse_m=float(np.sqrt(np.mean(np.square(np.linalg.norm(g.local-target.local,axis=-1))))),
                pinch_vector_m=np.linalg.norm(g.pinch-target.pinch,axis=-1).tolist(),
                pinch_distance_m=np.abs(np.linalg.norm(g.pinch,axis=-1)-np.linalg.norm(target.pinch,axis=-1)).tolist(),
                cross_hand_m=np.linalg.norm(g.cross-target.cross,axis=-1).tolist(),
                wrist_translation_m=np.linalg.norm(g.wrist-target.wrist,axis=-1).tolist(),
                wrist_rotation_rad=Rotation.from_matrix(g.rotation@target.rotation.transpose(0,2,1)).magnitude().tolist(),
                relative_wrist_translation_m=float(np.linalg.norm(g.relative_wrist-target.relative_wrist)),
                relative_wrist_rotation_rad=float(Rotation.from_matrix(g.relative_rotation@target.relative_rotation.T).magnitude()))


def wrist_seed(robot,target,q):
    q=np.asarray(q).copy()
    q=np.clip(q,*robot.bounds.T)
    def full(x):
        return np.r_[x,q[14:]]
    def cost(x):
        g=robot.geometry(full(x))
        p=(g.wrists-target.wrist)/.035
        w=Rotation.from_matrix(target.rotation.transpose(0,2,1)@g.rotations).as_rotvec()/.22
        return np.r_[p.ravel(),w.ravel(),(x-robot.home[:14])*.1]
    def jac(x):
        g=robot.geometry(full(x))
        p,w=[],[]
        for i,name in enumerate(WRISTS):
            jp,ja=robot.point_jac('site',name)
            p.append(jp[:,:14]/.035)
            w.append(so3(target.rotation[i],g.rotations[i],ja)[1][:,:14]/.22)
        return np.vstack((*p,*w,.1*np.eye(14)))
    result=least_squares(cost,q[:14],jac=jac,bounds=robot.bounds[:14].T,max_nfev=28)
    return full(result.x)


def joint_seed(objective,q):
    q=np.clip(q,*objective.robot.bounds.T)
    result=least_squares(lambda x:objective.evaluate(x)[0],q,
                         jac=lambda x:objective.evaluate(x)[1],bounds=objective.robot.bounds.T,max_nfev=24)
    return np.asarray(result.x)


def collision_gradient(robot,pair,gap):
    if pair is None or abs(gap)<1e-8:
        return np.zeros(54)
    m,d=robot.model,robot.data
    a,b=pair
    from_to=np.zeros(6)
    mujoco.mj_geomDistance(m,d,a,b,.016,from_to)
    normal=(from_to[3:]-from_to[:3])/gap
    if not np.isfinite(normal).all() or not .5<np.linalg.norm(normal)<1.5:
        return np.zeros(54)
    first,second=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    mujoco.mj_jac(m,d,first,None,from_to[:3],int(m.geom_bodyid[a]))
    mujoco.mj_jac(m,d,second,None,from_to[3:],int(m.geom_bodyid[b]))
    return normal@(second-first)[:,robot.v_indices]


class TimeBudget(Exception):
    pass


def run_start(obj,config,bounds,indices,initial,label):
    robot=obj.robot
    started=perf_counter()
    limits=config['solver']
    duration=limits['wall_seconds_per_start']
    lo,hi=bounds.T
    z=np.clip(initial,lo,hi)
    before=z.copy()
    best=None
    best_cost=float('inf')
    niter=0
    def gate():
        if perf_counter()-started>duration:
            raise TimeBudget
    def full(x):
        q=z.copy()
        q[indices]=x
        return q
    def f(x):
        gate()
        q=full(x)
        d,j=obj.evaluate(q)
        return .5*(d@d),(d@j)[indices]
    last_collision=None
    def col(x):
        nonlocal last_collision
        if last_collision is not None and np.array_equal(last_collision[0],x):
            return last_collision[1:]
        gate()
        robot.set(full(x))
        gap,pair=robot.contact_gap()
        jac=collision_gradient(robot,pair,gap)[indices]
        last_collision=(x.copy(),gap-robot.clearance_m,jac)
        return last_collision[1:]
    def witness(x):
        nonlocal best,best_cost
        q=full(x)
        value=obj.cost(q)
        if value<best_cost and robot.legality(q)['feasible']:
            best,best_cost=q,value
    def callback(x):
        nonlocal niter
        niter+=1
        gate()
        witness(x)
    x=z[indices]
    solver_success=False
    solver_status=None
    exhausted=False
    try:
        witness(x)
        if len(indices):
            cup=robot.coupling[:,indices]
            constraints=[{'type':'ineq','fun':lambda x:COUPLING_LIMIT-robot.coupling@full(x),'jac':lambda x:-cup},
                         {'type':'ineq','fun':lambda x:col(x)[0],'jac':lambda x:col(x)[1]}]
            result=minimize(f,x,method='SLSQP',jac=True,bounds=list(zip(lo[indices],hi[indices])),
                            constraints=constraints,callback=callback,
                            options={'maxiter':limits['max_iterations_per_start'],'ftol':limits['ftol']})
            status=str(result.message)
            solver_success=bool(result.success)
            solver_status=int(result.status)
            witness(result.x)
        else:
            status='ZERO_FREE_AXES'
    except TimeBudget:
        exhausted=True
        status='WALL_TIME_BUDGET'
    evidence=dict(label=label,initial=before.tolist(),input_initial=initial.tolist(),solver_success=solver_success,
                  solver_status=solver_status,termination=status,elapsed_s=perf_counter()-started,
                  budget_exhausted=exhausted or solver_status==9,iterations=niter,
                  feasible_best=best is not None)
    if best is not None:
        evidence.update(best_value=best_cost,best_q=best.tolist(),legality=robot.legality(best))
    return best,best_cost,evidence


def solve(robot,target,config,starts,indices=None,bounds=None):
    obj=Objective(robot,target,config)
    if bounds is None:
        bounds=robot.bounds.copy()
    indices=np.arange(54) if indices is None else np.asarray(indices)
    attempts=[]
    winners=[]
    for label,q in starts:
        result,cost,row=run_start(obj,config,bounds,indices,np.asarray(q),label)
        attempts.append(row)
        if result is not None:
            winners.append((cost,result))
    if not winners:
        state='SOLVER_TIMEOUT' if any(a['budget_exhausted'] for a in attempts) else 'UNKNOWN'
        return dict(status=state,q=None,attempts=attempts)
    value,q=min(winners,key=lambda x:x[0])
    check=robot.legality(q)
    if not check['feasible']:
        raise ValueError('Returned witness failed fresh forward/contact verification')
    stats=metrics(target,robot,q,config['cross_hand_pairs'])
    threshold=config['tolerance']['feasible_at_tolerance_tip_m']
    # The label names only the frozen ten-tip benchmark; all other objective residuals remain separate.
    status='FEASIBLE_AT_TOLERANCE' if stats['max_tip_m']<=threshold else 'BEST_FOUND_FEASIBLE_WITH_RESIDUAL'
    low,high=bounds.T
    return dict(status=status,q=q.tolist(),attempts=attempts,objective_value=value,metrics=stats,legality=check,
                active_limits=np.flatnonzero((q-low<1e-5)|(high-q<1e-5)).tolist(),
                active_coupling=check['coupling_margin_rad']<1e-5,
                active_collision=check['collision_clearance_m']-robot.clearance_m<1e-5)
