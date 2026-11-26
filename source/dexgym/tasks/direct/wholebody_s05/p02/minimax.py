"""Static diagnostic epigraph minimization for ten named robot fingertip errors."""
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .model import COUPLING_LIMIT
from .oracle import TimeBudget, collision_gradient, metrics


EPIGRAPH_SCALE_M = .01


def tip_constraints(robot,target,x):
    """Return z−|error_i(q)| and Jacobian, in 10 mm units."""
    q=x[:-1]
    error=(robot.geometry(q).tips-target.world[:,:,-1]).reshape(10,3)
    e=np.linalg.norm(error,axis=1)
    j=np.array([robot.point_jac('site',robot.model.site(i).name)[0] for i in robot.tip_ids])
    v=np.divide(error,e[:,None],out=np.zeros_like(error),where=e[:,None]>1e-13)
    return x[-1]-e/EPIGRAPH_SCALE_M, np.c_[-np.einsum('ti,tij->tj',v,j)/EPIGRAPH_SCALE_M,np.ones(10)]


def run_start(robot,target,initial,label,budget_s,iterations):
    started=perf_counter()
    q0=np.clip(initial,*robot.bounds.T)
    t=target.world[:,:,-1]
    max_e=lambda q:float(np.linalg.norm(robot.geometry(q).tips-t,axis=-1).max())
    x=np.r_[q0,max_e(q0)/EPIGRAPH_SCALE_M]
    best,best_value=None,float('inf')
    count=0
    cache_col=cache_tips=None
    def gate():
        if perf_counter()-started>budget_s:
            raise TimeBudget
    def tips(y):
        nonlocal cache_tips
        gate()
        if cache_tips is None or not np.array_equal(cache_tips[0],y):
            cache_tips=(y.copy(),*tip_constraints(robot,target,y))
        return cache_tips[1:]
    def collision(y):
        nonlocal cache_col
        gate()
        if cache_col is None or not np.array_equal(cache_col[0],y[:-1]):
            robot.set(y[:-1])
            gap,pair=robot.contact_gap()
            grad=collision_gradient(robot,pair,gap)
            cache_col=(y[:-1].copy(),gap-robot.clearance_m,np.r_[grad,0.])
        return cache_col[1:]
    def retain(y):
        nonlocal best,best_value
        q=y[:-1]
        value=max_e(q)
        if value<best_value and robot.legality(q)['feasible']:
            best,best_value=q.copy(),value
    def iteration(y):
        nonlocal count
        count+=1
        gate()
        retain(y)
    exhausted=False
    solution=None
    try:
        retain(x)
        solution=minimize(lambda y:y[-1],x,jac=lambda _:np.r_[np.zeros(54),1.],method='SLSQP',
                          bounds=list(map(tuple,robot.bounds))+[(0.,None)],
                          constraints=[
                              dict(type='ineq',fun=lambda y:tips(y)[0],jac=lambda y:tips(y)[1]),
                              dict(type='ineq',fun=lambda y: COUPLING_LIMIT-robot.coupling@y[:-1],
                                   jac=lambda _:np.c_[-robot.coupling,np.zeros(8)]),
                              dict(type='ineq',fun=lambda y:collision(y)[0],jac=lambda y:collision(y)[1])],
                          callback=iteration,options=dict(maxiter=iterations,ftol=1e-8))
        retain(solution.x)
    except TimeBudget:
        exhausted=True
    status=None if solution is None else int(solution.status)
    result=dict(label=label,initial=q0.tolist(),initial_epigraph_m=float(x[-1]*EPIGRAPH_SCALE_M),
                elapsed_s=perf_counter()-started,iterations=count,budget_exhausted=exhausted or status==9,
                termination='WALL_TIME_BUDGET' if solution is None else str(solution.message),solver_status=status,
                solver_success=solution is not None and bool(solution.success),feasible_best=best is not None,
                best_tip_m=None if best is None else best_value,
                terminal_epigraph_m=None if solution is None else float(solution.x[-1]*EPIGRAPH_SCALE_M))
    if best is not None:
        result.update(best_q=best.tolist(),legality=robot.legality(best))
    return best,best_value,result


def solve_minimax(robot,target,plan,starts,iterations=300):
    records=[run_start(robot,target,np.asarray(initial),label,plan['solver']['wall_seconds_per_start'],iterations)
             for label,initial in starts]
    attempts=[r[2] for r in records]
    feasible=[(q,v) for q,v,_ in records if q is not None]
    if not feasible:
        return dict(status='SOLVER_TIMEOUT' if any(a['budget_exhausted'] for a in attempts) else 'UNKNOWN',
                    q=None,attempts=attempts)
    q,z=min(feasible,key=lambda item:item[1])
    legality=robot.legality(q)
    if not legality['feasible']:
        raise ValueError('Minimax output failed repeated legality check')
    geometric=metrics(target,robot,q,plan['cross_hand_pairs'])
    if abs(geometric['max_tip_m']-z)>1e-10:
        raise ValueError('Archived epigraph value differs from actual tip residual')
    return dict(q=q.tolist(),epigraph_m=z,metrics=geometric,legality=legality,attempts=attempts,
                status='FEASIBLE_AT_TOLERANCE' if z<=plan['tolerance']['feasible_at_tolerance_tip_m']
                else 'BEST_FOUND_FEASIBLE_WITH_RESIDUAL')
