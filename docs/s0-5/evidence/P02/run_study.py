"""Measure independent mapped intent and fixed-target 14/54-DoF feasibility."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np

from dexgym.data.wuji_retarget import WujiRetargetSession
from dexgym.tasks.direct.wholebody_s05.hmb_reader import GEOMETRY, HMBReader
from dexgym.tasks.direct.wholebody_s05.p02.intent import Mapper, robot_target
from dexgym.tasks.direct.wholebody_s05.p02.model import COUPLING_LIMIT, RobotModel, digest, independent_frames
from dexgym.tasks.direct.wholebody_s05.p02.oracle import Objective, joint_seed, metrics, solve, wrist_seed


DOCS=Path('docs/s0-5')
E=DOCS/'evidence/P01_E_E1'
E0=DOCS/'evidence/P01_E_E0'
F=DOCS/'evidence/P01_F'


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path,data):
    with path.open('x',encoding='utf-8') as p:
        json.dump(data,p,indent=2,sort_keys=True,allow_nan=False,default=numpy_scalar)
        p.write('\n')


def numpy_scalar(value):
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(f'Unexpected value in P02 evidence: {type(value).__name__}')


def verify_approval(config):
    state=load(DOCS/'phase_state.json')
    if state['current']['P01']!='ACCEPTED' or state['current']['P02']!='IN_PROGRESS':
        raise ValueError('P02 requires independent P01 acceptance and P02 authorization')
    for f,attr in ((F/'p02_handoff_manifest.json','handoff_sha256'),(E/'hmb_manifest.json','e1_manifest_sha256'),
                   (Path(config['model']),'model_sha256')):
        if digest(f)!=config[attr]:
            raise ValueError(f'Frozen identity differs: {f}')
    handoff=load(F/'p02_handoff_manifest.json')
    if handoff['e1_manifest_sha256']!=config['e1_manifest_sha256'] or handoff['technical_gate']!='PASS':
        raise ValueError('Unapproved HMB candidate')
    return handoff


def samples(config,handoff):
    all_splits=('Train','Validation')
    index=load(E/'build_index.json')
    parts={a['partition']['key']:a['partition'] for a in index['partitions']}
    roots=load(E/'production_plan.json')['sources']
    tasks={r['root']:r['task'] for r in roots}
    cohorts=[]
    picks=load(F/'selected_samples.json')
    for root,row in sorted(picks['roots'].items()):
        cohorts.append((f'root:{root}',row,'ROOT'))
    probes=load(F/'tail_anomaly_review.json')['rows']
    for metric in config['tail_metrics']:
        x=max((r for r in probes if r['metric']==metric),key=lambda r:r['value'])
        cohorts.append((f'tail:{metric}',x,'TAIL'))
    readers={s:HMBReader(E/'hmb_manifest.json',E/'production_plan.json',E0/'candidate_config.json',
                        handoff['ubuntu_products'],s,GEOMETRY) for s in all_splits}
    output=[]
    for key,src,role in cohorts:
        b=parts[src['partition']]
        t=src['tick']
        a,meta,table,frames=readers[b['split']].read(b['key'])
        if meta['ticks']<=t or t<0:
            raise ValueError('Registered witness tick exceeds its approved partition')
        identity=dict(key=key,split=b['split'],root=b['parent'],partition=b['key'],tick=t,cohort=role,
                      tick_ns=int(a['tick_ns'][t]),task=tasks[b['parent']],frames=frames,
                      reason=a['reason_code'][t,:4].tolist(),pose_valid=a['pose_valid'][t,:4].tolist(),
                      safety_permission=bool(a['safety_permission'][t]),joint=bool(a['joint_valid'][t]),
                      archive_identity=meta['identity'],clock_quality=a['clock_quality'][t,:4].tolist())
        good=identity['joint'] and all(identity['pose_valid'])
        if not good:
            output.append((identity,None))
            continue
        topology=[]
        for i,s in enumerate(('manus_left','manus_right')):
            version=meta['dictionary']['topology_id'][int(a['topology_id'][t,i])]
            topology.append(next(r for r in meta['topologies'] if (r['stream'],r['version'])==(s,version)))
        source=[table[int(i)] for i in a['source_index'][t,:4,0]]
        identity['sources']=source
        proxy=np.linalg.norm(a['velocity_m_s'][t,:4],axis=-1)
        speed=float(np.nansum(np.nanmax(np.where(np.isfinite(proxy),proxy,0),axis=1)))
        if not a['velocity_valid'][t,:4].all():
            speed=1e9
        row=dict(key=key,split=b['split'],wrist=a['pose'][t,2:4,0],sdk_common=a['sdk_common_pose'][t,:2],
                 palm=a['pose'][t,:2],topology=topology,proxy_motion=speed)
        output.append((identity,row))
    if sum(i['cohort']=='ROOT' for i,_ in output)!=32:
        raise ValueError('Root denominator differs')
    return output


def sdk_one(lib,common):
    q=[]
    for side,points,deg in zip(('left','right'),common,(15.,-15.),strict=True):
        angle=np.deg2rad(deg)
        c,s=np.cos(angle),np.sin(angle)
        centered=np.asarray(points)@np.array([[c,s,0],[-s,c,0],[0,0,1]])
        with WujiRetargetSession(lib,side=side) as sdk:
            q.extend(np.asarray(sdk.step(centered),dtype=np.float64))
    return np.asarray(q)


def independent_q(robot,q,target,config):
    proof=robot.legality(q)
    g=robot.geometry(q)
    tree=independent_frames(robot,q)
    tips=np.asarray([tree['site',names[-1]][0] for group in robot.landmark_names for names in group]).reshape(2,5,3)
    residual=float(np.linalg.norm(tips-g.tips,axis=-1).max())
    exact_range=float(max(0,np.max(robot.bounds[:,0]-q),np.max(q-robot.bounds[:,1])))
    coupled=max(1.025*abs(q[5])+abs(q[6]),1.025*abs(q[12])+abs(q[13]))
    if (not proof['feasible'] or residual>config['tolerance']['fk_m'] or exact_range>1e-9 or coupled>COUPLING_LIMIT+1e-9):
        raise ValueError('Static q witness does not satisfy independent checks')
    return dict(reconstructed_tip_max_m=residual,coupling_absolute_envelope_rad=float(coupled),
                joint_range_violation_rad=exact_range,collision=proof,
                projection_max_m=float(np.linalg.norm(tips-target.world[:,:,-1],axis=-1).max()))


def scoped(rows,config):
    by=defaultdict(list)
    for row in rows:
        label=row['identity']
        for group in ('All',label['split'],label['cohort'],f"task:{label['task']}",
                      f"split_task:{label['split']}:{label['task']}"):
            by[group].append(row)
    result={}
    for name,items in sorted(by.items()):
        modes={}
        for mode in ('sdk_wrist_ik','arms14','whole54'):
            data=[x[mode]['metrics'] for x in items if mode in x and 'metrics' in x[mode]]
            vals=[d['max_tip_m'] for d in data]
            pct=[float(x) for x in np.percentile(vals,[50,95,99])] if vals else None
            modes[mode]=dict(scored=len(vals),planned=len(items),max_tip_p50_p95_p99_m=pct,
                 all_tip_hit={str(h):dict(n=sum(v<=h for v in vals),denominator=len(items),
                                          rate=sum(v<=h for v in vals)/len(items))
                              for h in config['tolerance']['hit_thresholds_m']},
                 status_counts=Counter(x.get(mode,{}).get('status','UNKNOWN') for x in items),
                 local_rmse_m_median=float(np.median([d['local_landmark_rmse_m'] for d in data])) if data else None)
        result[name]=modes
    return result


def show_vis(out,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Only illustrate a pre-registered sample with an actual feasible witness.
    row=next((x for x in cases if 'mapped' in x and 'feasible' in x and
              any(y['near_pinch_pairs'] for y in x['mapping']['hands'])),None)
    if row is None:
        row=next(x for x in cases if 'mapped' in x and 'feasible' in x)
    fig=plt.figure(figsize=(11,8))
    axes=[fig.add_subplot(221+i,projection='3d') for i in range(4)]
    for i,kind in enumerate(('mapped','feasible')):
        for j,s in enumerate(('left','right')):
            ax=axes[2*i+j]
            points=np.array(row[kind]['local'])[j]
            for n,line in enumerate(points):
                ax.plot(*line.T,marker='o',label=('thumb','index','middle','ring','pinky')[n],markersize=3)
            ax.set(xlabel='m',ylabel='m',zlabel='m',title=f'{kind} · {s} palm')
    axes[1].legend(loc='upper left',fontsize=8)
    fig.suptitle(f"{row['identity']['partition']} @ tick {row['identity']['tick']}\nHistorical HMB · simulated geometry")
    fig.subplots_adjust(top=.85,bottom=.06,left=.06,right=.96,wspace=.21,hspace=.25)
    fig.savefig(out/'intent_pinch_projection.png',dpi=145)
    plt.close(fig)


def main(plan_path,out,model_results):
    config=load(plan_path)
    handoff=verify_approval(config)
    audit=load(model_results)
    if not audit['model']['valid']:
        raise ValueError('P02-A has not passed the current model.')
    robot=RobotModel(config['model'],config['clearance_m'])
    independent=RobotModel(config['model'],config['clearance_m'])
    selected=samples(config,handoff)
    mapper=Mapper(robot,[s for ident,s in selected if s is not None and ident['split']=='Train'
                         and ident['cohort']=='ROOT'],config)
    write(out/'mapping_contract.json',mapper.contract()|{'sdk_path':config['sdk'],'sdk_sha256':digest(config['sdk']),
          'arm_baseline':'P02 bounded static wrist IK with unmodified Wuji offline retarget. S0-4 archived original IK is evaluated separately.',
          'input_handoff_sha256':config['handoff_sha256']})
    rows=[]
    rng=np.random.default_rng(config['seed'])
    for i,(ident,row) in enumerate(selected,1):
        evidence=dict(identity=ident)
        rows.append(evidence)
        if row is None:
            evidence['disposition']='UNKNOWN_INVALID_GEOMETRIC_COHORT_MEMBER'
            print(i,ident['key'],evidence['disposition'],flush=True)
            continue
        try:
            target,common,details=mapper.map(row)
        except (KeyError,ValueError) as exc:
            evidence.update(disposition='UNKNOWN_MAPPING_FAILED',error=str(exc))
            continue
        evidence.update(mapped=target.json(),mapping=details)
        try:
            hands=sdk_one(config['sdk'],common)
            baseline=wrist_seed(robot,target,np.r_[robot.home[:14],hands])
            evidence['sdk_wrist_ik']=dict(status='RECORDED_STATIC_SDK_WRIST_IK',metrics=metrics(target,robot,baseline,config['cross_hand_pairs']),
                        legality=robot.legality(baseline),q=baseline.tolist(),sdk_unclipped_q=hands.tolist(),
                        clipping_delta_rad=(baseline[14:]-hands).tolist())
        except (RuntimeError,OSError,ValueError) as exc:
            baseline=wrist_seed(robot,target,robot.home)
            evidence['sdk_wrist_ik']=dict(status='UNKNOWN_SDK',error=str(exc))
            evidence['arms14']=dict(status='UNKNOWN_SDK')
        quasi=joint_seed(Objective(robot,target,config),baseline)
        displaced=np.clip(robot.home+rng.normal(0,.045,54),*robot.bounds.T)
        if not robot.legality(displaced)['feasible']:
            displaced=robot.home.copy()
        seeds=[('SDK_AND_WRIST_IK',baseline),('JOINT_RESIDUAL_SEED',quasi),('PERTURBED_HOME',displaced)]
        if 'metrics' in evidence['sdk_wrist_ik']:
            fixed=[(label,np.r_[q[:14],baseline[14:]]) for label,q in seeds]
            evidence['arms14']=solve(robot,target,config,fixed,indices=np.arange(14))
        result=solve(robot,target,config,seeds)
        evidence['whole54']=result
        if result['q'] is not None:
            q=np.asarray(result['q'])
            evidence['independent_q']=independent_q(independent,q,target,config)
            evidence['feasible']=robot_target(robot,robot.geometry(q),config['cross_hand_pairs']).json()
        print(i,ident['key'],result['status'],round(result.get('metrics',{}).get('max_tip_m',0)*1000,3),flush=True)
    write(out/'study_records.json',rows)
    write(out/'summary.json',scoped(rows,config))
    assigned=[r for r in rows if 'mapped' in r]
    feasible=[r for r in rows if 'feasible' in r]
    if not feasible:
        raise ValueError('No validated witness to export')
    def array_bundle(items,attr):
        return {'case_key':np.asarray([r['identity']['key'] for r in items]),
                **{k:np.stack([r[attr][k] for r in items]) for k in items[0][attr]}}
    np.savez_compressed(out/'mapped_intent.npz',**array_bundle(assigned,'mapped'))
    np.savez_compressed(out/'feasible_scoring_target.npz',**array_bundle(feasible,'feasible'))
    np.savez_compressed(out/'q_oracle_offline.npz',case_key=np.asarray([r['identity']['key'] for r in feasible]),
                        q_rad=np.stack([r['whole54']['q'] for r in feasible]),joint_names=np.asarray(robot.names))
    np.savez_compressed(out/'projection_error.npz',case_key=np.asarray([r['identity']['key'] for r in feasible]),
        **{name:np.stack([np.asarray(r['feasible'][name])-np.asarray(r['mapped'][name]) for r in feasible])
           for name in ('world','local','wrist','pinch','cross','relative_wrist')})
    write(out/'view_manifest.json',dict(approved_handoff_sha256=config['handoff_sha256'],model_sha256=config['model_sha256'],
          time_scope='HISTORICAL_SIMULATED_CAUSAL_GEOMETRY_ONLY',planned=len(rows),mapped=len(assigned),feasible=len(feasible),
          primary_pose_contract='per root frozen representative plus preserved diagnostic tails',
          q_actor_and_online_nominal_access='FORBIDDEN',
          files={name:digest(out/name) for name in ('mapped_intent.npz','feasible_scoring_target.npz',
              'projection_error.npz','q_oracle_offline.npz','study_records.json','mapping_contract.json','summary.json')}))
    show_vis(out,rows)


if __name__=='__main__':
    main(*(Path(x) for x in sys.argv[1:]))
