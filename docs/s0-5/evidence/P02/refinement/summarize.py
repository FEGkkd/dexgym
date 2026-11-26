"""Package original-target and candidate-target metrics and reproducible figures."""
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.intent import robot_target
from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel,digest
from dexgym.tasks.direct.wholebody_s05.p02.oracle import metrics
from evaluate import recorded_target
from run_study import E,load,write
from workspace import DEF,ORIGINAL,SPLITS,norm,source,stats


SCOPES=('All','ROOT','TAIL','Train','Validation','Train_ROOT','Validation_ROOT')
METRICS=('max_tip_m','mean_tip_m','tip_m','wrist_translation_m','wrist_rotation_rad','local_landmark_rmse_m',
         'pinch_vector_m','pinch_distance_m','cross_hand_m','relative_wrist_translation_m','relative_wrist_rotation_rad')
THRESHOLDS=(.003,.005,.01,.015)


def scope(rows,label):
    return [r for r in rows if label in ('All',r['identity']['cohort'],r['identity']['split'],
                                r['identity']['split']+'_'+r['identity']['cohort'])]


def pct(values):
    return np.percentile(values,[50,95,99,100],axis=0).tolist() if len(values) else None


def measurement(rows,name):
    actual=[r for r in rows if r and 'metrics' in r]
    vals=np.array([x['metrics']['max_tip_m'] for x in actual])
    attempts=[x for r in rows if r for x in r.get('attempts',[])]
    dims={k:pct(np.asarray([r['metrics'][k] for r in actual]).ravel()) for k in METRICS}
    groups=Counter(r['status'] if r else 'UNKNOWN' for r in rows)
    return dict(scored=len(actual),planned=len(rows),distribution=dims,
                by_finger_P50_P95_P99_max_m=pct(np.array([r['metrics']['tip_m'] for r in actual])),
                hit_counts={str(int(t*1000)):int((vals<=t).sum()) for t in THRESHOLDS},
                statuses=groups, attempts=dict(n=len(attempts),converged=sum(a['solver_success'] for a in attempts),
                      exhausted=sum(a['budget_exhausted'] for a in attempts),
                      feasible=sum(a['feasible_best'] for a in attempts),
                      duration_s=pct([a['elapsed_s'] for a in attempts]),
                      iterations=pct([a['iterations'] for a in attempts]),
                      termination=Counter(a['termination'] for a in attempts)),mode=name)


def budget_stats(rows):
    out={}
    for label in SCOPES:
        planned=scope(rows,label)
        out[label]={str(it):measurement([r.get('budgets',{}).get(str(it)) for r in planned],str(it))
                    for it in (90,300,1000)}
    out['original_90_replay_absolute_delta_m']=pct([abs(r['original_90_tip_m']-r['budgets']['90']['metrics']['max_tip_m'])
                                                   for r in rows if 'budgets' in r])
    return out


def compare(out,model,base,rows,budget,fits):
    if [r['identity'] for r in rows]!=[r['identity'] for r in budget]:
        raise ValueError('Exact original source identity pairing failed')
    configs=fits['configs']
    results={k:{} for k in configs}
    by_case=[]
    for r in rows:
        identity=r['identity']
        row=dict(identity=identity,status=r['original_status'],variants={})
        by_case.append(row)
        if 'candidates' not in r:
            continue
        orig=recorded_target(r['candidates']['M0']['mapped'])
        by_case_rotation=np.asarray(configs['M0']['rotation'])
        for k,v in r['candidates'].items():
            target=recorded_target(v['mapped'])
            rback=by_case_rotation@np.asarray(configs[k]['rotation']).T
            distortion=norm(target.cross@rback.T-orig.cross)
            wrist_span_delta=abs(v['wrist_span_mapped_m']-v['wrist_span_raw_m'])
            row['variants'][k]=dict(radial_m=v['strict_radial_bound_m'],
                    original_mapped_max_tip_change_m=v['max_tip_change_from_M0_m'],
                    raw_span_m=v['wrist_span_raw_m'],mapped_span_m=v['wrist_span_mapped_m'],
                    wrist_span_delta_m=wrist_span_delta,world_cross_vector_delta_from_M0_m=v['cross_delta_from_M0_m'],
                    common_orientation_cross_vector_shape_delta_m=distortion.tolist(),
                    local_delta_m=float(norm(target.local-orig.local).max()),
                    pinch_delta_m=float(norm(target.pinch-orig.pinch).max()),
                    relative_wrist_delta_m=float(norm(target.relative_wrist-orig.relative_wrist)),
                    relative_orientation_max_component_delta=float(np.abs(target.relative_rotation-orig.relative_rotation).max()),
                    seed_record_s=v['seed_generation_s'],seed_timing_type='LOAD_FROM_SIGNED_ARCHIVE' if k=='M0' else 'GENERATE',
                    methods={})
            for m in ('multitask','minimax'):
                s=v[m]
                if s['q'] is None:
                    continue
                data=metrics(orig,model,np.asarray(s['q']),base['cross_hand_pairs'])
                row['variants'][k]['methods'][m]=dict(input_objective=s['metrics'],against_immutable_M0=data)
            if k in ('M0','M1') and max(wrist_span_delta,float(distortion.max()))>2e-9:
                raise ValueError('Shared rigid candidate did not preserve relative geometry')
    for k in configs:
        for label in SCOPES:
            traj=scope(rows,label)
            cmp=scope(by_case,label)
            values=[x['variants'][k] for x in cmp if k in x['variants']]
            s=dict(radial_lower_bound_m=pct([x['radial_m'] for x in values]),unknown=len(cmp)-len(values),
                   strict_excluded_counts={str(int(t*1000)):sum(x['radial_m']>t for x in values) for t in THRESHOLDS},
                   landmarks_tip_max_distortion_from_M0_m=pct([x['original_mapped_max_tip_change_m'] for x in values]),
                   raw_wrist_span_m=pct([x['raw_span_m'] for x in values]),
                   mapped_wrist_span_m=pct([x['mapped_span_m'] for x in values]),
                   wrist_distance_change_m=pct([x['wrist_span_delta_m'] for x in values]),
                   common_rotation_cross_shape_distortion_m=pct(np.ravel([x['common_orientation_cross_vector_shape_delta_m'] for x in values])),
                   max_local_change_m=max((x['local_delta_m'] for x in values),default=0.),
                   max_pinch_change_m=max((x['pinch_delta_m'] for x in values),default=0.),
                   seed_cost_type='LOAD_FROM_SIGNED_ARCHIVE' if k=='M0' else 'GENERATE',
                   seed_cost_s=pct([x['seed_record_s'] for x in values]))
            for m in ('multitask','minimax'):
                best=[x.get('candidates',{}).get(k,{}).get(m) for x in traj]
                s[m]=measurement(best,m)
                original=[dict(status='LEGAL_STATIC_WITNESS',metrics=x['variants'][k]['methods'][m]['against_immutable_M0'])
                          if k in x['variants'] and m in x['variants'][k]['methods'] else None for x in cmp]
                s[m+'_vs_M0']=measurement(original,m+'_vs_M0')
            results[k][label]=s
    write(out/'paired_cases_against_M0.json',by_case)
    return results


def export_candidate_views(out,base,model,rows,fits):
    records=[r for r in rows if 'candidates' in r]
    report={}
    for k in fits['configs']:
        mapped=[]; feasible=[]; q=[]; keys=[]
        for row in records:
            target=row['candidates'][k]
            if target['multitask']['q'] is None:
                continue
            keys.append(row['identity']['key'])
            mapped.append(target['mapped'])
            q_rad=np.asarray(target['multitask']['q'])
            q.append(q_rad)
            feasible.append(robot_target(model,model.geometry(q_rad),base['cross_hand_pairs']).json())
        if not q:
            raise ValueError('Candidate export has no legal targets')
        def arrays(data):
            return {name:np.array([x[name] for x in data]) for name in data[0]}
        m,f=arrays(mapped),arrays(feasible)
        name=k.lower()
        filenames=[name+'_'+p+'.npz' for p in ('mapped_intent','feasible_target','oracle_q','projection')]
        np.savez_compressed(out/filenames[0],case_key=keys,**m)
        np.savez_compressed(out/filenames[1],case_key=keys,**f)
        np.savez_compressed(out/filenames[2],case_key=keys,q_rad=q,joint_names=model.names)
        np.savez_compressed(out/filenames[3],case_key=keys,**{
            n:f[n]-m[n] for n in ('wrist','world','local','pinch','cross','relative_wrist')})
        report[k]=dict(count=len(keys),files={file:digest(out/file) for file in filenames})
    return report


def figure_workspace(geometry,path):
    fig,ax=plt.subplots(1,2,figsize=(10.4,4.5),sharey=True,layout='constrained')
    color=['#607487','#087a85','#a35518']
    rates=['3','5','10','15']
    for j,(sp,wr) in enumerate(geometry.items()):
        for i,(variant,data) in enumerate(wr['candidates'].items()):
            g=data['all']; y=[100*g['strict_above'][cut]['n']/g['scored'] for cut in rates]
            ax[j].bar(np.arange(4)+(i-1)*.24,y,width=.23,color=color[i],label=variant)
        ax[j].set(xticks=np.arange(4),xticklabels=rates,xlabel='Strict lower-bound threshold (mm)',
                  title=f"{sp} · {wr['counts']['qualified']:,} qualified ticks")
        ax[j].grid(axis='y',alpha=.25); ax[j].set_axisbelow(True)
    ax[0].set_ylabel('Qualified ticks above threshold (%)')
    ax[1].legend(frameon=False)
    fig.suptitle('Workspace registration changes the geometric rejection rate',fontsize=15)
    fig.savefig(path,dpi=165)
    plt.close(fig)


def figure_oracle(results,path):
    fig,axs=plt.subplots(1,3,figsize=(12,4.4),sharey=True,layout='constrained')
    for j,(k,s) in enumerate(results.items()):
        for row,label,color in [('multitask','Multi-task','#596678'),('minimax','Minimax tip','#087a85'),
                                ('minimax_vs_M0','Minimax scored vs original M0','#a35518')]:
            x=[s[g][row]['distribution']['max_tip_m'][1]*1000 for g in ('Train_ROOT','Validation_ROOT')]
            at=('multitask','minimax','minimax_vs_M0').index(row)
            axs[j].bar(np.arange(2)+(at-1)*.25,x,width=.24,color=color,label=label)
        axs[j].set(xticks=(0,1),xticklabels=('Train','Validation'),title=k)
        axs[j].grid(axis='y',alpha=.2); axs[j].set_axisbelow(True)
    axs[0].set_ylabel('Ten-tip maximum error P95 (mm)')
    axs[0].legend(loc='upper right',fontsize=7,frameon=False)
    fig.suptitle('An easier mapped target and improved tracking are different measurements',fontsize=14)
    fig.savefig(path,dpi=165)
    plt.close(fig)


def figure_pinch(geo,output):
    split='Validation'
    root=geo/split.lower()
    parts=[p for p in load(E/'production_plan.json')['partitions'] if p['split']==split]
    witness=next(w for w in load(root/'functional_summary.json')['fk_witnesses'] if w['direction']=='entry')
    partition=next(i for i,p in enumerate(parts) if p['key']==witness['partition'])
    side=0 if witness['side']=='left' else 1
    with np.load(root/'functional_traces.npz',allow_pickle=False) as a:
        t=np.flatnonzero((a['tick']==witness['tick'])&(a['part']==partition)).item()
        bounds=np.flatnonzero((a['part']==partition)&(np.abs(a['tick']-witness['tick'])<=12))
        show=np.arange(bounds[0],bounds[-1]+1)
        finger=np.flatnonzero((a['raw_pinch_m'][t,side]<=.020)&(a['raw_pinch_m'][t-1,side]>.020))[0]
        raw=a['raw_pinch_m'][show,side,finger]
        time=(a['time_ns'][show]-a['time_ns'][t])*.000000001
        scales={}
        for name,var in [('Hard 20 mm','hard_tip'),('Smooth 16–24 mm','soft_tip')]:
            trace=a[var][show,side,finger+1]
            step=np.r_[np.nan,1000*norm(np.diff(trace,axis=0))]
            step[~a['consecutive'][show]]=np.nan
            scales[name]=step
    fig,ax=plt.subplots(2,1,figsize=(8.4,5.5),sharex=True,layout='constrained')
    ax[0].plot(time,raw*1000,color='#607487',marker='o',markersize=3)
    ax[0].axhline(20,color='#a35518',linestyle='--')
    ax[0].set(ylabel='Raw pinch distance (mm)')
    for color,(label,y) in zip(('#a35518','#087a85'),scales.items(),strict=True):
        ax[1].plot(time,y,marker='.',label=label,color=color)
    ax[1].legend(frameon=False)
    ax[1].set(xlabel='Historical HMB time from threshold crossing (s)',ylabel='Mapped finger step (mm)')
    for a in ax: a.grid(alpha=.25)
    fig.suptitle(f"Validation {witness['side']} pinch entry · {witness['partition']} · tick {witness['tick']}",fontsize=10)
    fig.savefig(output,dpi=165)
    plt.close(fig)


def main(run,geo,out):
    base,plan,_,_=source()
    original=load(DEF/'review_receipt.json')
    tracked={DEF/'output_sha256.json':original['receipts']['artifact_manifest'],
             DEF/'mapping_contract.json':load(DEF/'output_sha256.json')['intent-02/mapping_contract.json'],
             DEF/'plan.json':original['plan_sha256']}
    for f,h in tracked.items():
        if digest(f)!=h:
            raise ValueError(f'Tracked refereed file differs: {f}')
    checks=[geo/'train.exit',geo/'validation.exit',*[run/(x+'.exit') for x in (
        'budget-01','matched-01','controls-02','supplement-01','review_tests','constraint_activity')]]
    if any(path.read_text().strip()!='0' for path in checks):
        raise ValueError('Required experiment did not exit successfully')
    matched=load(run/'matched-01/matched_candidates.json')
    budget=load(run/'budget-01/budget_results.json')
    geometry={s:load(geo/s.lower()/'workspace_summary.json') for s in SPLITS}
    frozen=geo/'fitted_candidates.json'
    sha=digest(frozen)
    if matched['candidates_sha256']!=sha or any(g['fits_sha256']!=sha for g in geometry.values()):
        raise ValueError('Candidate SHA diverged')
    fits=load(frozen)
    signed=load(E/'full_coverage.json')
    for s,g in geometry.items():
        if g['signed_joint_valid']!=signed[s]['counts']['causal.joint_valid'] or g['counts']['planned']!=signed[s]['counts']['planned_ticks']:
            raise ValueError('Signed HMB counts disagree')
    root=RobotModel(base['model'],base['clearance_m'])
    out.mkdir(mode=0o700)
    cmp=compare(out,root,base,matched['rows'],budget,fits)
    b=budget_stats(budget)
    views=export_candidate_views(out,base,root,matched['rows'],fits)
    write(out/'quantitative_summary.json',dict(note='P50/P95/P99/max where four percentiles; full-planned hit denominator',
           matched=cmp,budgets=b))
    write(out/'scoring_freeze_candidate.json',dict(
        requested_use='HISTORICAL_STATIC_RESEARCH_ONLY',formal_scoring_freeze='PENDING_HUMAN_APPROVAL',
        reviewed_original_target_retained=True,fit_sha256=sha,model_sha256=base['model_sha256'],views=views,
        propose='Keep M0 as a required audit column. Review M1 as a second, rigid simulation-development target. M2 and minimax remain diagnostics; keep all unknown and excluded cases in planned denominators.',
        separate_projection_and_execution=True,unknown_masks='Matched paired-case dispositions retained',
        forbid_online_actor_or_nominal_from_loading_oracle_q=True,p03_authorized=False,
        p03_independent_prerequisites=['P02','P01-T','C2-RT','robot-base']))
    figure_workspace(geometry,out/'workspace_bounds.png')
    figure_oracle(cmp,out/'oracle_metrics.png')
    figure_pinch(geo,out/'pinch_transition.png')
    print('Independent P02 refinement archive and output figures built',flush=True)


if __name__=='__main__':
    main(*(Path(x) for x in sys.argv[1:]))
