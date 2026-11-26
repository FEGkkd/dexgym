"""Seal the independent review run while keeping archived P01/P02 bytes immutable."""
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import mujoco
import numpy
import scipy

from dexgym.tasks.direct.wholebody_s05.p02.model import asset_files,digest
from run_study import E,F,load,write
from workspace import DEF,ORIGINAL,source


def git(*args):
    return subprocess.check_output(['git',*args])


def main(run,geometry,out):
    base,plan,_,_=source()
    model_assets=asset_files(base['model'])
    if model_assets!=load(DEF/'model_report.json')['model']['contract']['assets']:
        raise ValueError('Compiled model XML, include or asset fingerprint differs')
    ref=load(DEF/'output_sha256.json')
    for name,sha in ref.items():
        if digest(ORIGINAL/name)!=sha:
            raise ValueError(f'Original P02 fingerprint differs: {name}')
    p01_locked=(
        'docs/s0-5/evidence/P01_E_E0','docs/s0-5/evidence/P01_E_E1',
        'docs/s0-5/evidence/P01_F','source/dexgym/tasks/direct/wholebody_s05/data')
    git('diff','--exit-code',plan['reviewed_commit'],'--',*p01_locked,
        'docs/s0-5/reports/P02_acceptance.md','docs/s0-5/evidence/P02/plan.json',
        'docs/s0-5/evidence/P02/mapping_contract.json','docs/s0-5/evidence/P02/review_receipt.json',
        'docs/s0-5/evidence/P02/P03_scoring_handoff.json')
    outputs=run/'review-01'
    required={'geometry_train':geometry/'train.exit','geometry_validation':geometry/'validation.exit',
              **{f:run/(f+'.exit') for f in ('budget-01','matched-01','controls-02','supplement-01',
                                            'review_tests','constraint_activity','review-01')}}
    if any(p.read_text().strip()!='0' for p in required.values()):
        raise ValueError('Required process result missing')
    geometry_stats={s:load(geometry/s/'workspace_summary.json') for s in ('train','validation')}
    frozen=geometry/'fitted_candidates.json'
    if any(x['fits_sha256']!=digest(frozen) for x in geometry_stats.values()):
        raise ValueError('Candidate SHA diverged from geometry output')
    out.mkdir(mode=0o700)
    write(out/'model_assets_sha256.json',model_assets)
    knobs={
        'train_workspace_summary.json':geometry/'train/workspace_summary.json',
        'validation_workspace_summary.json':geometry/'validation/workspace_summary.json',
        'train_functional_summary.json':geometry/'train/functional_summary.json',
        'validation_functional_summary.json':geometry/'validation/functional_summary.json',
        'fitted_candidates.json':frozen,
        'pinch_statistics_correction.json':run/'budget-01/pinch_statistics_correction.json',
        'raw_tracker_step_overlap.json':run/'supplement-01/raw_tracker_step_overlap.json',
        'quantitative_summary.json':outputs/'quantitative_summary.json',
        'scoring_freeze_candidate.json':outputs/'scoring_freeze_candidate.json',
        'workspace_bounds.png':outputs/'workspace_bounds.png',
        'oracle_metrics.png':outputs/'oracle_metrics.png',
        'pinch_transition.png':outputs/'pinch_transition.png',
        'review_tests.txt':run/'review_tests.stderr',
        'semantic_red.txt':run/'semantic_red.stderr'}
    for name,path in knobs.items():
        shutil.copyfile(path,out/name)
    constraints=load(run/'constraint_activity.json')
    reduced={k:v for k,v in constraints.items() if k not in ('witnesses','rows','cases','per_witness')}
    write(out/'constraint_activity_summary.json',reduced)
    functional=load(run/'functional-01/functional_fk_checks.json')
    p=[]
    for t in functional['paired_continuity_fk']:
        result=t['solver']
        p.append(dict(identity=t['identity'],method=t['method'],source_vectors_m=t['raw_pinch_m'],
                      mapped_vectors_m=t['mapped_pinch_m'],status=result['status'],metrics=result.get('metrics')))
    write(out/'functional_fk_summary.json',dict(historical_reconstruction=functional['historical_reconstruction'],
               paired_continuity_fk=p,archive_sha256=digest(run/'functional-01/functional_fk_checks.json')))
    controls=load(run/'controls-02/minimax_controls.json')
    write(out/'minimax_controls_summary.json',[
        {k:v for k,v in t.items() if k!='result'}|{'max_tip_m':t['result']['epigraph_m'],
                                               'status':t['result']['status'],
                                               'attempts':[a['termination'] for a in t['result']['attempts']]}
        for t in controls])
    mounted=load(run/'supplement-01/mount_ablation.json')
    write(out/'mount_ablation_summary.json',dict(
        fit=load(run/'supplement-01/mount_diagnostic_train_freeze.json'),
        config_sha256=digest(run/'supplement-01/mount_diagnostic_train_freeze.json'),
        tests=[dict(identity=x['identity'],modes={n:dict(status=v['fk']['status'],metrics=v['fk'].get('metrics'),
                                                         strict_bound_m=v['lower_m'],seed_s=v['seed_s'])
                                                for n,v in x['modes'].items()}) for x in mounted]))
    paths=[p for p in sorted(run.rglob('*')) if p.is_file() and not p.is_relative_to(out)]
    write(out/'archive_sha256.json',{str(p.relative_to(run)):digest(p) for p in paths})
    track=[n for n in git('ls-files','-z','docs/s0-5/evidence/P02/refinement','docs/s0-5/evidence/P02/refinement_plan.json',
                           'docs/s0-5/evidence/P02/seal_p02.py','tests/s0_5/unit/test_p02_review.py',
                           'tests/s0_5/unit/test_p02_refinement_geometry.py',
                           'source/dexgym/tasks/direct/wholebody_s05/p02').decode().split('\0') if n]
    write(out/'source_sha256.json',{n:digest(n) for n in track})
    failed={str(p.relative_to(run)):int(p.read_text()) for p in run.rglob('*.exit') if p.read_text().strip()!='0'}
    res={str(p.relative_to(run)):p.read_text() for p in run.rglob('*.resources')}
    write(out/'execution_receipt.json',dict(schema='s05.p02.independent_refinement.receipt.v1',
        run=plan['run_id'],archive=str(run),executed_head=git('rev-parse','HEAD').decode().strip(),
        reviewed_baseline=plan['reviewed_commit'],baseline_p02_original_artifacts_rehashed=len(ref),
        plan_sha256=digest(DEF/'refinement_plan.json'),model_sha256=base['model_sha256'],
        train_frozen_candidates_sha256=digest(frozen),model_audit_sha256=digest(DEF/'model_report.json'),
        historical_hmb_sha256=base['e1_manifest_sha256'],f1_handoff_sha256=base['handoff_sha256'],
        phases={'P01':'ACCEPTED','P02':'READY_FOR_REVIEW','P03':'NOT_STARTED'},
        engineering_blocks=dict.fromkeys(('C2-RT','robot-base','P01-T','Final Test'),'BLOCKED'),
        approval='DCR-003 research scope only; formal target and phase acceptance pending',
        scope='HISTORICAL_HMB_OFFLINE_STATIC_RESEARCH',authorized_p03=False,
        geometry_count=sum(t['counts']['qualified'] for t in geometry_stats.values()),
        exits={k:int(v.read_text()) for k,v in required.items()}, failed_attempts=failed,resources=res,
        runtime=dict(python=sys.executable,version=platform.python_version(),
                    package_versions={m.__name__:m.__version__ for m in (mujoco,numpy,scipy)},
                    import_paths={m.__name__:m.__file__ for m in (mujoco,numpy,scipy)}),
        output_sha256={f.name:digest(f) for f in out.iterdir() if f.is_file()}))
    print('PASS refinement receipt',digest(out/'execution_receipt.json'),len(paths),'run artifacts',len(ref),'original files')


if __name__=='__main__':
    main(*(Path(x) for x in sys.argv[1:]))
