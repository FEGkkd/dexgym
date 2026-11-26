"""Seal the authorized P02 closure and its three independent work packages."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


DOCS = Path('docs/s0-5')
P02 = DOCS/'evidence/P02'
HERE = P02/'final_handoff'


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def write(path,data):
    with path.open('x',encoding='utf-8') as p:
        json.dump(data,p,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        p.write('\n')


def require(p,why):
    if not p:
        raise ValueError(why)


def protected_archive(root,table):
    for name,h in table.items():
        require(digest(root/name)==h,f'Immutable historical artifact differs: {root/name}')
    return len(table)


def seal(run,previous_original,previous_refinement,out):
    plan=read(HERE/'work_plan.json')
    review=plan['reviewed_baseline']
    phase=read(DOCS/'phase_state.json')['current']
    require(all(phase[p]==v for p,v in plan['phase_status'].items()),'P01/P02/P03 phase scope changed')
    require(not any(plan['execution_authorizations'].values()),'Scope contains real operation or P03 authorization')
    staged_frozen=(run/'frozen_work_plan.sha256').read_text().split()[0]
    require(digest(HERE/'work_plan.json')==staged_frozen,'Final handoff plan differs from preregistration')
    for name,sha in plan['approved_inputs_sha256'].items():
        require(digest(P02/'refinement'/name)==sha,f'Reviewed input changed: {name}')
    original_receipt=read(P02/'review_receipt.json')
    refined_receipt=read(P02/'refinement/execution_receipt.json')
    original_index=P02/'output_sha256.json'
    refined_index=P02/'refinement/archive_sha256.json'
    require(digest(original_index)==original_receipt['receipts']['artifact_manifest'],'Original index SHA error')
    require(digest(refined_index)==refined_receipt['output_sha256']['archive_sha256.json'],'Refinement index SHA error')
    protected={
        'P02-20261126-54c2e06':protected_archive(previous_original,read(original_index)),
        'P02-REFINE-20261126-8753921':protected_archive(previous_refinement,read(refined_index))}
    subprocess.run(['git','diff','--exit-code',review,'--',
            str(P02/'refinement'),str(P02/'review_receipt.json'),str(P02/'P03_scoring_handoff.json'),
            str(P02/'output_sha256.json'),str(P02/'plan.json'),str(P02/'mapping_contract.json'),
            'docs/s0-5/reports/P02_acceptance.md','docs/s0-5/reports/P02_review_refinement.md',
            str(DOCS/'evidence/P01_E_E0'),str(DOCS/'evidence/P01_E_E1'),str(DOCS/'evidence/P01_F'),
            'source/dexgym/tasks/direct/wholebody_s05/data','source/dexgym/assets/marvin_wuji'],check=True)
    ap=run/'m1-01/m1_pairing.json'
    bp=run/'audit-01/independent_audit.json'
    a,b=read(ap),read(bp)
    c=read(HERE/'hardware_evidence.json')
    require(c['current_status']==plan['hardware'] and c['hardware_operation_authorized'] is False,
            'Hardware development changed its blocked gates or operation scope')
    require(c['source_snapshot']['provenance_sha256']==digest(run/'teleop_code_provenance.json'),
            'Agent C audit does not bind the working-tree source snapshot')
    test=(run/'m1-01/tests.stderr').read_text(encoding='utf-8')
    exits={path:int((run/path).read_text()) for path in ('m1-01/tests.exit','m1-01/exit','audit-01/exit')}
    require(set(exits.values())=={0} and 'Ran 20 tests' in test and test.rstrip().endswith('OK'),'Test or study process failed')
    require(a['schema']=='s05.p02.m1_pairing.v1' and len(a['rows'])==38 and not a['formal_score_authorized'],
            'Matched result does not preserve the approved case count or schedule')
    require(b['status']=='PASS_STATIC_RESEARCH_ONLY' and not b['authorized_p03'] \
            and b['agent_a_result_sha256']==digest(ap) and b['pairedscored']==a['summary']['All']['strictly_paired'],
            'Independent B audit did not bind A matched experiment')
    require(digest(HERE/'m1_pairing.py')==a['source_sha256']['pairing_script']
            and a['source_sha256']['final_handoff_work_plan']==staged_frozen,'Agent A actual source is not staged')
    out.mkdir(mode=0o700)
    write(out/'m1_pairing_summary.json',{k:v for k,v in a.items() if k!='rows'}|{
          'raw_result_sha256':digest(ap),
          'unknown_and_no_solution_rows':[dict(key=r['identity']['key'],
                statuses={m:s['status'] for m,s in r['results'].items()})
                for r in a['rows'] if any(x['q'] is None for x in r['results'].values())],
          'guarded_state_evaluations':sum(r.get('constrained_geometry_evaluations',0) for r in a['rows']),
          'admissible_embedded_no_extra_solve':sum(r.get('embedded_14_witness_no_extra_solve',{}).get('legal_in_54',False) for r in a['rows'])})
    write(out/'independent_audit_summary.json',{k:v for k,v in b.items() if k!='rows'}|{
          'raw_audit_sha256':digest(bp)})
    (out/'p02_unit_tests.txt').write_text(''.join(s.rstrip()+'\n' for s in test.splitlines()),encoding='utf-8')
    teleop=read(run/'teleop_code_provenance.json')
    require(len(teleop['files'])==19 and teleop['head']=='30cf13bdc496c43b3016462cfa0a206334d1ff7f',
            'Current read-only Teleop snapshot identity differs')
    write(out/'teleop_code_provenance.json',teleop)
    compact_inputs={
        'm1_pairing_summary.json':digest(out/'m1_pairing_summary.json'),
        'independent_audit_summary.json':digest(out/'independent_audit_summary.json'),
        'hardware_evidence.json':digest(HERE/'hardware_evidence.json')}
    write(out/'limited_scope_candidate.json',dict(
        schema='s05.p02.limited_research_handoff.v1',status='PENDING_HUMAN_LIMITED_SIGNOFF',
        original_scoring_candidate_sha256=plan['approved_inputs_sha256']['scoring_freeze_candidate.json'],
        accepted_research_uses=plan['accepted_research_uses'],case_denominator='32 root + 6 tail; paired winners separately counted',
        retain_M0_intent_and_excluded_targets=True,formal_physical_scoring=plan['formal_physical_scoring'],
        physical_readiness=plan['hardware'],startup_authorized=False,exact_hardware_motion_authorized=False,
        p03_requires=['P02_ACCEPTED','P01-T_ACCEPTED','C2-RT_ACCEPTED','robot-base_ACCEPTED',
                      'APPROVED_FORMAL_SCORING_SCOPE','P03_START_AUTHORIZATION'],
        evidence_sha256=compact_inputs))

    rel=lambda p:str(p.relative_to(DOCS))
    reports={id_:DOCS/'reports'/name for id_,name in {
        'A':'P02_M1_dof_pairing.md','B':'P02_final_independent_audit.md',
        'C':'P02_hardware_prerequisite_readiness.md','Main':'P02_final_handoff.md'}.items()}
    code_a=HERE/'m1_pairing.py'
    test_a=Path('tests/s0_5/unit/test_p02_m1_pairing.py')
    code_b=HERE/'independent_audit.py'
    cplan=DOCS/'specs/HARDWARE_SHARED_MEASUREMENT_PLAN.md'
    extra_c=[p for p in (HERE/'hardware_controls.py',HERE/'offline_hardware_controls.py') if p.is_file()]

    def work(agent,role,scope,reads,writes,basis,cmd,positive,negative,findings,unknown,public,archives,source_publications=()):
        hashes={rel(f):digest(f) if f.is_file() else digest(out/f.name) for f in public}
        hashes.update({str(f.relative_to(run)):digest(f) for f in archives})
        hashes.update({str(f):digest(f) for f in source_publications})
        return dict(agent_role=agent,role=role,task_scope=scope,input_commit=review,
                    read_ownership=reads,write_ownership=writes,engineering_basis=basis,actual_commands=cmd,
                    positive_tests=positive,negative_tests=negative,findings=findings,
                    remaining_uncertainty=unknown,output_paths=list(hashes),output_sha256=hashes)

    pa=work('A_M1_match','Experiment implementation',
            'M1 only, matched frozen fixed SDK arms14 and whole54 study',
            ['reviewed M0 original and M1 matching archives','signed model assets'],
            [str(code_a),str(test_a)],
            'Apply identical arm seed vectors to the same multi-task constrained problem. Rerun 3 starts per mode.',
            [(run/'m1-01/command').read_text().strip(),
             '.venv/bin/python -P -m unittest discover -s tests/s0_5/unit -p test_p02*.py -v'],
            ['Ubuntu total 20 P02 unit tests PASS','A emitted 35 strict matched legal pairs with repeated model checks'],
            ['unit negatives reject swapped identity and changes to an SDK hold. Timeout and unavailable source rows kept'],
            ['fixed-hand guard checks every evaluated 14-mode state', 'no optional 14→54 extra solve'],
            ['static nonconvex witness search and independent practical Servo remain separate'],
            [reports['A'],HERE/'m1_pairing_summary.json',HERE/'p02_unit_tests.txt'],
            [ap,run/'m1-01/stderr',run/'m1-01/resources.txt'],[code_a,test_a])
    pb=work('B_independent_math_score','Independent reviewer',
            'Independent signed-source, collision, per-mode fairness and scoring-boundary audit',
            ['A result and program','signed original, refinement, model and scoring candidate'],
            [str(code_b),str(reports['B'])],
            'Independent FK metrics, exhaustive oracle-pair distance recheck and constructive legal upper bound',
            [(run/'audit-01/command').read_text().strip()],
            ['full independent recheck of 35 strict paired cases and 71 model assets on Ubuntu'],
            ['synthetic translated target positive radial rejection in local audit','scope checks reject formal or P03 authorization'],
            ['PASS_STATIC_RESEARCH_ONLY','H1–H4 control algorithm candidate documented without P03 implementation'],
            ['external physical scoring and future nominal-control empirical feasibility remain pending'],
            [reports['B'],HERE/'independent_audit_summary.json'],
            [bp,run/'audit-01/stdout',run/'audit-01/resources.txt'],[code_b])
    pc=work('C_hardware_plan','Independent hardware evidence design',
            'Physical robot-base registration, C2-RT and P01-T shared acquisition specification',
            ['19 selected actual Teleop working-tree sources','existing read-only design and historical evidence'],
            [str(reports['C']),str(cplan),str(HERE/'hardware_evidence.json'),*map(str,extra_c)],
            'Physical reference, explicit clock error bounds, actual application identity, disjoint system-ID validation',
            [*c['repro']['inspection_commands'],c['repro']['command']],
            ['source snapshot byte checks and pure offline synthetic calibration/joins, see hardware_evidence'],
            ['signed transform and reflection, timestamp and command-identity counterexamples, see hardware_evidence'],
            ['three hardware prerequisites independently BLOCKED','shared capture and site checklists prepared'],
            ['installed runtime, device trace, calibrated tools and site approval still unverified'],
            [reports['C'],cplan,HERE/'hardware_evidence.json',HERE/'teleop_code_provenance.json'],
            [run/'teleop_code_provenance.json'],extra_c)
    pm=work('Main_integrator','Scope, execution and integration',
            'Freeze run and dependencies. Execute A/B on official Ubuntu environment. Preserve prior results and seal candidate.',
            ['A/B/C outputs, approved work plan and prior archive manifests'],
            [str(reports['Main']),str(DOCS/'MULTI_AGENT_PROTOCOL.md'),str(HERE/'work_plan.json'),str(HERE/'seal_final_handoff.py')],
            'Separate approval, technical evidence, commands and evidence provenance',
            ['git rev-parse HEAD origin/s0-5','python3 -B docs/s0-5/evidence/P00/check_documents.py --require-history',
             'python -P docs/s0-5/evidence/P02/final_handoff/seal_final_handoff.py NEW ORIGINAL REFINE OUT'],
            ['protected archive rehashes and A/B provenance cross-bindings'],
            ['committed input hash mismatch or P03 authorization stop the seal'],
            ['submit limited static research candidate and separate hardware execution request'],
            ['P02 remains READY_FOR_REVIEW; no C prerequisite has moved'],
            [reports['Main'],DOCS/'MULTI_AGENT_PROTOCOL.md',HERE/'limited_scope_candidate.json'],
            [run/'preflight.stdout', run/'approved_inputs.sha256'],[HERE/'work_plan.json',HERE/'seal_final_handoff.py'])
    write(out/'agent_work_manifest.json',dict(schema='s05.agent_work_manifest.v1',run=plan['run_id'],
        input_commit=review,implementation_and_review_separated=True,parallel_agents=3,
        main_executed_authoritative_ubuntu_jobs=['A_M1_match','B_independent_math_score'],
        work_packages=[pa,pb,pc,pm]))
    archives={str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*'))
              if p.is_file() and not p.is_relative_to(out)}
    write(out/'archive_sha256.json',archives)
    write(out/'execution_receipt.json',dict(schema='s05.p02.final_handoff.receipt.v1',run=plan['run_id'],
        reviewed_baseline=review,execution_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        authorized_scope=plan['scope'],work_plan_sha256=staged_frozen,approved_inputs_sha256=plan['approved_inputs_sha256'],
        historical_artifacts_rehashed=protected,exit_codes=exits,tests=20,
        phases=plan['phase_status'],formal_physical_scoring=plan['formal_physical_scoring'],
        hardware_prerequisites=plan['hardware']|{'Final Test':'BLOCKED'},
        authorized_p03=False,authorized_robot_motion=False,rl_training_started=False,archive_root=str(run),
        published_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print('PASS P02 limited handoff seal',digest(out/'execution_receipt.json'),protected)


if __name__=='__main__':
    seal(*(Path(x) for x in sys.argv[1:]))
