"""Seal tested E1 outputs and exact build identity for submission to independent review."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def write(path,data):
    with path.open('x',encoding='utf-8') as f:
        f.write(json.dumps(data,sort_keys=True,indent=2)+'\n')


def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):
            h.update(data)
    return h.hexdigest()


def evidence_for(step):
    results={}
    for filename in sorted(step.glob('*.exit')):
        name=filename.stem
        results[name]=dict(exit=int(filename.read_text()), command=(step/(name+'.command')).read_text().strip(),
                           files_sha256={suffix:digest(step/(name+'.'+suffix))
                                         for suffix in ('time','stdout','stderr') if (step/(name+'.'+suffix)).exists()})
    return results


def seal(run,review,repo):
    manifest=read(review/'hmb_manifest.json')
    assert manifest['scope']=='E1_PRODUCTION' and manifest['run_id']==run.name
    gate=manifest['technical_gate']
    assert gate in {'PASS','FAIL','BLOCKED'}
    verification=int((review/'review.exit').read_text())
    assert verification==(0 if gate=='PASS' else 3)
    plan=read(run/'production_plan.json')
    files={}
    for path in sorted(run.glob('*')):
        if path.is_dir() and path.name!='products':
            files.update({str(f.relative_to(run)):digest(f) for f in path.rglob('*') if f.is_file()})
        elif path.is_file():
            files[path.name]=digest(path)
    outputs=read(review/'output_sha256.json')
    assert set(outputs['files']).isdisjoint(files) or files['production_plan.json']==outputs['files']['production_plan.json']
    all_output_sha256=dict(schema='s05.p01e.e1_run_outputs.v1',root=str(run),files=outputs['files']|files)
    copied={}
    chosen=('hmb_manifest','source_disposition','full_coverage','full_qc','segment_statistics','split_isolation',
            'runtime_resources','raw_traceability')
    for path in [review/(name+'.json') for name in chosen]+[run/'production_plan.json',
            run/'products/build_index.json',run/'products/build_log.jsonl',run/'session-2/resource_preflight.json']:
        target=repo/path.name
        assert not target.exists()
        shutil.copyfile(path,target)
        copied[target.name]=digest(target)
    commands={d.name:evidence_for(d) for d in sorted(run.iterdir())
              if d.is_dir() and d.name.startswith(('session-','smoke-','review-'))}
    assert any(x.get('build',{}).get('exit')==0 for x in commands.values())
    test_results=dict(schema='s05.p01e.e1.tests.v1',processes=commands,raw_witness=read(review/'raw_traceability.json'),
                      document_checks_pass=all((d/'documents.stderr').stat().st_size==0 for d in run.glob('session-*')
                          if (d/'documents.stderr').is_file()),technical_e1=gate,
                      final_review_scope=review.name)
    write(repo/'test_results.json', test_results)
    write(repo/'output_sha256.json',all_output_sha256)
    for name in ('output_sha256.json','test_results.json'):
        copied[name]=digest(repo/name)
    phase=read(ROOT.parent.parent/'phase_state.json')
    auth=phase['p01e_production_authorization']
    assert auth['run_id']==run.name and auth['scope']==manifest['scope']
    assert auth['candidate_config_sha256']==manifest['execution_candidate_sha256']
    assert auth['execution_lock_sha256']==manifest['execution_lock_sha256']
    build_refs={d.name:(d/'git_head.txt').read_text().strip() for d in run.glob('session-*')
                if (d/'git_head.txt').is_file()}
    audit_files={name:digest(ROOT/name) for name in ('audit_metrics.py','preflight.py','run_e1.sh','run_review.sh',
                                                    'seal_e1.py','verify_e1.py')}
    receipt=dict(schema='s05.p01e.e1.receipt.v1', phase='P01-E',scope='E1_PRODUCTION',run_id=run.name,
                 technical_gates={'E1':gate,'C2-RT':'BLOCKED','robot-base':'BLOCKED','P01-T':'BLOCKED','Final_Test':'BLOCKED'},
                 source_commit=auth['reviewed_commit'],git_during_build=build_refs,
                 git_during_review=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 verified_control_ticks=0,approval='AWAITING_HUMAN',
                 execution_candidate_sha256=auth['candidate_config_sha256'],execution_lock_sha256=auth['execution_lock_sha256'],
                 e0_receipt_sha256=phase['p01e_e0_acceptance']['receipt_sha256'],
                 review_artifacts_sha256=copied,review_implementation_sha256=audit_files,
                 raw_plan_sha256=digest(run/'production_plan.json'),counts=plan['total'],
                 published=manifest['published_partitions'],quarantined=manifest['quarantined_partitions'],
                 ubuntu_output=str(run),ubuntu_review_attempt=review.name)
    write(repo/'review_receipt.json',receipt)
    print(json.dumps(dict(gate=gate,receipt_sha256=digest(repo/'review_receipt.json'),counts=plan['total'],
                          manifest_sha256=copied['hmb_manifest.json'],published=manifest['published_partitions'],
                          quarantined=manifest['quarantined_partitions'],
                          archive_bytes=read(review/'runtime_resources.json')['total_archive_bytes']),sort_keys=True))


if __name__=='__main__':
    seal(*(Path(s).resolve() for s in sys.argv[1:]))
