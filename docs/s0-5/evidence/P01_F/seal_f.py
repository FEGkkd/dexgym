"""Seal the independent F1 qualification and its scoped P02 data handoff."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


HERE=Path(__file__).resolve().parent
E=HERE.parent/'P01_E_E1'
BASE='b22f8be3c35bf1b5ad9bae98335ecd1510cacc1e'
MANIFEST='e63df2698feb812bc300593d9bf8d1ad3dcae1d2dc59aa4d3a3dfa96851d8074'
RECEIPT='0930e732e92e6213742e8b48cd1921d980c1315c7c108bda1ea24c0c59de483b'
PLAN='0f94b842e2558ec04da257aff94d469ae9b965968fff1a4593382fc2eea09404'


def file_sha(p):
    with p.open('rb') as f:
        return sha256_file(f)


def sha256_file(f):
    h=sha256()
    for block in iter(lambda:f.read(2**20),b''):
        h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def write(p,data):
    with p.open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def attempts(run):
    tests={}
    for d in sorted(run.iterdir()):
        if d.is_dir() and (d/'run.exit').is_file():
            tests[d.name]=dict(exit=int((d/'run.exit').read_text()),command=(d/'run.command').read_text().strip(),
                         implementation_base=(d/'git_head.txt').read_text().strip(),
                         sha256={n:file_sha(d/n) for n in ('run.stderr','run.stdout','run.time')})
    return tests


def lineage(old_a,roots):
    hashes=read(old_a/'output_sha256.json')
    result=defaultdict(lambda:dict(lineage_witnesses=[],relationship_counts=Counter(),use=Counter()))
    for name in ('prior_use.jsonl','lineage.jsonl'):
        assert file_sha(old_a/name)==hashes[name]
        with (old_a/name).open() as f:
            for line in f:
                record=json.loads(line)
                parents=set(record.get('root_episode_ids') or [])
                p=record.get('root_episode_id')
                if p:
                    parents.add(p)
                for candidate in sorted(set(roots)&parents):
                    item=result[candidate]
                    if name=='lineage.jsonl':
                        relation=record['relationship']
                        item['relationship_counts'][relation]+=1
                        if record['episode_id'] in (candidate,roots[candidate]['recording_id']):
                            item['lineage_witnesses'].append({k:record.get(k) for k in
                               ('episode_id','relationship','adjudication','parent_ids')})
                    else:
                        item['use'][record.get('role','UNKNOWN')]+=1
    for root,source in roots.items():
        witnesses=result[root]['lineage_witnesses']
        assert any(x['episode_id']==root and x['relationship']=='CAPTURE_ROOT' for x in witnesses),root
        assert any(x['episode_id']==source['recording_id'] and root in x['parent_ids'] for x in witnesses),root
    return {r:dict(**result[r], final_test='BLOCKED',unknown_is_proof_of_nonuse=False) for r in sorted(roots)}


def maxima(raw):
    result=defaultdict(float)
    for goal in raw['cases']:
        for row in goal['checks']:
            mode=row['mode']
            for key in ('pose_residual_m_rad','sdk_residual_m_rad'):
                if key in row:
                    x,y=row[key]
                    result[mode+'.'+key+'.m']=max(result[mode+'.'+key+'.m'],x)
                    result[mode+'.'+key+'.rad']=max(result[mode+'.'+key+'.rad'],y)
            v=row.get('velocity_max_abs_residual')
            if v is not None:
                result[mode+'.velocity_abs_m_s']=max(result[mode+'.velocity_abs_m_s'],v)
        for t in goal['calibration']:
            for key in ('translation_m','rotation_rad'):
                result['software_registration.'+key]=max(result['software_registration.'+key],t[key])
    return result


def tail_compact(report):
    data=[]
    for case in report['cases']:
        g=case['goal']
        label=g['name']
        intervals=case['source_interval_checks']
        if label.endswith('.speed'):
            a,b,_=label.split('.')
            intervals=[x for x in intervals if (x['mode'],x['stream'])==(a,b)]
        data.append(dict(metric=label,value=g['value'],partition=g['partition'],tick=g['tick'],node=g['node'],
            root=case['binding']['parent'],split=case['binding']['split'],
            target_tick_ns=case['window']['ticks_ns'][case['window']['tick_indices'].index(g['tick'])],
            original_intervals=intervals,classification=case['classification'],disposition=case['disposition'],
            observed_invalid_nearby=case['tracking_interruption_observed'],cause=case['physical_cause'],
            neighboring_raw_count=len(case['window']['adjacent_raw']),
            neighboring_motion_edges=len(case['window']['raw_pose_changes'])))
    return dict(top_k=report['top_k'],rows=data,verified_physical_timing=False,
                decision='RETAIN_FROZEN_QC' if all(x['disposition']=='RETAIN_FROZEN_QC' for x in data) else 'REVIEW')


def seal(run,e1,old_a,archive,raw_dir,unit,out):
    assert file_sha(E/'hmb_manifest.json')==MANIFEST and file_sha(E/'review_receipt.json')==RECEIPT
    assert file_sha(E/'production_plan.json')==PLAN
    view=read(archive/'independent_integrity.json')
    stat=read(archive/'history_reference_statistics.json')
    selected=read(archive/'selected_samples.json')
    raw=read(raw_dir/'raw_reconstruction.json')
    tail=tail_compact(read(raw_dir/'tail_anomalies.json'))
    reader=read(raw_dir/'reader_check.json')
    images=read(raw_dir/'visual_receipt.json')
    assert file_sha(e1/'review-1/hmb_manifest.json')==MANIFEST
    res=read(E/'output_sha256.json')
    for name,want in res['files'].items():
        filename=(e1/name).resolve()
        assert filename.is_relative_to(e1)
        if name in view['hashes']:
            assert view['hashes'][name]==want
        else:
            assert file_sha(filename)==want,name
    verify_history=lineage(old_a,stat['roots'])
    tests=attempts(run)
    numbers=(unit/'run.stderr').read_text()
    matches=re.search(r'Ran (\d+) tests?',numbers)
    assert matches and tests[unit.name]['exit']==0
    gate='PASS' if (tests[archive.name]['exit']==tests[raw_dir.name]['exit']==0 and
          view['status']==reader['status']=='PASS' and not raw['failures'] and len(raw['confirmed_roots'])==32 and
          not selected['missing_roots'] and not selected['missing_scenarios'] and tail['decision']=='RETAIN_FROZEN_QC'
          and len(tail['rows'])==33 and len(images)>=8) else 'BLOCKED'
    compact_view={k:v for k,v in view.items() if k!='hashes'}
    compact_view.update(partition_files=len(view['hashes']),e1_outputs_verified=len(res['files']),
        partition_hashes_sha256=file_sha(archive/'independent_integrity.json'),
        lineage_and_prior_use=verify_history,manifest_sha256=MANIFEST,plan_sha256=PLAN,receipt_sha256=RECEIPT)
    raw_brief={k:v for k,v in raw.items() if k!='cases'}
    raw_brief.update(maximum_residuals=maxima(raw),
        source_witnesses=[dict(root=c['goal']['name'],partition=c['partition']['key'],tick=c['goal']['tick'],
                            sources={x['stream']:x['source'] for x in c['checks'] if x['mode']=='causal'},
                            calibration=c['calibration']) for c in raw['cases'] if c['goal']['kind']=='root'],
        repeated_sdk=[c['repeated_sdk'] for c in raw['cases'] if 'repeated_sdk' in c],
        scenario_witness_counts=Counter(c['goal']['name'] for c in raw['cases'] if c['goal']['kind']=='scenario'))
    assert out.parent==run and out.name.startswith('sealed')
    out.mkdir(mode=0o700)
    report_objects={
        'independent_integrity.json':compact_view,
        'history_reference_statistics.json':{k:v for k,v in stat.items() if k!='partitions'},
        'raw_reconstruction_report.json':raw_brief,
        'tail_anomaly_review.json':tail,
        'reader_check.json':reader,
        'selected_samples.json':selected,
        'test_results.json':dict(unit_count=int(matches.group(1)),runs=tests,technical_gate=gate),
    }
    products={name:'QUALIFIED_AWAITING_HUMAN' if gate=='PASS' else 'BLOCKED'
              for name in ('HMB_GEOMETRY_DEVELOPMENT','HMB_SIMULATED_CAUSAL','HMB_OFFLINE_DIAGNOSTIC')}
    products.update(HMB_REALTIME_VERIFIED='BLOCKED',Robot_Telemetry_Bank='BLOCKED',Independent_Final_Test='BLOCKED')
    report_objects['data_qualification_matrix.json']=dict(technical_gate=gate,products=products,
        in_scope='historical HMB development',P02_start_authorized=False,physical_robot_base='BLOCKED',
        P03_required_accepted=['P02','P01-T','C2-RT','robot-base'])
    counts=stat['groups']['All']
    report_objects['p02_handoff_manifest.json']=dict(schema='s05.p01f.handoff.v1',archive_reference_only=True,
         e1_manifest_sha256=MANIFEST,e1_review_receipt_sha256=RECEIPT,e1_plan_sha256=PLAN,
         e1_run_id='P01-E-E1-20261126-3f90652',source_commit=BASE,
         ubuntu_products=str(e1/'products'),scope=products,technical_gate=gate,
         split_roots={s:sorted(r for r,row in stat['roots'].items() if row['split']==s) for s in ('Train','Validation')},
         schema_name='s05.hmb.simulation.v1',reader='dexgym.tasks.direct.wholebody_s05.hmb_reader.HMBReader',
         actor_allowlist_sha256=file_sha(HERE/'actor_feature_allowlist.json'),
         aggregate={k:counts[k] for k in ('ticks','history_pose','history_velocity','history_common')},
         P02_start_authorized=False)
    for name,doc in report_objects.items():
        write(out/name,doc)
    for name in images:
        shutil.copyfile(raw_dir/name,out/name)
    output_index={str(p.relative_to(run)):file_sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write(out/'output_sha256.json',dict(schema='s05.p01f.outputs.v1',root=str(run),files=output_index))
    checksums={f.name:file_sha(f) for f in sorted(out.iterdir())}
    receipt=dict(schema='s05.p01f.receipt.v1',phase='P01-F',scope='INDEPENDENT_HMB_QUALIFICATION',run_id=run.name,
        approved_e1_commit=BASE,approved_e1_manifest_sha256=MANIFEST,approved_e1_receipt_sha256=RECEIPT,
        approved_e1_plan_sha256=PLAN,technical_gates={'F1':gate,'C2-RT':'BLOCKED','robot-base':'BLOCKED',
                                  'P01-T':'BLOCKED','Final_Test':'BLOCKED'},verified_control_ticks=0,
        approval='AWAITING_HUMAN',P02_start_authorized=False,evidence_sha256=checksums,
        sample_plan_sha256=file_sha(HERE/'sample_plan.json'),ubuntu_run=str(run),
        git_at_seal=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(out/'review_receipt.json',receipt)
    print(json.dumps(dict(gate=gate,receipt_sha256=file_sha(out/'review_receipt.json'),files=len(checksums),
                          history={s:{k:v for k,v in stat['groups'][s].items() if 'history' in k or 'reference' in k}
                                   for s in ('All','Train','Validation')},
                          raw={k:v for k,v in raw_brief.items() if k in ('counts','raw_pairs','raw_files','reuse_vs_e1',
                                 'maximum_residuals','scenario_witness_counts')},images=images)))


if __name__=='__main__':
    seal(*(Path(p).resolve() for p in sys.argv[1:]))
