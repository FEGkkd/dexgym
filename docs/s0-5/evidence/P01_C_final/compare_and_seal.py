"""Compare the final patch to the last fixed-window review and record SHA evidence."""
import argparse
from collections import Counter
from hashlib import file_digest
from itertools import zip_longest
import json
from pathlib import Path
import re
import subprocess


BASE='aecfd2deff75a564b2ef70538f09fbc997cbd4a3'
CODE='source/dexgym/tasks/direct/wholebody_s05/data/synchronize.py'
TEST='tests/s0_5/unit/test_p01c_offline_boundary.py'
TOOL='docs/s0-5/evidence/P01_C_final/compare_and_seal.py'
PRODUCTS=('paired_regression.json','source_trace.jsonl','technical_gates.json')
CORE=('manus_left','manus_right','tracker_left','tracker_right')


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path,data):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,sort_keys=True)
        stream.write('\n')


def offline_velocity(state):
    interval=state['time_interval']
    return bool(state['valid'] and interval and interval[1]>interval[0])


def compare(before,after):
    status={mode:{'joint':Counter(),'streams':{}} for mode in ('causal','offline')}
    all_reasons={mode:{} for mode in status}
    safety=Counter()
    total=0
    with (before/'source_trace.jsonl').open() as a,(after/'source_trace.jsonl').open() as b:
        for x,y in zip_longest(a,b):
            if x is None or y is None:
                raise ValueError('The fixed tick denominator differs')
            old,new=json.loads(x),json.loads(y)
            identity=('recording_id','window','tick_ns','cutoff_ns')
            if any(old[k]!=new[k] for k in identity):
                raise ValueError('Window grid or source changed')
            total+=1
            safety[f"{old['safety_reason']}->{new['safety_reason']}"]+=1
            for mode in status:
                old_joint=old['joint_valid'] if mode=='causal' else offline_joint(old)
                new_joint=new['joint_valid'] if mode=='causal' else offline_joint(new)
                status[mode]['joint'][f'{int(old_joint)}->{int(new_joint)}']+=1
                for stream,left in old[mode].items():
                    right=new[mode][stream]
                    stream_status=status[mode]['streams'].setdefault(stream,Counter())
                    old_v=left['velocity_valid'] if mode=='causal' else offline_velocity(left)
                    new_v=right['velocity_valid'] if mode=='causal' else offline_velocity(right)
                    stream_status[f"pose:{int(left['valid'])}->{int(right['valid'])}"]+=1
                    stream_status[f'velocity:{int(old_v)}->{int(new_v)}']+=1
                    if left['reason']!=right['reason']:
                        all_reasons[mode].setdefault(stream,Counter())[f"{left['reason']}->{right['reason']}"]+=1
    return dict(ticks=total,mask_transitions=status,reason_changes=all_reasons,safety_transitions=safety)


def offline_joint(row):
    # The paired replay freezes both skew limits at the configured 35 ms.
    spans=row['offline_source_span_ns'].values()
    return all(row['offline'][stream]['valid'] for stream in CORE) and all(
        span is not None and span<=35_000_000 for span in spans)


def process_state(folder,step):
    logs={ext:folder/f'{step}.{ext}' for ext in ('command','stdout','stderr','exit')}
    count=re.search(r'Ran (\d+) tests? in ',logs['stderr'].read_text())
    return dict(exit=int(logs['exit'].read_text()),tests=int(count[1]) if count else None,
                command=logs['command'].read_text().strip(),sha256={p.name:sha(p) for p in logs.values()})


def seal(root,previous,attempt):
    expected=(root/'baseline.git').read_text().strip()
    if expected!=BASE or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=BASE:
        raise ValueError('Review baseline differs')
    folder=root/attempt
    old=previous/'products'
    new=folder/'products'
    parent=read(previous/'review_receipt.json')
    if any(sha(old/name)!=parent['products_sha256'][name] for name in PRODUCTS):
        raise ValueError('Last review evidence differs')
    red=process_state(root,'red')
    steps={key:process_state(folder,key) for key in ('unit','legacy_order','paired','integration')}
    if red['exit']!=1 or red['tests']!=8 or any(v['exit'] not in {0} for v in steps.values()):
        raise ValueError('Expected red/green evidence missing')
    a,b=[read(d/'paired_regression.json') for d in (old,new)]
    if a['thresholds']!=b['thresholds'] or b['thresholds']['skew_ns']!=35_000_000:
        raise ValueError('Frozen sync parameters changed')
    if (a['geometry']!=b['geometry'] or len(b['results'])!=23 or
            any(x['raw_realtime_interval_ns']!=y['raw_realtime_interval_ns'] or
                x['recording_id']!=y['recording_id'] or x['window']!=y['window']
                for x,y in zip(a['results'],b['results'],strict=True))):
        raise ValueError('Fixed corpus or software calibration changed')
    deltas=compare(old,new)
    for key,label in (('causal','causal_new'),('offline','offline_new')):
        transitions=deltas['mask_transitions'][key]['joint']
        if (transitions['1->1']+transitions['1->0']!=a['totals'][label]
                or transitions['1->1']+transitions['0->1']!=b['totals'][label]):
            raise ValueError('Tick counts disagree with paired run')
    write(new/'final_delta.json',deltas)
    outputs={file.name:sha(file) for file in sorted(new.iterdir()) if file.is_file()}
    write(new/'output_sha256.json',outputs)
    record=dict(phase='P01-C',scope='HMB_SIM_REVIEW',base=BASE,run_id=root.name,attempt=attempt,
                technical_gates=read(new/'technical_gates.json')['gates'],authority='Independent human approval pending',
                previous_review_receipt_sha256=sha(previous/'review_receipt.json'),
                red=red,green=steps,baseline=a['totals'],replay=b['totals'],changes=deltas,
                execution_source_sha256={p:sha(Path(p)) for p in
                    (CODE,TEST,TOOL,'docs/s0-5/evidence/P01_C_review/run_review.sh',
                     'source/dexgym/tasks/direct/wholebody_s05/data/review_audit.py')},
                preflight_sha256={name:sha(root/name) for name in ('freeze.sha256','red_test.sha256','review_preflight.log')},
                products_sha256=outputs,products_manifest_sha256=sha(new/'output_sha256.json'))
    write(folder/'final_receipt.json',record)
    print(sha(folder/'final_receipt.json'),record['products_manifest_sha256'])
    print(json.dumps(dict(changes=deltas['reason_changes'],joint={k:v['joint'] for k,v in deltas['mask_transitions'].items()},
                          before=a['totals'],after=b['totals'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('previous',type=Path)
    parser.add_argument('attempt')
    args=parser.parse_args()
    seal(args.root.resolve(),args.previous.resolve(),args.attempt)
