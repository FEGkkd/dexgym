"""Show why checking only pose misses the original simultaneous-delivery defect."""
import json
import subprocess
import sys
from types import ModuleType

import numpy as np

from dexgym.tasks.direct.wholebody_s05.data.sample_identity import Identity


def main():
    package='dexgym.tasks.direct.wholebody_s05.data'
    name=f'{package}.legacy_order_witness'
    module=ModuleType(name)
    module.__package__=package
    sys.modules[name]=module
    path='source/dexgym/tasks/direct/wholebody_s05/data/synchronize.py'
    commit='0d9d351b73e543eab07cfe6fef8f55a7d12d912d'
    frozen=subprocess.check_output(['git','show',f'{commit}:{path}'])
    exec(compile(frozen,str(path),'exec'),module.__dict__)
    def event(seq,t,stream,safe=None):
        return module.Event(stream,Identity(stream,seq,seq,seq,str(seq)),{'physical_message':seq},
                            t,(t,t),(t,t),'SIMULATED',np.array([[seq,0,0,0,0,0,1.]]),safe=safe)
    events=[event(1,0,'tracker_safety',True),event(1,3,'manus_left'),event(2,3,'manus_left')]
    params={'age_ns':60,'skew_ns':35,'safety_age_ns':100,'gap_ns':60}
    a,b=[module.batch_causal(e,[(6,6)],params,'SIMULATED')[0]['streams']['manus_left']
         for e in (events,events[::-1])]
    same_pose=np.array_equal(a['pose'],b['pose'])
    if not same_pose or a['input_reason']==b['input_reason']:
        raise ValueError('Original simultaneous-delivery diagnostic discrepancy not reproduced')
    print(json.dumps(dict(baseline=commit,same_pose=same_pose,
                          original_input_reason=[a['input_reason'],b['input_reason']])))


if __name__=='__main__':
    main()
