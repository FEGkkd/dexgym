"""Snapshot data-path source lines, Git identities and literal SDK representations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def witness(root, name, bounds):
    path = root/name
    raw = path.read_bytes()
    lines = raw.decode(errors='replace').splitlines()
    rev = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    return {'path':str(path),'file_sha256':hashlib.sha256(raw).hexdigest(),'current_git_head':rev,
            'excerpts':{str(i):lines[i-1] for a,b in bounds for i in range(a,b+1) if i<=len(lines)}}


if __name__=='__main__':
    teleop = Path('/home/ubuntu/teleop_ws')
    recorder = Path('/home/ubuntu/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection')
    old = Path(sys.argv[1])
    result = {
        'sources':[
            witness(teleop,'src/manus_ros2/src/ManusDataPublisher.cpp',[(67,70),(370,399),(594,599)]),
            witness(teleop,'src/ManusSDK/include/ManusSDKTypes.h',[(938,955)]),
            witness(teleop,'src/ManusSDK/include/ManusSDK.h',[(92,102)]),
            witness(recorder,'ros2_ws/src/dex_data_collection/src/recorder.cpp',
                    [(138,151),(195,205),(473,481),(619,625),(694,699)]),
        ],
        'approved_a_samples_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
        'interpretation':[
            'ManusTimestamp.time is an opaque compressed 64-bit value in this adapter.',
            'The publisher copies publishTime.time into source_timestamp_us. Steady-clock duration is explicitly microseconds only for receive_timestamp_us.',
            'CoreSdk_GetTimestampInfo exposes calendar/timecode components. Capture mode, epoch, device-vs-SDK origin and physical sync are unverified for these recordings.',
            'Timer publication can use a cached skeleton; repeated value alone cannot identify repeated capture.',
        ],
        'clock_mapping_applied':False,
    }
    print(json.dumps(result,indent=2,sort_keys=True))
