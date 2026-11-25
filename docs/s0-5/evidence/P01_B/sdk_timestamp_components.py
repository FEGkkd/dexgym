"""Offline ABI check of CoreSdk_GetTimestampInfo, with no SDK session initialization."""
from collections import Counter,defaultdict
import ctypes
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import sys


class ManusTimestamp(ctypes.Structure):
    _fields_ = [('time',ctypes.c_uint64)]


class ManusTimestampInfo(ctypes.Structure):
    _fields_ = [('fraction',ctypes.c_uint16),('second',ctypes.c_uint8),('minute',ctypes.c_uint8),
                ('hour',ctypes.c_uint8),('day',ctypes.c_uint8),('month',ctypes.c_uint8),
                ('year',ctypes.c_uint32),('timecode',ctypes.c_bool)]


def decode_clock(path, library):
    with library.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!='b4e98f04144585cf51428c41e6af202dc664a599fb70a61adbcfe6ff0958825b':
        raise ValueError('Unexpected SDK binary')
    sdk = ctypes.CDLL(str(library))
    unpack = sdk.CoreSdk_GetTimestampInfo
    unpack.argtypes = (ManusTimestamp,ctypes.POINTER(ManusTimestampInfo))
    unpack.restype = ctypes.c_int
    buckets = defaultdict(list)
    with path.open(encoding='utf-8') as stream:
        for raw in stream:
            row = json.loads(raw)
            if 'manus' not in row or row['clock']['sdk_source']['value'] is None:
                continue
            value = row['clock']['sdk_source']['value']
            info = ManusTimestampInfo()
            status = unpack(ManusTimestamp(value),ctypes.byref(info))
            fields = {name:getattr(info,name) for name,_ in info._fields_}
            calendar = None
            if status==0 and not info.timecode:
                try:
                    calendar = datetime(info.year,info.month,info.day,info.hour,info.minute,info.second,
                                        info.fraction*1000)
                except ValueError:
                    pass
            key = row['recording_id'],row['index']['topic']
            buckets[key].append((value,status,fields,calendar,row['index']))
    records = []
    for (rid,topic),rows in sorted(buckets.items()):
        delta = [(b[3]-a[3]).total_seconds()*1000 for a,b in zip(rows,rows[1:])
                 if a[3] is not None and b[3] is not None]
        records.append(dict(recording_id=rid,topic=topic,total=len(rows),
                            sdk_return_codes=dict(Counter(str(r[1]) for r in rows)),
                            valid_calendar=sum(r[3] is not None for r in rows),
                            timecode_flag_true=sum(r[2]['timecode'] for r in rows),
                            calendar_step_p50_ms=median(delta) if delta else None,
                            calendar_zero_steps=delta.count(0),
                            calendar_backward_steps=sum(x<0 for x in delta),
                            examples=[dict(sdk_raw_value=r[0],sdk_components=r[2],index=r[4])
                                      for r in rows[:2]]))
    root=Path('/home/ubuntu/teleop_ws')
    identical_blob=subprocess.check_output(['git','-C',str(root),'hash-object',str(library)],text=True).strip()
    return dict(library_sha256=digest,library_blob=identical_blob,
                matches_reviewed_teleop_lib_blob=identical_blob=='363be71c6609a9624002e076c7d5e819d231fc28',
                per_recording_and_side=records,
                semantics='If timecode=false, SDK documentation defines the decoded fraction as ms. This is a component interpretation, not a ROS or controller clock fit.',
                online_capture_clock_verified=False)


if __name__=='__main__':
    lib = Path('/home/ubuntu/teleop_ws/src/ManusSDK/lib/amd64/libManusSDK-amd64.so')
    value = decode_clock(Path(sys.argv[1]),lib)
    print(json.dumps(value,indent=2,sort_keys=True))
