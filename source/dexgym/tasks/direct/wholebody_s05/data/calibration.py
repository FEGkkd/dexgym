"""Load the recorded tracker extrinsics and test independent registered messages."""
from collections import defaultdict
import numpy as np
import yaml

from .geometry import compose,inverse,pose,residual,rotation,summary
from .raw_events import header_value


def extrinsics(path):
    data=yaml.safe_load(path.read_bytes())['tracker_calibration_node']['ros__parameters']
    result={}
    for side in ('left','right'):
        role=f'{side}_wrist'
        parent, child = data[role+'_parent'], data[role+'_child']
        if (parent,child)!=(role+'_raw',role):
            raise ValueError(f'Misnamed {side} calibration: {parent}/{child}')
        x=np.r_[data[role+'_translation'],data[role+'_quaternion_xyzw']].astype(np.float64)
        if x.shape!=(7,) or not np.isfinite(x).all():
            raise ValueError('Invalid extrinsic shape')
        r=rotation(x[3:]).as_matrix()
        reconstruction=compose(x,inverse(x))
        result[side]=dict(transform=x,parent=parent,child=child,quaternion_norm=float(np.linalg.norm(x[3:])),
                          determinant=float(np.linalg.det(r)),
                          orthogonality=float(np.linalg.norm(r.T@r-np.eye(3))),
                          inverse_residual=residual(reconstruction,np.array([0,0,0,0,0,0,1])))
    return result


def verify_pairs(raw,registered,calib,target):
    """Pair actual independent publications with the copied raw header, never by nearest time."""
    def group(events):
        output=defaultdict(list)
        for e in events:
            stamp=header_value(e['fields'])
            if stamp is not None:
                output[stamp].append(e)
        return output
    group_a,group_b=group(raw),group(registered)
    linear, angular, norms, matches, failed=[],[],[],[],[]
    repeated=0
    world=np.array([0,0,0,0,2**-0.5,2**-0.5,0])
    for timestamp in sorted(group_a.keys()&group_b.keys()):
        a,b=group_a[timestamp],group_b[timestamp]
        if len(a)!=1 or len(b)!=1:
            repeated += 1
            continue
        ae,be=a[0],b[0]
        va,vb=ae['fields'],be['fields']
        if not vb.get('valid'):
            failed.append(dict(stamp=timestamp,reason='REGISTERED_INVALID'))
            continue
        left,right=va['header']['frame_id'],vb['header']['frame_id']
        if right!=target or (left!=right and (left,right)!=('openvr_world','teleop_world')):
            failed.append(dict(stamp=timestamp,reason='CALIBRATION_INVALID_FRAME'))
            continue
        pa,pb=pose(va['pose']),pose(vb['pose'])
        result=compose(pa,calib)
        if left!=right:
            result=compose(world,result)
        t,w=residual(result,pb)
        linear.append(t); angular.append(w)
        norms.extend((abs(np.linalg.norm(pa[3:])-1),abs(np.linalg.norm(pb[3:])-1)))
        if len(matches)<3:
            matches.append(dict(raw=ae['index'],registered=be['index'],translation_m=t,rotation_rad=w))
    return dict(raw=len(raw),registered=len(registered),matched=len(linear),ambiguous_header_stamps=repeated,
                raw_unpaired=len(raw)-len(linear),registered_unpaired=len(registered)-len(linear),
                translation_m=summary(linear),rotation_rad=summary(angular),
                input_quat_unit_error=summary(norms),failures=failed,examples=matches)
