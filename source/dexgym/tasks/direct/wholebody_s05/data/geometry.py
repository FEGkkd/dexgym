"""Physical input geometry. Positions are metres, quaternion arrays are XYZW."""
import math
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def rotation(q):
    q=np.asarray(q,dtype=np.float64)
    if q.shape[-1]!=4 or not np.isfinite(q).all() or np.any(np.linalg.norm(q,axis=-1)<=0):
        raise ValueError('Non-finite or zero XYZW quaternion')
    return Rotation.from_quat(q)


def pose(values):
    state=np.r_[[values['position'][k] for k in 'xyz'],
                [values['orientation'][k] for k in 'xyzw']].astype(np.float64)
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite pose')
    rotation(state[3:])
    return state


def compose(t_a_b,t_b_c):
    r=rotation(t_a_b[3:])
    return np.r_[t_a_b[:3]+r.apply(t_b_c[:3]),(r*rotation(t_b_c[3:])).as_quat()]


def inverse(t):
    r=rotation(t[3:]).inv()
    return np.r_[-r.apply(t[:3]),r.as_quat()]


def residual(a,b):
    linear=np.linalg.norm(a[:3]-b[:3])
    angular=(rotation(a[3:]).inv()*rotation(b[3:])).magnitude()
    return float(linear),float(angular)


def interpolate(a,b,w):
    if not 0<=w<=1 or a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=7:
        raise ValueError('Invalid aligned pose interpolation inputs')
    values=(1-w)*a+w*b
    # Slerp normalizes and uses the shortest relative quaternion arc.
    for i,(l,r) in enumerate(zip(a,b)):
        values[i,3:]=Slerp([0,1],rotation(np.vstack((l[3:],r[3:]))))([w]).as_quat()[0]
    return values


def topology(nodes, expected_side, declared_side):
    ids=[v['node_id'] for v in nodes]
    parents={v['node_id']:v['parent_node_id'] for v in nodes}
    if not nodes or len(set(ids))!=len(ids) or any(type(i) is not int for i in ids):
        raise ValueError('Missing or repeated MANUS node ID')
    roots=[i for i in ids if parents[i] in {-1,i}]
    if len(roots)!=1 or str(declared_side).lower()!=expected_side:
        raise ValueError('Invalid root convention or MANUS side')
    root=roots[0]
    offset={n:i for i,n in enumerate(ids)}
    for i in ids:
        seen=set()
        while i!=root:
            if i in seen or i not in parents:
                raise ValueError('Non-root topology loop or missing parent')
            seen.add(i)
            i=parents[i]
    for n in nodes:
        if not n.get('joint_type') or not n.get('chain_type'):
            raise ValueError('Missing joint or chain identity')
        parent=parents[n['node_id']]
        if n['node_id']!=root and parent!=root and nodes[offset[parent]]['chain_type']!=n['chain_type']:
            raise ValueError('Different fingers below a non-root parent')
    order=[n.get('original_order') for n in nodes]
    if order!=list(range(len(nodes))):
        raise ValueError('Original MANUS ordering changed')
    return offset[root],[(offset[i],offset[parents[i]]) for i in ids if i!=root]


def hand_geometry(nodes, expected_side, declared_side):
    root,edges=topology(nodes,expected_side,declared_side)
    z=np.asarray([np.r_[v['node_position'],v['node_orientation']] for v in nodes],dtype=np.float64)
    if z.shape!=(len(nodes),7) or not np.isfinite(z).all():
        raise ValueError('Nonfinite MANUS node')
    inv=rotation(z[root,3:]).inv()
    palm=np.column_stack((inv.apply(z[:,:3]-z[root,:3]),(inv*rotation(z[:,3:])).as_quat()))
    lengths=np.asarray([np.linalg.norm(z[i,:3]-z[j,:3]) for i,j in edges])
    return z,palm,lengths,edges


def summary(values):
    values=np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Cannot silently drop non-finite metric values')
    return dict(count=len(values),**{name:float(f(values)) if len(values) else None for name,f in
                (('p50',lambda v:np.percentile(v,50)),('p95',lambda v:np.percentile(v,95)),
                 ('p99',lambda v:np.percentile(v,99)),('min',np.min),('max',np.max))})
