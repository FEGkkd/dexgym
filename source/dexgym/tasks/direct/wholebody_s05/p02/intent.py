"""Semantic MANUS chains and simulation-only bimanual target construction."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


CHAINS=('Thumb','Index','Middle','Ring','Pinky')


def unit(v):
    v=np.asarray(v,dtype=float)
    n=np.linalg.norm(v,axis=-1,keepdims=True)
    if not np.isfinite(n).all() or np.any(n<1e-7):
        raise ValueError('Human geometry basis or bone is degenerate')
    return v/n


def hand_basis(knuckles):
    long=unit(np.mean(knuckles,axis=0))
    across=knuckles[0]-knuckles[-1]
    thumbward=unit(across-long*(long@across))
    return np.c_[long,thumbward,np.cross(long,thumbward)]


def semantic_order(topology):
    """Resolve chains by parent graph and function, irrespective of original array order."""
    nodes=topology['joints']
    by_id={x['node_id']:x for x in nodes}
    if len(by_id)!=25:
        raise ValueError('Expected approved 25-node hand topology')
    root,=[x for x in nodes if x['parent_node_id']==x['node_id']]
    if root['chain_type'] in CHAINS:
        raise ValueError('An anatomical finger cannot be the MANUS root')
    chains=[]
    for name in CHAINS:
        group=[x for x in nodes if x['chain_type']==name]
        by_type={x['joint_type'].upper():x for x in group}
        order=('MCP','PIP','DIP','TIP')
        path=[by_type[k] for k in order]
        for a,b in zip(path,path[1:]):
            seen=set()
            p=b['parent_node_id']
            while p!=a['node_id']:
                if p in seen or p==root['node_id'] or by_id[p]['chain_type']!=name:
                    raise ValueError(f'Malformed {name} anatomical chain')
                seen.add(p)
                p=by_id[p]['parent_node_id']
        chains.append([q['original_order'] for q in path])
    return root['original_order'], np.asarray(chains)


@dataclass(frozen=True)
class Target:
    wrist: np.ndarray
    rotation: np.ndarray
    local: np.ndarray
    world: np.ndarray
    pinch: np.ndarray
    cross: np.ndarray
    relative_wrist: np.ndarray
    relative_rotation: np.ndarray

    def json(self):
        return {k:v.tolist() for k,v in vars(self).items()}


def pinch(tips):
    return tips[:,1:]-tips[:,:1]


def cross_hand(tips,pairs):
    return np.asarray([tips[1,b]-tips[0,a] for a,b in pairs])


def target_geometry(wrist,rotation,local,mount,pairs):
    p,r=mount
    palm_position=wrist+np.einsum('sij,sj->si',rotation,p)
    palm_rotation=rotation@r
    world=local@np.transpose(palm_rotation,(0,2,1))[:,None]+palm_position[:,None,None]
    tips=world[:,:,-1]
    return Target(wrist,rotation,local,world,pinch(local[:,:,-1]),cross_hand(tips,pairs),
                  rotation[0].T@(wrist[1]-wrist[0]),rotation[0].T@rotation[1])


def robot_target(robot,g,pairs):
    return Target(g.wrists,g.rotations,g.local,g.landmarks,pinch(g.local[:,:,-1]),
                  cross_hand(g.tips,pairs),g.rotations[0].T@(g.wrists[1]-g.wrists[0]),
                  g.rotations[0].T@g.rotations[1])


def preserve_near_pinch(points,human,scale,transform,threshold):
    """Match active human vectors exactly with the least squares endpoint translation."""
    old=points[:,-1]
    raw=human[1:,-1]-human[:1,-1]
    active=np.flatnonzero(np.linalg.norm(raw,axis=1)<=threshold)
    wanted=(raw@transform.T)*scale
    if not len(active):
        return points.copy(),active,wanted
    tips=old.copy()
    thumb=(old[0]+(old[active+1]-wanted[active]).sum(axis=0))/(len(active)+1)
    tips[0]=thumb
    tips[active+1]=thumb+wanted[active]
    return points+(tips-old)[:,None,:]*np.array([0,1/3,2/3,1])[None,:,None],active,wanted


def layout(row):
    k=[semantic_order(t) for t in row['topology']]
    fingers=np.asarray([p[inds,:3]-p[r,:3] for p,(r,inds) in zip(row['palm'],k,strict=True)])
    common=np.asarray([np.concatenate((p[r:r+1,:3],p[inds,:3].reshape(20,3)),axis=0)
                       for p,(r,inds) in zip(row['sdk_common'],k,strict=True)])
    return fingers, common


class Mapper:
    def __init__(self,robot,train,plan):
        if not train or any(x['split']!='Train' for x in train):
            raise ValueError('Mapping calibration can only access the training cohort')
        self.pairs=plan['cross_hand_pairs']
        self.threshold=plan['pinch_raw_threshold_m']
        g=robot.geometry(robot.home)
        self.mount=(np.einsum('sji,sj->si',g.rotations,g.palm-g.wrists),g.rotations.transpose(0,2,1)@g.palm_rotation)
        self.roots=g.local[:,:,0]
        self.lengths=np.linalg.norm(np.diff(g.local,axis=2),axis=-1)
        self.basis=np.stack([hand_basis(knuckles[1:]) for knuckles in self.roots])
        self.anchor=anchor=min(train,key=lambda a:(a['proxy_motion'],a['key']))
        p=anchor['wrist'][:,:3]
        u=p[1]-p[0]
        v=g.wrists[1]-g.wrists[0]
        yaw=np.arctan2(v[1],v[0])-np.arctan2(u[1],u[0])
        self.world_rotation=Rotation.from_rotvec([0.,0.,yaw]).as_matrix()
        self.world_translation=g.wrists.mean(axis=0)-self.world_rotation@p.mean(axis=0)
        tracker_rotation=Rotation.from_quat(anchor['wrist'][:,3:]).as_matrix()
        self.wrist_mount=(self.world_rotation@tracker_rotation).transpose(0,2,1)@g.rotations
        ratios=[]
        for row in train:
            fingers,_=layout(row)
            ratios.append(self.lengths/np.linalg.norm(np.diff(fingers,axis=2),axis=-1))
        self.hand_scale=np.median(ratios,axis=(0,2,3))

    def map(self,row):
        fingers,common=layout(row)
        points=[]
        audits=[]
        for s,f in enumerate(fingers):
            k=self.basis[s]@hand_basis(f[1:,0]).T
            direction=unit(np.diff(f,axis=1))@k.T
            shape=np.concatenate((self.roots[s,:,None],
                                  self.roots[s,:,None]+np.cumsum(direction*self.lengths[s,:,:,None],axis=1)),axis=1)
            corrected,active,desired=preserve_near_pinch(shape,f,self.hand_scale[s],k,self.threshold)
            actual=corrected[1:,-1]-corrected[:1,-1]
            before=shape[1:,-1]-shape[:1,-1]
            final_lengths=np.linalg.norm(np.diff(corrected,axis=1),axis=-1)
            audits.append(dict(near_pinch_pairs=active.tolist(),tol_active=np.linalg.norm((actual-desired)[active],axis=1).tolist(),
                raw_vectors=(f[1:,-1]-f[:1,-1]).tolist(),desired_vectors=desired.tolist(),
                before_vectors=before.tolist(),after_vectors=actual.tolist(),
                before_error_m=np.linalg.norm(before-desired,axis=1).tolist(),
                after_error_m=np.linalg.norm(actual-desired,axis=1).tolist(),
                bone_length_delta_m=(final_lengths-self.lengths[s]).tolist(),scale=float(self.hand_scale[s]),
                proper_rotation_det=float(np.linalg.det(k))))
            points.append(corrected)
        w=row['wrist'][:,:3]@self.world_rotation.T+self.world_translation
        r=self.world_rotation@Rotation.from_quat(row['wrist'][:,3:]).as_matrix()@self.wrist_mount
        target=target_geometry(w,r,np.stack(points),self.mount,self.pairs)
        wrist_distortion=target.wrist[1]-target.wrist[0]-self.world_rotation@(row['wrist'][1,:3]-row['wrist'][0,:3])
        return target,common,dict(hands=audits,relative_wrist_translation_delta_m=float(np.linalg.norm(wrist_distortion)))

    def contract(self):
        return dict(scope='HISTORICAL_HMB_SIMULATION_GEOMETRY_ONLY',anchor=self.anchor['key'],
            translation_scale=1.,world_rotation=self.world_rotation.tolist(),world_translation_m=self.world_translation.tolist(),
            tracker_to_tcp_orientation_mount=self.wrist_mount.tolist(),hand_scale=self.hand_scale.tolist(),
            wrist_to_palm_translation=self.mount[0].tolist(),wrist_to_palm_rotation=self.mount[1].tolist(),
            cross_hand_pairs=self.pairs,near_pinch_raw_threshold_m=self.threshold)
