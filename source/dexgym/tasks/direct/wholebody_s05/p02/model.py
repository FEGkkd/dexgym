"""Named Marvin/Wuji geometry and static constraints for the P02 Oracle."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


SIDES=('l','r')
FINGERS=('thumb','index_finger','middle_finger','ring_finger','pinky')
CODES=('THJ','FFJ','MFJ','RFJ','LFJ')
WRISTS=('tcp_L','tcp_R')
PALMS=('l_wrist','r_wrist')
COUPLING_LIMIT=np.deg2rad(110.5)
COUPLING_WEIGHTS=(1.025,1.0)


def digest(path):
    with Path(path).open('rb') as f:
        return sha256(f.read()).hexdigest()


def asset_files(xml):
    """Resolve the packaged MJCF XML, includes and actual asset references."""
    xml=Path(xml).resolve()
    root=ET.parse(xml).getroot()
    compiler=root.find('compiler')
    settings={} if compiler is None else compiler.attrib
    paths={xml}
    for node in root.iter():
        if 'file' not in node.attrib:
            continue
        name=node.get('file')
        if node.tag=='include':
            paths.update(asset_files(xml.parent/name))
        else:
            kind='meshdir' if node.tag=='mesh' else 'texturedir' if node.tag=='texture' else 'assetdir'
            folder=settings.get(kind,settings.get('assetdir',''))
            paths.add((xml.parent/folder/name).resolve())
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'MJCF asset is missing: {missing}')
    return {str(p):digest(p) for p in sorted(paths)}


@dataclass(frozen=True)
class Geometry:
    wrists: np.ndarray
    rotations: np.ndarray
    palm: np.ndarray
    palm_rotation: np.ndarray
    landmarks: np.ndarray
    local: np.ndarray

    @property
    def tips(self):
        return self.landmarks[:,:,-1]


class RobotModel:
    def __init__(self,xml,clearance_m=0.0005):
        self.path=Path(xml).resolve()
        m=self.model=mujoco.MjModel.from_xml_path(str(self.path))
        self.data=mujoco.MjData(m)
        self.clearance_m=clearance_m
        arm=[f'Joint{j}_{s.upper()}' for s in SIDES for j in range(1,8)]
        groups=[arm[:7],arm[7:]]
        available={int(m.actuator(i).trnid[0]):m.actuator(i).name for i in range(m.nu)}
        joints_by_name={m.joint(i).name:i for i in range(m.njnt)}
        for s in SIDES:
            groups.append([m.joint(int(m.actuator(f'{s}_{code}{i}').trnid[0])).name
                           for code in CODES for i in range(4)])
        self.groups=groups
        self.names=tuple(j for g in groups for j in g)
        if not (m.nq==m.nv==m.nu==m.njnt==54 and len(set(self.names))==54 and
                set(self.names)==set(joints_by_name) and len(available)==54):
            raise ValueError('Expected exactly 54 named, independently driven hinges')
        self.joints=[m.joint(name) for name in self.names]
        if any(int(j.type[0])!=mujoco.mjtJoint.mjJNT_HINGE or not bool(j.limited[0]) for j in self.joints):
            raise ValueError('Expected bounded hinge states')
        self.actuators=[m.actuator(available[j.id]) for j in self.joints]
        self.q_indices=np.array([int(j.qposadr[0]) for j in self.joints])
        self.v_indices=np.array([int(j.dofadr[0]) for j in self.joints])
        joint_bounds=np.array([j.range for j in self.joints])
        drive_bounds=np.array([a.ctrlrange for a in self.actuators])
        self.bounds=np.c_[np.maximum(joint_bounds[:,0],drive_bounds[:,0]),
                          np.minimum(joint_bounds[:,1],drive_bounds[:,1])]
        if np.any(self.bounds[:,0]>=self.bounds[:,1]):
            raise ValueError('Empty joint/drive range intersection')
        mujoco.mj_resetDataKeyframe(m,self.data,m.key('home').id)
        mujoco.mj_forward(m,self.data)
        self.home=self.data.qpos[self.q_indices].copy()
        self.landmark_names=[[[self.names[14+s*20+f*4+i] for i in (0,2,3)]+[f'{side}_{finger}_tip']
                              for f,finger in enumerate(FINGERS)] for s,side in enumerate(SIDES)]
        self.coupling=np.zeros((8,54))
        for s in range(2):
            for n,(a,b) in enumerate(product((-1,1),repeat=2)):
                self.coupling[s*4+n,s*7+5:s*7+7]=(a*COUPLING_WEIGHTS[0],b*COUPLING_WEIGHTS[1])
        self.collision_pairs=self._pairs()
        self.geom_a=np.array([a for a,_ in self.collision_pairs])
        self.geom_b=np.array([b for _,b in self.collision_pairs])
        self.radius=m.geom_rbound[self.geom_a]+m.geom_rbound[self.geom_b]
        self.has_plane=(m.geom_type[self.geom_a]==mujoco.mjtGeom.mjGEOM_PLANE)|(
                         m.geom_type[self.geom_b]==mujoco.mjtGeom.mjGEOM_PLANE)
        self.tip_ids=[m.site(f'{s}_{f}_tip').id for s in SIDES for f in FINGERS]

    def _pairs(self):
        m=self.model
        explicit=set(zip(m.pair_geom1,m.pair_geom2))
        exclude={int(a) for a in m.exclude_signature}
        result=[]
        for i,j in combinations(range(m.ngeom),2):
            if (i,j) in explicit or (j,i) in explicit:
                result.append((i,j))
                continue
            if not ((m.geom_contype[i]&m.geom_conaffinity[j]) or (m.geom_contype[j]&m.geom_conaffinity[i])):
                continue
            a,b=sorted((int(m.geom_bodyid[i]),int(m.geom_bodyid[j])))
            if a==b or ((a<<16)+b) in exclude:
                continue
            w,z=m.body_weldid[a],m.body_weldid[b]
            if w==z:
                continue
            if w and z and not m.opt.disableflags&mujoco.mjtDisableBit.mjDSBL_FILTERPARENT:
                if m.body_weldid[m.body_parentid[w]]==z or m.body_weldid[m.body_parentid[z]]==w:
                    continue
            result.append((i,j))
        return result

    def set(self,q,contacts=False):
        q=np.asarray(q)
        if q.shape!=(54,) or not np.isfinite(q).all():
            raise ValueError('Expected a finite 54-joint radian state')
        self.data.qpos[self.q_indices]=q
        if contacts:
            mujoco.mj_fwdPosition(self.model,self.data)
        else:
            mujoco.mj_kinematics(self.model,self.data)
            mujoco.mj_comPos(self.model,self.data)

    def geometry(self,q=None,*,contacts=False):
        if q is not None:
            self.set(q,contacts)
        d=self.data
        w=np.array([d.site(s).xpos for s in WRISTS])
        r=np.array([d.site(s).xmat.reshape(3,3) for s in WRISTS])
        p=np.array([d.body(s).xpos for s in PALMS])
        pr=np.array([d.body(s).xmat.reshape(3,3) for s in PALMS])
        x=np.array([[[*(d.joint(j).xanchor.copy() for j in f[:3]),d.site(f[-1]).xpos.copy()]
                     for f in group] for group in self.landmark_names])
        return Geometry(w,r,p,pr,x,(x-p[:,None,None,:])@pr[:,None,:,:])

    def point_jac(self,kind,name):
        m,d=self.model,self.data
        jac,ang=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        if kind=='site':
            mujoco.mj_jacSite(m,d,jac,ang,m.site(name).id)
        elif kind=='body':
            mujoco.mj_jacBody(m,d,jac,ang,m.body(name).id)
        else:
            joint=m.joint(name)
            mujoco.mj_jac(m,d,jac,ang,d.joint(name).xanchor,int(joint.bodyid[0]))
        return jac[:,self.v_indices],ang[:,self.v_indices]

    def contact_gap(self,cap=.015):
        m,d=self.model,self.data
        p=d.geom_xpos
        lower=np.linalg.norm(p[self.geom_a]-p[self.geom_b],axis=1)-self.radius
        minimum=cap
        who=None
        for k in np.nonzero((lower<cap)|self.has_plane)[0]:
            i,j=self.collision_pairs[k]
            value=mujoco.mj_geomDistance(m,d,i,j,cap,None)
            if value<minimum:
                minimum=value
                who=(i,j)
        return float(minimum),who

    def legality(self,q):
        self.set(q,contacts=True)
        low,high=self.bounds.T
        range_violation=float(max(0,np.max(low-q),np.max(q-high)))
        coupled=COUPLING_LIMIT-self.coupling@q
        gap,pair=self.contact_gap()
        contacts=[float(c.dist) for c in self.data.contact]
        pairs=None if pair is None else [self.model.geom(i).name or
                     f'{self.model.body(int(self.model.geom(i).bodyid[0])).name}#geom{i}' for i in pair]
        return dict(range_violation_rad=range_violation,coupling_margin_rad=float(coupled.min()),
                    collision_clearance_m=gap,collision_pair=pairs,
                    penetration_m=min(0,*contacts) if contacts else 0.,
                    feasible=range_violation<=1e-9 and coupled.min()>=-1e-9 and
                    gap>=self.clearance_m-1e-7 and all(c>=-1e-7 for c in contacts))

    def contract(self):
        m=self.model
        home=self.geometry(self.home)
        mount_translation=np.einsum('sji,sj->si',home.rotations,home.palm-home.wrists)
        mount_rotation=np.transpose(home.rotations,(0,2,1))@home.palm_rotation
        return dict(model=str(self.path),assets=asset_files(self.path),mujoco=mujoco.__version__,
                    counts=dict(nq=m.nq,nv=m.nv,nu=m.nu,bodies=m.nbody,geoms=m.ngeom,exclusions=m.nexclude,
                                configured_collision_pairs=len(self.collision_pairs),equality=m.neq),
                    units={'joint_position':'rad','distance':'m','rotation':'proper 3x3'},
                    joint_order=self.names,groups=self.groups,
                    joints=[dict(name=j.name,actuator=a.name,position=int(j.qposadr[0]),velocity=int(j.dofadr[0]),
                                 physical_axis_body=j.axis.tolist(),anchor_body=j.pos.tolist(),
                                 qpos_zero=float(m.qpos0[i]),home=float(self.home[k]),range_rad=self.bounds[k].tolist(),
                                 model_range_rad=j.range.tolist(),drive_range_rad=a.ctrlrange.tolist())
                            for k,(j,a,i) in enumerate(zip(self.joints,self.actuators,self.q_indices,strict=True))],
                    wrist_sites=WRISTS,palms=PALMS,landmarks=self.landmark_names,
                    wrist_to_palm_translation_m=mount_translation.tolist(),wrist_to_palm_rotation=mount_rotation.tolist(),
                    tip_local_sites=[dict(site=m.site(i).name,body=m.body(int(m.site(i).bodyid[0])).name,
                                         position_m=m.site(i).pos.tolist(),type=int(m.site(i).type[0]))
                                     for i in self.tip_ids],clearance_m=self.clearance_m,
                    J67=dict(weights=COUPLING_WEIGHTS,limit_rad=COUPLING_LIMIT),
                    home_constraints=self.legality(self.home))


def independent_frames(robot,q):
    """Direct traversal of the compiled static transform tree, independent of mj_kinematics."""
    m=robot.model
    q_all=np.empty(m.nq)
    q_all[robot.q_indices]=q
    p=np.zeros((m.nbody,3))
    rotations=np.repeat(np.eye(3)[None],m.nbody,axis=0)
    for b in range(1,m.nbody):
        parent=int(m.body_parentid[b])
        p[b]=p[parent]+rotations[parent]@m.body_pos[b]
        base=Rotation.from_quat(m.body_quat[b,[1,2,3,0]]).as_matrix()
        r=rotations[parent]@base
        for k in range(int(m.body_jntadr[b]),int(m.body_jntadr[b]+m.body_jntnum[b])):
            if k<0:
                continue
            offset=m.jnt_pos[k]
            r2=r@Rotation.from_rotvec(m.jnt_axis[k]*(q_all[m.jnt_qposadr[k]]-m.qpos0[m.jnt_qposadr[k]])).as_matrix()
            p[b]+=r@offset-r2@offset
            r=r2
        rotations[b]=r
    points={}
    for name in WRISTS+tuple(n for hand in robot.landmark_names for finger in hand for n in finger[-1:]):
        s=m.site(name)
        b=int(s.bodyid[0])
        points['site',name]=(p[b]+rotations[b]@s.pos,
                              rotations[b]@Rotation.from_quat(s.quat[[1,2,3,0]]).as_matrix())
    for name in PALMS:
        b=m.body(name).id
        points['body',name]=(p[b],rotations[b])
    for name in (n for hand in robot.landmark_names for finger in hand for n in finger[:3]):
        j=m.joint(name)
        b=int(j.bodyid[0])
        points['joint',name]=(p[b]+rotations[b]@j.pos,rotations[b])
    return points
