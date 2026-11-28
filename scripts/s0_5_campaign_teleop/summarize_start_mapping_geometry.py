import json
from hashlib import file_digest
from pathlib import Path
import sys
import numpy as np

run=Path(sys.argv[1])
index_path=run/"layer_index.json"
index=json.loads(index_path.read_bytes())
layers=index["layers"]
names=("m1","native_shared_raw","issued_fk","measured_fk")
pick=[layers.index(k) for k in names]
world=[]
score=[]
per_root=[]
for root in index["by_root"]:
    with np.load(run/"layers-verified"/root["array_path"],allow_pickle=False) as data:
        valid=data["hmb_joint_valid"]&data["wrist_valid"][:,pick].all(axis=(1,2))
        parity=data["raw_shared_reconstruction_error_m_rad"]
        valid&=(np.isfinite(parity)&(parity<=[1e-4,1e-3])).all(axis=(1,2))
        pose=data["wrist_pose_world"][valid][:,pick,:,:3]
        v=(pose[:,1]-pose[:,0])*1000
        world.append(v)
        score.append(np.linalg.norm(pose[:,3]-pose[:,0],axis=-1).max(axis=1)*1000)
        if len(v):
            per_root.append({"root":root["root"],"rows":len(v),
                "bimanual_midpoint_xyz_median_mm":np.median(v.mean(axis=1),axis=0).tolist()})
world=np.concatenate(world)
score=np.concatenate(score)
assert len(world)==index["overall"]["native_shared_on_same_four_layer_support"]["n"]
def summarize(v):
    mid=v.mean(axis=1)
    half=(v[:,1]-v[:,0])/2
    mid_dist=np.linalg.norm(mid,axis=1)
    half_dist=np.linalg.norm(half,axis=1)
    energy=np.sum(mid_dist**2)
    return {"n":len(v),"midpoint_position_p50_p95_mm":np.percentile(mid_dist,[50,95]).tolist(),
        "relative_wrist_vector_change_p50_p95_mm":np.percentile(2*half_dist,[50,95]).tolist(),
        "midpoint_xyz_median_mm":np.median(mid,axis=0).tolist(),
        "midpoint_share_of_squared_wrist_position_discrepancy":float(energy/(energy+np.sum(half_dist**2)))}
with index_path.open("rb") as f:
    identity=file_digest(f,"sha256").hexdigest()
cut=float(np.percentile(score,95))
result={"schema":"s05.teleop.start_handoff_mapping_geometry.v1","layer_index_sha256":identity,
        "contrast":"Native shared minus frozen M1 in MuJoCo world axes",
        "all_common":summarize(world),"position_tail_threshold_mm":cut,
        "worst_position_tail":summarize(world[score>=cut]),"by_root":per_root,
        "identity":"For each frame, mean of the squared left and right pose position differences equals squared midpoint difference plus squared half-relative-vector difference."}
with (run/"mapping_geometry.json").open("x") as f:
    json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    f.write("\n")
print(result["all_common"])
print(result["worst_position_tail"])
