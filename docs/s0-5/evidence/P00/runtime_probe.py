"""Read-only ROS bag, source and existing CPU test audit on Ubuntu."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

HOME = Path.home()
D = HOME/'dexgym'
T = HOME/'teleop_ws'
W = Path('/tmp/dexgym-s04-main.Wwwqff99')
R = D/'data/processed/wholebody-s04/world_hand20_gravity_20000_20261125/checkout'
J = D/'data/processed/wholebody-s04/world_hand20_aligned_from_scratch_20261125'
P = 'source/dexgym/tasks/direct/'
PY = HOME/'miniconda3/envs/wuji/bin/python'


def cmd(argv, cwd=D, timeout=40, env=None):
    try:
        x=subprocess.run(argv,cwd=cwd,capture_output=True,encoding='utf-8',errors='replace',timeout=timeout,env=env)
        return dict(argv=list(map(str,argv)),cwd=str(cwd),exit=x.returncode,stdout=x.stdout,stderr=x.stderr)
    except (OSError,subprocess.TimeoutExpired) as error:
        return dict(argv=list(map(str,argv)),cwd=str(cwd),status='BLOCKED',error=str(error))


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def excerpt(root, relative, pattern, limit=35):
    p=root/relative
    if not p.is_file():
        return {'status':'MISSING'}
    lines=[(i,line.strip()) for i,line in enumerate(p.read_text().splitlines(),1) if re.search(pattern,line)]
    return dict(sha256=digest(p),matched=len(lines),lines=lines[:limit])


def data():
    root=D/'data/raw/wholebody_s0'
    raw_dirs=sorted(p for day in root.iterdir() for p in day.iterdir() if p.is_symlink() or p.is_dir())
    chosen=raw_dirs[0]
    files=sorted(p.relative_to(chosen).as_posix() for p in chosen.rglob('*') if p.is_file())
    recordings=sorted(p for p in chosen.rglob('*.mcap') if p.is_file())
    official=next(iter(sorted((T/'data').glob('teleop_no_camera_*/*.mcap'))))
    ros_command=['bash','-lc','source /opt/ros/jazzy/setup.bash && ros2 bag info -s mcap "$1"','p00']
    cpu_witness='''import mujoco,numpy as np,sys
m=mujoco.MjModel.from_xml_path(sys.argv[1]); d=mujoco.MjData(m)
print('nq/nv/nu/njnt',m.nq,m.nv,m.nu,m.njnt,'dt',m.opt.timestep,'gravity',m.opt.gravity.tolist())
for i in range(10): mujoco.mj_step(m,d)
assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
print('ten_substeps_finite',d.time,'step_warnings',sum(int(w.number) for w in d.warning))
'''
    models=[p for p in sorted((D/'source/dexgym/assets/marvin_wuji').rglob('*.xml')) if 'wholebody' in p.name or 'dual_arm_marvin' in p.name]
    model=next((p for p in models if p.name=='dual_arm_marvin_wholebody_s0.xml'),models[0] if models else None)
    return dict(raw_episode_slots=len(raw_dirs),resolved_episode_slots=sum(p.is_dir() for p in raw_dirs),
                sample_episode=str(chosen),resolved_root=str(chosen.resolve()),sample_episode_files=files,
                inspected_bags=[dict(path=str(p),sha256=digest(p),inspect=cmd(ros_command+[p],timeout=45))
                                for p in [*recordings[:3], official]],
                model_catalog=[str(p) for p in models],model=dict(path=str(model),sha256=digest(model),
                    probe=cmd([PY,'-c',cpu_witness,model])) if model else None)


def source():
    targets=[(D,P+'wholebody_s0/'+f,p) for f,p in (
        ('pipeline.py','align_episode|retarget|split_episodes|export_library|raw_episode|reference_one'),
        ('bag.py','mcap|from.*deserialize|def read_episode|def manus_keypoint_order'),
        ('sync.py','def causal_index|def align_episode|interpolate\\(|brackets|query.*stamp'),
        ('intent.py','class HumanIntent|self\\.|return IntentTargets'),
        ('ppo_task.py','TanhGaussian|OnPolicyRunner|ResidualEnv|apply_newton|def learn|def main|time_out|log_prob|process_env_step|save\\('),
        ('domain.py','robust|def apply|random|multiply|bias|mass|gain|friction|delay'),
        ('residual_env.py','apply_newton|domain|reset|truncation|termination'),
        ('evaluate_residual.py','def lag_ms|class NativeEvaluator|def eval_case|reference|robust|lag_ms') )]
    targets += [(R,P+'wholebody_s0_1/'+f,p) for f,p in (
        ('targets.py','class |TargetMapper|def |Intent|project|coupl'),('kinematics.py','class |jacobian|mj_jac|differential'),
        ('control.py','class |def |clamp|limit|arm'),('finger_reward.py','def |class |thumb|tip|pinch'),
        ('bank.py','class |def |rsi|egopro'),('prepare.py','def |align|manifest|q_ref'))]
    targets += [(R,P+'wholebody_s0_2/'+f,p) for f,p in (
        ('interface.py','ACTION|OBS|DIM|CONTRACT|class '),('task.py','class |action|command|reward'),
        ('hand_shape.py','def |class |pinch|right|left'))]
    targets += [(W,P+'wholebody_s04/'+f,p) for f,p in (
        ('__init__.py','GOAL|ARM|ACTION|CONTRACT|HISTORY|TARGET'),
        ('task.py','class |def |return torch.cat|\.65|clamp|q_ref|rsi|hand_reference'),
        ('hand_points.py','def |GOAL_POINTS|joint_|world|tip'),
        ('aligned_physics.py','class |def |filter|gravity|substeps'),
        ('aligned_task.py','class |def |gravity|smooth|terminal|rate_change'),
        ('train_aligned.py','def |WristSampling|wm_only|horizon|test|best|score\\('),
        ('sweep.py','def |hard_stop|invalid|p95|reset|frame|hit_15'))]
    targets += [(J,'aligned_task.py','class |def |random|rate_change|reset'),
                (J,'train.py','def |WristSampling|wm_only|horizon|test|best|score\\('),
                (R,P+'wholebody_s04/task.py','def |class |1\.025|110\.5|\.65|\.35|task_id|hand_ref'),
                (R,P+'wholebody_s04/campaign.py','^def |egopro|teleop|rsi|source_kind'),
                (D,P+'marvin54_s0/task.py','class |def |ACTION|Jacobian|jacobian|Physics'),
                (D,P+'marvin54_s0/env.py','class |def |random|reset|step|done'),
                (D,'source/dexgym/algorithms/supermimic/dex_reference/engine.py','class |def |imagine|replay|start|transition')]
    targets += [(T,f,p) for f,p in (
        ('src/whole_body_teleop/launch/whole_body_teleop.launch.py','Node\\(|package=|parameters='),
        ('src/whole_body_teleop/src/teleop_controller_node.cpp','control_rate_hz_|left_tracker_topic|right_tracker_topic|joint_command|void .*tick|teleop_core_'),
        ('src/whole_body_teleop/src/teleop_core.cpp','::.*step|::.*Tick|registration_|solveIk|solve_ik|::process|::tick'),
        ('src/tianji_real_driver/src/real_driver_node.cpp','joint_command|create_wall_timer|on_command'),
        ('src/wuji_hand2_real_driver/config/real.yaml','[^#]*(topic|rate|retarget|sdk)'),
        ('src/whole_body_teleop/config/bimanual_position_real_no_head.yaml','control_rate|tracker_topic|command_topic|ik'))]
    hand_cpp=next(iter(sorted((T/'src/wuji_hand2_real_driver/src').glob('*node.cpp'))),None)
    if hand_cpp:
        targets.append((T,str(hand_cpp.relative_to(T)),'create_subscription|retarget|command|publish|timer|manus'))
    return {str(root/f):excerpt(root,f,p,24) for root,f,p in targets}


def tests():
    files=['tests/assets/test_marvin_binding.py','tests/assets/test_marvin_contract.py','tests/assets/test_marvin_controller.py']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(D/'source'),CUDA_VISIBLE_DEVICES='')
    return cmd([PY,'-m','pytest','-p','no:cacheprovider','-q',*files],timeout=55,env=env)


def followup():
    raw=D/'data/raw/wholebody_s0/2026-11-12/18e74c646013f1e6-2156b79bc2482857'
    script='''source /opt/ros/jazzy/setup.bash || exit 1
source "$HOME/teleop_ws/install/local_setup.bash" || exit 1
for overlay in "$HOME/dex_data_platform/install/local_setup.bash" "$HOME/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash"; do
 if [ -f "$overlay" ]; then source "$overlay"; fi
done
python3 - "$1" <<'PY'
import sys, pathlib, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
for tag in ('manus_left','manus_right','tracker_left','tracker_right'):
 path=next((pathlib.Path(sys.argv[1])/'streams'/tag).glob('*.mcap'))
 r=rosbag2_py.SequentialReader()
 r.open(rosbag2_py.StorageOptions(uri=str(path),storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
 types={t.name:t.type for t in r.get_all_topics_and_types()}
 found=set()
 while r.has_next() and len(found)<len(types):
  topic,value,stamp=r.read_next()
  if topic in found: continue
  cls=get_message(types[topic]); msg=deserialize_message(value,cls);found.add(topic)
  print(tag, topic, types[topic], stamp, sorted(msg.get_fields_and_field_types()))
  if hasattr(msg,'nodes'): print('nodes',len(msg.nodes),sorted(msg.nodes[0].get_fields_and_field_types()) if msg.nodes else [])
 assert len(found)==len(types),(tag,types,found)
PY'''
    bag_witness=cmd(['bash','-lc',script,'p00',raw],timeout=40)
    retry_env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    adapted=cmd(['bash','scripts/supermimic/host.sh','--pytest','-p','no:cacheprovider','-q',
                 'tests/assets/test_marvin_binding.py','tests/assets/test_marvin_contract.py',
                 'tests/assets/test_marvin_controller.py'],timeout=55,env=retry_env)
    isolated=cmd([PY,'-m','pytest','-p','no:cacheprovider','-q','tests/assets/test_marvin_binding.py'],
                 env=dict(retry_env,PYTHONPATH=str(D/'source')))
    profile=J.parent/'hardware_servo_calibration_20261125/servo_profile.json'
    launch=J/'launch.sh'
    manifest=Path(json.loads((J/'train/provenance.json').read_text())['bank'])
    bank=json.loads(manifest.read_text())
    model_lines=[(i,l.strip()) for i,l in enumerate(launch.read_text().splitlines(),1) if 'MODEL' in l]
    retarget_files=list((HOME/'manus_wuji_real').glob('**/retarget_node.cpp'))+list(T.glob('src/**/retarget_node.cpp'))
    return dict(bag_witness=bag_witness,wrapped_tests=adapted,isolated_test=isolated,model_launch=model_lines,
                retarget_sources={str(path):excerpt(path.parent,path.name,'retarget|manus_glove|hand/command',22)
                                  for path in retarget_files[:5]},servo_profile=json.loads(profile.read_text()),
                bank=dict(path=str(manifest),sha256=digest(manifest),keys=list(bank),
                          scalars={k:v for k,v in bank.items() if isinstance(v,(str,int,float,bool))},
                          dictionaries={k:list(v)[:32] for k,v in bank.items() if isinstance(v,dict)},
                          list_shapes={k:[len(v),list(v[0]) if v and isinstance(v[0],dict) else None]
                                       for k,v in bank.items() if isinstance(v,list)}))


mode=sys.argv[1]
os.environ.update(PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',LC_ALL='C')
result={'section':mode,'observed':datetime.now().astimezone().isoformat(),
        'result':{'raw':data,'source':source,'tests':tests,'followup':followup}[mode]()}
payload=json.dumps(result,indent=2,sort_keys=True).encode()
if '--transport' in sys.argv:
    encoded=base64.b64encode(gzip.compress(payload,mtime=0)).decode()
    for i in range(0,len(encoded),64):
        print('Z05 '+encoded[i:i+64])
else:
    print(payload.decode())
