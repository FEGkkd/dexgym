"""Read-only P00 review witness. Bounded decoding, no ROS node or data writer."""
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


HOME = Path.home()
CHECKOUT = HOME / 'dexgym_worktrees/s0-5'
DEXGYM = HOME / 'dexgym'
TELEOP = HOME / 'teleop_ws'
COLLECTOR = HOME / 'MANUS_STEAMVR_ws/dex_data_platform-cpp-collection'
OUTPUT = HOME / 'dexgym_data/s0-5/p01'
BASE = 'cc05f09847254ddae3b76581fb9e1feed4f31065'
ROOT = DEXGYM / 'data/processed'
NATIVE_ID = '18e74c646013f1e6-2156b79bc2482857'
NATIVE = DEXGYM / f'data/raw/wholebody_s0/2026-11-12/{NATIVE_ID}'
MERGED = TELEOP / 'data/teleop_no_camera_20261016_133930/teleop_no_camera_20261016_133930_0.mcap'
REPLAY = TELEOP / 'data/s0_sim_20261117_153201/s0_sim_20261117_153201_0.mcap'
TOPICS = ('manus_glove', '/vive/raw/', '/vive/tracker/', '/left_arm/', '/right_arm/',
          '/left_hand/', '/right_hand/', '/tianji/', '/tianji_real_driver/',
          '/dex/collection/observations/', '/whole_body_teleop/')
NATIVE_STREAMS = ('manus_left','manus_right','tracker_left','tracker_right',
                  'tracker_left_raw','tracker_right_raw','tracker_safety')


def cmd(argv, timeout=20, binary=False):
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout)
        output = (hashlib.sha256(result.stdout).hexdigest() if binary else
                  result.stdout.decode('utf-8', 'replace'))
        return dict(argv=list(map(str, argv)), exit=result.returncode, stdout=output,
                    stderr=result.stderr.decode('utf-8', 'replace'))
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=list(map(str, argv)), status='BLOCKED', error=str(error))


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def witness(path):
    real = path.resolve(strict=True)
    return dict(path=str(path), realpath=str(real), bytes=real.stat().st_size, sha256=sha(real))


def git_identity(path):
    git = ['git', '-C', str(path)]
    return dict(head=cmd(git + ['rev-parse','HEAD']), branch=cmd(git + ['branch','--show-current']),
                tracked_paths=cmd(git + ['diff','--name-status','HEAD']),
                tracked_diff_sha256=cmd(git + ['diff','--binary','HEAD'],binary=True),
                status=(cmd(git + ['status','--porcelain=v1','-uall']) if path==CHECKOUT else None))


def module_witness(name):
    try:
        module = importlib.import_module(name)
        return dict(status='PASS', version=getattr(module,'__version__',None), file=witness(Path(module.__file__)))
    except (ImportError, OSError, TypeError) as error:
        return dict(status='BLOCKED', error=str(error))


def interfaces():
    """Hash message definitions and installed type support used by the parser."""
    from ament_index_python.packages import get_package_prefix

    entries = {}
    for package, names in {
        'manus_ros2_msgs':('ManusGlove','ManusRawNode'),
        'tianji_interfaces':('TrackerPose','SafetyStatus','RobotState','JointCommand','TeleopState'),
        'manus_wuji_interfaces':('HandCommand','HandState','HandDiagnostics'),
        'dex_collection_interfaces':('SampleObservation','CollectionStatus','TopicCatalogue'),
        'sensor_msgs':('JointState',),
        'geometry_msgs':('PoseStamped','TwistStamped','WrenchStamped'),
        'diagnostic_msgs':('DiagnosticArray',),
    }.items():
        prefix = Path(get_package_prefix(package)).resolve()
        msgdir = prefix/'share'/package/'msg'
        definitions = [witness(p) for p in sorted(msgdir.glob('*.msg'))]
        entry = dict(prefix=str(prefix), requested=list(names), definitions=definitions,
                     manifest=witness(prefix/'share'/package/'package.xml'))
        entry['typesupport'] = [witness(p) for p in sorted((prefix/'lib').glob(f'lib{package}__rosidl_typesupport*.so'))
                                if p.is_file()]
        entries[package] = entry
    return entries


def environment():
    modules = ('dexgym','dexgym.tasks.direct.wholebody_s0.bag',
               'dexgym.tasks.direct.wholebody_s0.sync','numpy','scipy','yaml',
               'rosbag2_py','rclpy.serialization','rosidl_runtime_py.utilities','mcap','mcap.reader',
               'gymnasium','cloudpickle','zstandard','lz4','mcap_ros2.decoder','mcap_ros2._dynamic',
               'manus_ros2_msgs.msg','tianji_interfaces.msg','manus_wuji_interfaces.msg',
               'dex_collection_interfaces.msg')
    found = {n:module_witness(n) for n in modules}
    imports = [v['file']['realpath'] for v in found.values() if 'file' in v]
    foreign = [p for p in sys.path+imports if str(DEXGYM/'data') in p or '/tmp/dexgym-' in p or
               'marvinsupermimic' in p or 'runtime_overrides' in p]
    wrong_dexgym = [n for n,v in found.items() if n.startswith('dexgym') and
                    ('file' not in v or not Path(v['file']['realpath']).is_relative_to(CHECKOUT/'source'))]
    output = dict(path=str(OUTPUT), exists=OUTPUT.is_dir(), resolved=str(OUTPUT.resolve()),
                  writable=os.access(OUTPUT,os.W_OK), entries=sorted(x.name for x in OUTPUT.iterdir()) if OUTPUT.is_dir() else [],
                  distinct_from_source=not OUTPUT.resolve().is_relative_to(CHECKOUT) and
                  not OUTPUT.resolve().is_relative_to(DEXGYM/'data'))
    if OUTPUT.is_dir():
        fs = os.statvfs(OUTPUT)
        output.update(available_bytes=fs.f_bavail*fs.f_frsize,filesystem=str(OUTPUT.stat().st_dev),
                      checkout_filesystem=str(CHECKOUT.stat().st_dev))
    platform_packages = [f'ros-jazzy-{n}' for n in ('rclpy','rosbag2-py','rosbag2-storage-mcap','rosidl-runtime-py')]
    platform_packages += ['python3','python3-numpy','python3-scipy','python3-yaml','python3-lz4']
    sync = importlib.import_module('dexgym.tasks.direct.wholebody_s0.sync')
    declarations = {}
    for name in ('sample_indices','interpolate','causal_index','align_episode'):
        lines, start = inspect.getsourcelines(getattr(sync,name))
        declarations[name] = dict(start=start, lines=[(i,line.strip()) for i,line in enumerate(lines,start)
            if any(x in line for x in ('track.header','interpola','causal','searchsorted','brackets','duplicate_source'))])
    return dict(executable=sys.executable, python=sys.version, cwd=os.getcwd(), sys_path=sys.path,
                distributions={name:importlib.metadata.version(name) for name in
                    ('mcap','mcap-ros2-support','gymnasium','cloudpickle','Farama-Notifications',
                     'zstandard','lz4','numpy','scipy','PyYAML')},
                overlays={n:os.environ.get(n) for n in ('AMENT_PREFIX_PATH','PYTHONPATH','LD_LIBRARY_PATH','ROS_DISTRO')},
                overlay_setup_files=[witness(p) for p in (Path('/opt/ros/jazzy/setup.bash'),
                                         TELEOP/'install/local_setup.bash',COLLECTOR/'install/local_setup.bash')],
                imports=found, wrong_dexgym=wrong_dexgym, foreign=foreign, output=output,
                interfaces=interfaces(), sync_declarations=declarations,
                repositories={str(p):git_identity(p) for p in (CHECKOUT,DEXGYM,TELEOP,COLLECTOR)},
                commands=[cmd(['dpkg-query','-W',*platform_packages]),cmd(['df','-B1',str(OUTPUT),str(CHECKOUT)]),
                          cmd(['free','-b']), cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version',
                                                 '--format=csv,noheader'])])


def meta(file):
    import yaml
    data = yaml.safe_load(file.read_text())
    if 'rosbag2_bagfile_information' in data:
        info = data['rosbag2_bagfile_information']
        return dict(file=witness(file), version=info.get('version'), storage=info.get('storage_identifier'),
                    duration_ns=info.get('duration',{}).get('nanoseconds'), counts={x['topic_metadata']['name']:
                    x['message_count'] for x in info.get('topics_with_message_count',[])})
    return dict(file=witness(file), keys=sorted(data),
                top_level_scalars={k:v for k,v in data.items() if isinstance(v,(str,int,float,bool))},
                collections={k:(list(v) if isinstance(v,dict) else len(v)) for k,v in data.items()
                             if isinstance(v,(dict,list))})


def field_summary(msg):
    data = dict(fields=msg.get_fields_and_field_types())
    header = getattr(msg,'header',None)
    if header is not None:
        data['frame_id'] = header.frame_id
    for f in ('position','velocity','effort','positions','velocities','currents','joint_names','names','arms','raw_nodes'):
        v = getattr(msg,f,None)
        if v is not None and hasattr(v,'__len__'):
            data[f+'_length'] = len(v)
    nodes = getattr(msg,'raw_nodes',None)
    if nodes is not None:
        data['raw_node_count'] = int(msg.raw_node_count)
        data['topology'] = [[int(x.node_id),int(x.parent_node_id)] for x in nodes]
        data['node_fields'] = nodes[0].get_fields_and_field_types() if len(nodes) else {}
    for f in ('valid','safe','quality','safety_code','glove_id'):
        v = getattr(msg,f,None)
        if isinstance(v,(int,float,bool,str)):
            data[f] = v
    return data


def stamp_values(msg, log_ns):
    stamps = {'reader_log_ns':int(log_ns)}
    header = getattr(msg,'header',None)
    if header is not None:
        stamps['ros_header_ns'] = 1_000_000_000*int(header.stamp.sec)+int(header.stamp.nanosec)
    for k in ('source_timestamp_us','receive_timestamp_us','receive_ros_timestamp_ns',
              'host_observed_monotonic_ns','host_realtime_ns','source_sequence_id','episode_sample_index'):
        v = getattr(msg,k,None)
        if isinstance(v,int):
            stamps[k] = v
    return stamps


def stats(rows):
    import numpy as np
    fields = {}
    names = {k for row in rows for k in row}
    for k in sorted(names):
        v = np.array([row[k] for row in rows if k in row],dtype=np.int64)
        diff = np.diff(v[v>0])
        fields[k] = dict(count=len(v),positive=int((v>0).sum()), min=int(v.min()),max=int(v.max()),
                         repeats=int((diff==0).sum()),backwards=int((diff<0).sum()),
                         step_p50=float(np.median(diff)) if len(diff) else None)
    return fields


def bounded_mcap(path, per_topic=20, scan_limit=1800):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path),storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr','cdr'))
    catalog = {t.name:t.type for t in reader.get_all_topics_and_types()}
    topics = [n for n in catalog if any(n.startswith(k) or k=='manus_glove' and k in n for k in TOPICS)]
    rows = defaultdict(list)
    fields = {}
    failed = {}
    scanned = 0
    for _ in range(scan_limit):
        if not reader.has_next():
            break
        topic,payload,timestamp = reader.read_next()
        scanned += 1
        if topic not in topics or len(rows[topic])>=per_topic or topic in failed:
            continue
        try:
            message_type = get_message(catalog[topic])
            msg = deserialize_message(payload,message_type)
            if topic not in fields:
                fields[topic] = field_summary(msg)
                fields[topic]['python_type_file'] = witness(Path(inspect.getfile(message_type)))
            rows[topic].append(stamp_values(msg,timestamp))
        except Exception as error:
            failed[topic] = f'{type(error).__name__}: {error}'
        if all(len(rows[n])>=per_topic or n in failed for n in topics):
            break
    return dict(path=str(path), size=path.stat().st_size, format_catalog=catalog, scanned=scanned,
                scan_limit=scan_limit, per_topic_limit=per_topic, failures=failed,
                decoded={n:dict(count=len(v),first=v[0],last=v[-1],timestamps=stats(v),**fields[n])
                         for n,v in rows.items() if v}, no_sample=sorted(set(topics)-set(fields)),
                publish_time='UNKNOWN in this API witness', controller_receive_time='UNKNOWN')


def raw_mcap_clocks(path, limit=32):
    """Read both wire-level MCAP clocks for a bounded prefix."""
    from itertools import islice
    from mcap.reader import make_reader

    with path.open('rb') as handle:
        reader = make_reader(handle)
        rows = [dict(topic=ch.topic, sequence=msg.sequence, log_time_ns=msg.log_time,
                     publish_time_ns=msg.publish_time) for _,ch,msg in
                     islice(reader.iter_messages(log_time_order=False),limit)]
    return dict(path=str(path), limit=limit, count=len(rows),
                unequal=sum(r['log_time_ns']!=r['publish_time_ns'] for r in rows),
                zero_publish=sum(r['publish_time_ns']==0 for r in rows), rows=rows)


def embedded_schema_telemetry(path, limit=12):
    from itertools import islice
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    topics = [f'/{side}_{channel}' for side in ('left','right') for channel in
              ('arm/joint_states','arm/ee_pose','arm/ee_twist','arm/wrench',
               'hand/command','hand/state','hand/diagnostics')]
    topics += ['/tianji/robot_state','/tianji/joint_command','/tianji/teleop_state','/tianji/safety']
    factory = DecoderFactory()
    evidence = {}
    with path.open('rb') as source:
        reader = make_reader(source)
        for topic in topics:
            rows = []
            sample = {}
            try:
                for schema,ch,record in islice(reader.iter_messages(topics=[topic],log_time_order=False),limit):
                    decode = factory.decoder_for(ch.message_encoding,schema)
                    if decode is None:
                        raise ValueError('Recorded schema could not supply a CDR decoder')
                    msg = decode(record.data)
                    if not sample:
                        slots = tuple(getattr(msg,'__slots__',()))
                        sample = dict(fields=list(slots),schema_name=schema.name,schema_encoding=schema.encoding,
                                      schema_sha256=hashlib.sha256(schema.data).hexdigest(),
                                      array_lengths={name:len(v) for name in slots if
                                          isinstance(v:=getattr(msg,name,None),(list,tuple))})
                    rows.append(stamp_values(msg,record.log_time))
                evidence[topic] = dict(count=len(rows),status='PASS' if rows else 'NO_SAMPLE',
                                       timestamps=stats(rows),**sample)
            except Exception as error:
                evidence[topic] = dict(count=len(rows),status='FAILED',error=f'{type(error).__name__}: {error}',**sample)
    return dict(per_topic_limit=limit,topics=evidence)


def samples():
    import numpy as np

    streams = {s:bounded_mcap(next((NATIVE/'streams'/s).glob('*.mcap'))) for s in NATIVE_STREAMS}
    derived = ROOT/f'wholebody_s0/2026-11-12/episodes/{NATIVE_ID}'
    with np.load(derived/'parsed.npz',allow_pickle=False) as z:
        arrays = {n:dict(shape=list(z[n].shape),dtype=str(z[n].dtype)) for n in z.files if
                  n in ('manus_left__pose','manus_right__pose','tracker_left__pose','tracker_right__pose',
                        'manus_left__header_ns','manus_left__publish_ns','manus_left__log_ns',
                        'manus_left__receive_ros_ns','manus_left__sdk_receive_us','manus_left__source_stamp')}
    with np.load(derived/'aligned.npz',allow_pickle=False) as z:
        aligned_frames = len(z['time_ns'])
        aligned_nbytes = sum(z[n].nbytes for n in z.files)
    clocks = {s:raw_mcap_clocks(Path(v['path'])) for s,v in streams.items()}
    clocks['merged'] = raw_mcap_clocks(MERGED)
    return dict(native_manifest=meta(NATIVE/'manifest.yaml'),native_effective_config=meta(NATIVE/'effective_config.yaml'),
                native_streams=streams, native_path_real=str(NATIVE.resolve()),
                merged_meta=meta(MERGED.parent/'metadata.yaml'), merged=bounded_mcap(MERGED),
                merged_embedded_schema=embedded_schema_telemetry(MERGED),
                wire_clocks=clocks,
                replay_evidence=meta(REPLAY.parent/'metadata.yaml'), replay_check=cmd(['ros2','bag','info','-s','mcap',str(REPLAY)]),
                parsed_schema=arrays, size_witness=dict(aligned_frames=aligned_frames, aligned_array_nbytes=aligned_nbytes,
                    parsed_file_bytes=(derived/'parsed.npz').stat().st_size,aligned_file_bytes=(derived/'aligned.npz').stat().st_size,
                    evidence_scope='existing S0 episode, not a P01 sizing benchmark'))


def historical():
    groups = {
        'wholebody_s0':[ROOT/'wholebody_s0'], 'wholebody_s0_1':[ROOT/'wholebody_s0-1'],
        'wholebody_s0_2':[ROOT/'wholebody_s0-2',DEXGYM/'wholebody_s0_2'],
        'marvin54':[HOME/'dexgym_data/marvin54_hand_centric_s0_20261112',
                    DEXGYM/'marvin54_hand_centric_s0_reports', ROOT/'marvin54_hand_centric_s0'],
        'wholebody_s04':[ROOT/'wholebody-s04'],
    }
    result = {}
    for family,roots in groups.items():
        found = []
        for root in roots:
            patterns = ('**/manifest.json','final_library/*.json','final_library/*/index.json',
                        'final_library/*/*.jsonl') if family=='marvin54' else ('**/manifest.json',)
            paths = sorted({path for pattern in patterns for path in root.glob(pattern)})[:8]
            for path in paths:
                try:
                    val = json.loads(path.read_text())
                    if not isinstance(val,dict):
                        continue
                    found.append(dict(**witness(path), keys=sorted(val),
                        identity={k:val[k] for k in ('schema','task','kind','rate_hz','source_manifest','source_manifest_sha256')
                                  if k in val and isinstance(val[k],(str,int,float))},
                        split_names=list(val.get('splits',{})),clip_count=len(val['clips']) if 'clips' in val else None))
                except (OSError,ValueError,TypeError) as error:
                    found.append(dict(**witness(path), error=str(error)))
        result[family] = dict(search_roots=[str(p) for p in roots],existing=[str(p) for p in roots if p.is_dir()],
                              samples=found,status='BOUNDED_INVENTORY' if found else 'UNRESOLVED')
    job = ROOT/'wholebody-s04/world_hand20_aligned_from_scratch_20261125'
    p = Path(json.loads((job/'train/provenance.json').read_text())['bank'])
    bank = json.loads(p.read_text())
    return dict(groups=result,s04_actual=dict(**witness(p),split_names=sorted(bank['splits']),
                keys=sorted(bank),job_state=witness(job/'train/state.json')),
                conclusion='candidate historical registries only; no episode gains final-test eligibility from this inventory')


if __name__ == '__main__':
    section = sys.argv[1]
    result = {'environment':environment,'samples':samples,'historical':historical}[section]()
    print(json.dumps(dict(schema='s05.readiness.witness.v1',section=section,
        observed=datetime.now().astimezone().isoformat(),result=result), ensure_ascii=True,indent=2))
