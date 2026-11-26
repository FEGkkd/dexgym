"""Bind E0 runtime and E1 launch authority without rewriting the D1 approval."""
import importlib
from importlib.metadata import PackageNotFoundError, distribution
import os
from pathlib import Path
import platform
import subprocess
import sys
import zlib

from .hmb_build import USE, SYNC_HASH, lock_sync
from .p01d_freeze import SDK, RUN_BASE, runtime_provenance
from .registry_gate import digest
from .sync_audit import DOCS, read


D_DIR=DOCS/'evidence/P01_D/products'
D_RECEIPT=DOCS/'evidence/P01_D/review_receipt.json'
BASE='34e44b6debcd4bb6f10be53a36dc8fec8bc4e7b2'
D_SHA='f7f2eeba2cfc8adc9d1649618f353956470eb477a6a99448bdbe7332dd5a6d81'
D_RECEIPT_SHA='40bf56f01337d0ceb32f8977efc80154e40144bbc949becbb7050d12963bed93'
EVIDENCE=DOCS/'evidence/P01_E_E0'
PLAN=EVIDENCE/'plan.json'
SOURCE_TREE=Path('source/dexgym/tasks/direct/wholebody_s05/data')
PACKAGES=('numpy','scipy','mcap','mcap-ros2-support','PyYAML','zstandard','lz4')
DEBIAN_PACKAGES={'numpy':'python3-numpy','scipy':'python3-scipy','PyYAML':'python3-yaml',
                 'zstandard':'python3-zstandard','lz4':'python3-lz4'}
DEBIAN_IMPORTS={'PyYAML':'yaml'}


def audited_d():
    if digest(D_DIR/'frozen_config.json')!=D_SHA or digest(D_RECEIPT)!=D_RECEIPT_SHA:
        raise ValueError('Original D1 freeze or approved receipt changed')
    cfg=read(D_DIR/'frozen_config.json')
    if cfg['allowed_use']!=USE or cfg['sync_source_sha256']!=SYNC_HASH:
        raise ValueError('D1 use or synchronizer proof differs')
    if any(digest(D_DIR/file)!=h for file,h in cfg['artifact_sha256'].items()):
        raise ValueError('D1 frozen sidecars changed')
    return cfg


def package_record(name):
    try:
        info=distribution(name)
    except PackageNotFoundError:
        if name in {'zstandard','lz4'}:
            return {'available':False}
        raise
    from hashlib import sha256
    files={}
    debian=None
    if info.files is None:
        debian=DEBIAN_PACKAGES.get(name)
        if debian is None:
            raise ValueError(f'No trusted package inventory: {name}')
        paths=subprocess.check_output(['/usr/bin/dpkg-query','--listfiles',debian],text=True).splitlines()
        files={line:Path(line) for line in paths}
        origin=Path(importlib.import_module(DEBIAN_IMPORTS.get(name,name)).__file__).resolve()
        if not origin.is_relative_to(Path(info.locate_file('')).resolve()) or str(origin) not in files:
            raise ValueError(f'Imported {name} is outside the installed Debian package')
    else:
        files={str(rel):Path(info.locate_file(rel)) for rel in info.files if '..' not in rel.parts}
    h=sha256()
    count=0
    for label,f in sorted(files.items()):
        if '__pycache__' not in f.parts and f.suffix!='.pyc' and f.is_file():
            h.update(f'{label}\0{digest(f)}\n'.encode())
            count+=1
    if not count:
        raise ValueError(f'No package file provenance: {name}')
    return {'available':True,'version':info.version,'files':count,'debian_package':debian,
            'distribution_tree_sha256':h.hexdigest()}


def lock_inputs(registry_path, cfg):
    lock_sync()
    if digest(registry_path/'registry.jsonl')!=cfg['registry_sha256']:
        raise ValueError('Source registry differs from D1')
    runtime=runtime_provenance()
    setup=[Path('/opt/ros/jazzy/setup.bash'),Path('/home/ubuntu/teleop_ws/install/local_setup.bash'),
           Path('/home/ubuntu/MANUS_STEAMVR_ws/dex_data_platform-cpp-collection/install/local_setup.bash')]
    files=sorted(SOURCE_TREE.rglob('*.py'))+[
        Path('source/dexgym/__init__.py'),Path('source/dexgym/tasks/__init__.py'),
        Path('source/dexgym/tasks/direct/__init__.py'),Path('source/dexgym/tasks/direct/wholebody_s05/__init__.py'),
        DOCS/'evidence/P01_B_review/topic_dependency_matrix.json',
        DOCS/'evidence/P01_B/sample_plan.json',PLAN,EVIDENCE/'run_e0.sh']
    loaded={name:Path(importlib.import_module(name).__file__).resolve()
            for name in ('mcap','mcap_ros2','numpy','scipy','yaml')}
    return dict(schema='s05.p01e.execution_lock.v1',approved_d_commit=BASE,approved_d_config_sha256=D_SHA,
                approved_d1_receipt_sha256=D_RECEIPT_SHA,algorithm_sha256=SYNC_HASH,
                registry_sha256=cfg['registry_sha256'],raw_whitelist_sha256=cfg['artifact_sha256']['source_whitelist.json'],
                source_sha256={str(p):digest(p) for p in files if p.is_file()},
                python_executable_sha256=digest(Path(sys.executable).resolve()),runtime=runtime,
                platform={'machine':platform.machine(),'python_abi':sys.implementation.cache_tag,
                          'libc':list(platform.libc_ver()),'zlib':zlib.ZLIB_RUNTIME_VERSION},
                package_distributions={name:package_record(name) for name in PACKAGES},
                import_sha256={name:{'resolved':str(p),'sha256':digest(p)} for name,p in loaded.items()},
                sdk={'resolved':SDK,'sha256':digest(Path(SDK))},
                ros_setup_sha256={str(p):digest(p) for p in setup},ros_distro=os.environ.get('ROS_DISTRO'))


def candidate(cfg, lock_hash):
    return dict(schema='s05.p01e.execution_candidate.v1',approval='AWAITING_HUMAN',allowed_use=USE,
                parent_d_config_sha256=D_SHA,parent_d_receipt_sha256=D_RECEIPT_SHA,
                implementation_lock_sha256=lock_hash,registry_sha256=cfg['registry_sha256'],
                artifact_sha256=cfg['artifact_sha256'],parameters=cfg['parameters'],
                numerical_qc=cfg['numerical_qc'],
                sync_source_sha256=SYNC_HASH,plan_sha256=digest(PLAN))


def verify_lock(registry, cfg_path, lock_path):
    old=audited_d()
    if read(cfg_path)!=candidate(old,digest(lock_path)):
        raise ValueError('E execution candidate changes a frozen D1 semantic field')
    if read(lock_path)!=lock_inputs(registry,old):
        raise ValueError('Actual runtime or implementation differs from the E execution lock')
    return old


def authorize(scope, output, cfg_hash, lock_hash):
    path=DOCS/'phase_state.json'
    state=read(path)
    events=state['p01_subphase_events']
    d=next((x for x in reversed(events) if x['phase']=='P01-D' and x['to']=='ACCEPTED'),{})
    e=next((x for x in reversed(events) if x['phase']=='P01-E' and x['to']=='IN_PROGRESS'),{})
    if (state['p01_subphases']['P01-D']!='ACCEPTED' or state['p01_subphases']['P01-E']!='IN_PROGRESS' or
            d.get('frozen_config_sha256')!=D_SHA or d.get('review_receipt_sha256')!=D_RECEIPT_SHA or
            d.get('reviewed_commit')!=BASE or d.get('approval',{}).get('type')!='human'):
        raise ValueError('D1 human approval and E start are required')
    relative=output.resolve().relative_to(RUN_BASE)
    if len(relative.parts)<2:
        raise ValueError('Use an isolated output directory within its run')
    if scope=='E0_TRANSIENT_NON_PRODUCTION':
        if (e.get('scope')!='E0_PREFLIGHT' or e.get('run_id')!=relative.parts[0] or
                not relative.parts[0].startswith('P01-E-E0-')):
            raise ValueError('E0 output or authorization has wrong scope')
        return
    full=state.get('p01e_production_authorization') or {}
    committed=subprocess.run(['git','show',f'HEAD:{path}'],capture_output=True,check=True).stdout
    approval_ref=full.get('approval_ref','')
    if (committed!=path.read_bytes() or not relative.parts[0].startswith('P01-E-') or
            relative.parts[0].startswith('P01-E-E0-') or full.get('type')!='human' or
            full.get('candidate_config_sha256')!=cfg_hash or full.get('execution_lock_sha256')!=lock_hash or
            full.get('run_id')!=relative.parts[0] or not approval_ref or
            not (DOCS/approval_ref).resolve().is_relative_to(DOCS.resolve()) or
            not (DOCS/approval_ref).is_file()):
        raise ValueError('E1 requires separate committed human approval of the exact new execution lock and configuration')
