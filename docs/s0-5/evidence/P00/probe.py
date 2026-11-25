"""Read-only Ubuntu P00 audit. Run with the Wuji Python interpreter."""
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
MAIN = HOME / 'dexgym'
TELEOP = HOME / 'teleop_ws'
ALIGNED = Path('/tmp/dexgym-s04-main.Wwwqff99')
S04_ROOT = MAIN / 'data/processed/wholebody-s04'
RUNTIME = S04_ROOT / 'world_hand20_gravity_20000_20261125/checkout'
JOB = S04_ROOT / 'world_hand20_aligned_from_scratch_20261125'
PACKAGE = Path('source/dexgym/tasks/direct')


def command(args, cwd=None, timeout=25, env=None):
    try:
        result = subprocess.run(args, cwd=cwd, timeout=timeout, env=env,
                                encoding='utf-8', errors='replace', capture_output=True)
        return dict(argv=list(map(str, args)), cwd=str(cwd) if cwd else None,
                    exit=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=list(map(str, args)), status='BLOCKED', error=str(error))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def git_view(root):
    calls = [command(['git', *args], root) for args in (
        ['log', '-1', '--format=%H %s'], ['branch', '--show-current'],
        ['diff', '--name-status', 'HEAD'], ['status', '--porcelain=v1', '-uno'],
        ['ls-files', '--others', '--exclude-standard'])]
    names = calls[-1].get('stdout', '').splitlines()
    calls[-1]['stdout'] = dict(total=len(names), roots=sorted(set(n.split('/')[0] for n in names)),
                               path_list_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest())
    diff = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=root, capture_output=True)
    return dict(path=str(root), commands=calls, tracked_diff_sha256=hashlib.sha256(diff.stdout).hexdigest())


def environment():
    runtime_query = '''import importlib.metadata as m, importlib.util as u, sys, json
print(sys.executable, sys.version)
ds = m.packages_distributions()
mods = 'torch mujoco warp mujoco_warp isaaclab newton rsl_rl numpy mcap rosbags pytest'.split()
for x in mods:
 s=u.find_spec(x); print(x, s.origin if s else None, [(d,m.version(d)) for d in ds.get(x,[])])
import torch,mujoco
print('imports',torch.__version__,torch.version.cuda,mujoco.__version__,torch.cuda.is_available())
'''
    overlay = MAIN/'artifacts/hand_s0/runtime_v1'
    overlay_paths = [overlay/f'dexgym-{p}' for p in ('numpy-fresh','isaaclab-fresh','scipy-fresh','omniclient-fresh','usd-overlay')]
    overlay_paths += [HOME/'dexgym_data/marvin_supermimic_m42/runtime_overrides', RUNTIME/'source']
    wrapped_env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,overlay_paths)), OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
    wrapper_query = ("import torch,sys;sys.path.append('/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages')\n"
                     + runtime_query + '''import importlib
for x in ('warp','mujoco_warp','rsl_rl','newton','isaaclab'):
 try:
  y=importlib.import_module(x);print('import',x,'OK',getattr(y,'__version__',None))
 except Exception as e:
  print('import',x,type(e).__name__,str(e))
''')
    isaac = HOME / 'isaacsim-6.0.1/bin/python'
    py = HOME / 'miniconda3/envs/wuji/bin/python'
    return dict(git=[git_view(root) for root in (MAIN, ALIGNED, RUNTIME, TELEOP)],
                host=[command(args) for args in (
                    ['date', '-Iseconds'], ['uname', '-r'], ['lscpu'], ['free', '-h'],
                    ['df', '-h', str(HOME), '/tmp'], ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv,noheader'])],
                interpreters=[command([py, '-c', runtime_query]), command([isaac, '-c', runtime_query]),
                              command([isaac, '-c', wrapper_query], RUNTIME, timeout=35, env=wrapped_env)],
                overlay_paths=list(map(str,overlay_paths)),
                guidance=[str(p.relative_to(root)) for root in (MAIN, TELEOP, ALIGNED)
                          for p in (root/'AGENTS.md', root/'docs/AGENTS.md', root/'source/AGENTS.md') if p.is_file()])


def schema(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        arrays = {}
        for key in archive.files:
            try:
                value = archive[key]
                arrays[key] = [str(value.dtype), list(value.shape)]
                if key in ('t_ns', 'stamp_ns', 'grid_ns', 'time_ns') and value.ndim == 1 and len(value) > 1:
                    arrays[key] += [int(np.min(np.diff(value))), int(np.max(np.diff(value)))]
            except ValueError as error:
                arrays[key] = str(error)
    return dict(path=str(path), sha256=sha(path), arrays=arrays)


def dataset():
    raw_root = TELEOP/'data'
    derived = MAIN/'data/processed/wholebody_s0'
    paths = sorted(raw_root.glob('*/*.mcap'))
    samples = []
    for prefix in ('teleop_no_camera_', 'bag_no_camera_', 's0_sim_'):
        path = next((p for p in paths if p.name.startswith(prefix)), None)
        if path is None:
            continue
        item = dict(path=str(path), size=path.stat().st_size, sha256=sha(path))
        try:
            from mcap.reader import make_reader
            with path.open('rb') as file:
                summary = make_reader(file).get_summary()
                counts = summary.statistics.channel_message_counts
                item['channels'] = [[channel.topic, summary.schemas[channel.schema_id].name,
                                     counts.get(i, 0)] for i, channel in summary.channels.items()]
        except Exception as error:
            item['reader_error'] = str(error)
        samples.append(item)
    paired = [p.parent for p in sorted(derived.rglob('parsed.npz')) if (p.parent/'aligned.npz').is_file()
              and (p.parent/'reference.npz').is_file()]
    metadata = []
    for p in sorted((MAIN/'data/raw/wholebody_s0').rglob('*.json'))[:7]:
        val = json.loads(p.read_text())
        metadata.append(dict(path=str(p), keys=sorted(val) if isinstance(val, dict) else type(val).__name__))
    library = []
    for p in sorted(derived.rglob('*.json')):
        if p.name in ('manifest.json','library.json','motion_library.json'):
            val = json.loads(p.read_text())
            library.append(dict(path=str(p), sha256=sha(p), keys=list(val),
                                groups={k:len(v) for k,v in val.items() if isinstance(v,(list,dict))}))
    return dict(raw_count=len(paths), raw_samples=samples, paired_episode_count=len(paired),
                expected_npz_counts={n:len(list(derived.rglob(n))) for n in ('parsed.npz','aligned.npz','reference.npz')},
                derived_samples=[schema(p/n) for p in paired[:1] for n in ('parsed.npz','aligned.npz','reference.npz')],
                episode_metadata=[str(p) for p in sorted(paired[0].glob('*.json'))] if paired else [],
                raw_manifest_samples=metadata, library=library)


def historical():
    state = json.loads((JOB/'train/state.json').read_text())
    weights = sorted({state['reviewed_iteration'],state['recommended_iteration']})
    rows = []
    for i in weights:
        actor = JOB/f'train/model_{i}_actor.pt'
        row = next(x for x in state['reviews'] if x['iteration'] == i)
        rows.append(dict(iteration=i, validation=row['validation'], test=row['test'], hold=row['hold'],
                         hard={k:v for k,v in row['hard'].items() if k!='details'}, actor_sha256=sha(actor)))
    settings = {k:v for k,v in state.items() if k != 'reviews' and not isinstance(v,(list,dict))}
    artifacts = [JOB/p for p in ('train.py','aligned_task.py','aligned_physics.py','train/provenance.json','backend_parity.json')]
    artifacts += [S04_ROOT/'hardware_servo_calibration_20261125/servo_profile.json',
                  ALIGNED/PACKAGE/'wholebody_s04/servo_profile.json']
    provenance = json.loads((JOB/'train/provenance.json').read_text())
    parity = json.loads((JOB/'backend_parity.json').read_text())
    return dict(settings=settings, reviews=rows, provenance=provenance, parity=parity,
                artifacts={str(p):sha(p) for p in artifacts if p.is_file()})


def source_index():
    tables = {
        'wholebody_s0': 'pipeline intent domain ppo_task evaluate_residual bag sync library retarget residual_env',
        'wholebody_s0_1': 'targets kinematics control finger_reward bank prepare',
        'wholebody_s0_2': 'interface hand_shape task training_arm_servo training_hand_servo',
        'wholebody_s04': 'task hand_points aligned_physics aligned_task train_aligned sweep',
        'marvin54_s0': 'task env physics warp_physics newton_physics',
    }
    result = {}
    for package, names in tables.items():
        for module in names.split():
            root = {'wholebody_s0_1': RUNTIME, 'wholebody_s0_2': RUNTIME,
                    'wholebody_s04': ALIGNED}.get(package, MAIN)
            file = root/PACKAGE/package/(module+'.py')
            if file.is_file():
                result[str(file)] = dict(sha256=sha(file), lines=[(i,line.strip()) for i,line in
                    enumerate(file.read_text().splitlines(),1) if re.match(r'^    (?:async )?def |^def |^class ',line)])
            else:
                result[str(file)] = dict(status='MISSING')
    return result


def main():
    os.environ.update(GIT_OPTIONAL_LOCKS='0', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1')
    name = sys.argv[1]
    sections = {'host':environment, 'data':dataset, 's04':historical, 'code':source_index}
    result = dict(section=name, observed=datetime.now().astimezone().isoformat(), result=sections[name]())
    payload = json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2).encode()
    if '--transport' in sys.argv:
        encoded = base64.b64encode(gzip.compress(payload,mtime=0)).decode()
        for i in range(0,len(encoded),64):
            print('Z05 '+encoded[i:i+64])
    else:
        print(payload.decode())


if __name__ == '__main__':
    main()
