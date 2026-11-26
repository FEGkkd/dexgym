"""Approved E1 Human Motion Bank and causal history access for S0-5."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .data.hmb_build import SCHEMA, SYNC_HASH, USE, read_partition
from .data.synchronize import CORE_STREAMS


MANIFEST_SHA = 'e63df2698feb812bc300593d9bf8d1ad3dcae1d2dc59aa4d3a3dfa96851d8074'
PLAN_SHA = '0f94b842e2558ec04da257aff94d469ae9b965968fff1a4593382fc2eea09404'
CONFIG_SHA = '6aeaa203e39bea392bc20d5fceee26ea5f85752d219824f98f96b2c852846317'
SCOPE = 'E1_PRODUCTION'
GEOMETRY = 'HMB_GEOMETRY_DEVELOPMENT'
SIMULATED = 'HMB_SIMULATED_CAUSAL'
DIAGNOSTIC = 'HMB_OFFLINE_DIAGNOSTIC'
PURPOSES = {GEOMETRY: 'causal', SIMULATED: 'causal', DIAGNOSTIC: 'offline'}
ACTOR_FIELDS = ('pose', 'node_mask', 'pose_valid', 'velocity_m_s', 'velocity_valid',
                'age_ns', 'velocity_age_ns', 'safety_permission', 'joint_valid')
STREAM_FIELDS = frozenset(ACTOR_FIELDS) - {'safety_permission', 'joint_valid'}
VERSION_FIELDS = ('clock_generation', 'calibration_id', 'topology_id')
FRAMES = {'sdk_common_pose': 'MANUS recorded SDK common, metres and XYZW',
          'manus_pose': 'per-side MANUS root/palm, metres and XYZW',
          'registered_wrist': 'recorded software registration, actual frame in source table',
          'raw_tracker': 'recorded device output frame in source table'}


def file_sha(path):
    with path.open('rb') as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    h = sha256()
    for block in iter(lambda: stream.read(2**20), b''):
        h.update(block)
    return h.hexdigest()


def checked_json(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'HMB identity mismatch: {path.name}')
    return json.loads(raw)


def actor_features(causal, tick):
    """Copy only current and past causal values. Segment membership is an external sampler concern."""
    if type(tick) is not int or not 4 <= tick < len(causal['pose']):
        raise ValueError('Five causal frames are required')
    indices = slice(tick - 4, tick + 1)
    return {name: (causal[name][indices, :len(CORE_STREAMS)] if name in STREAM_FIELDS else
                   causal[name][indices]).copy() for name in ACTOR_FIELDS}


def eligible_history(causal, tick, *, require_velocity=False):
    if type(tick) is not int or not 4 <= tick < len(causal['tick_ns']):
        return False
    i = slice(tick - 4, tick + 1)
    segment = causal['segment_id'][i]
    if (segment[0] < 0 or not np.all(segment == segment[0]) or
            not np.all(np.diff(causal['tick_ns'][i]) == 20_000_000) or
            not np.all(causal['safety_permission'][i] & causal['joint_valid'][i]) or
            not np.all(causal['pose_valid'][i, :len(CORE_STREAMS)])):
        return False
    for field in VERSION_FIELDS:
        versions = causal[field][i, :len(CORE_STREAMS)]
        if not np.all(versions == versions[0]):
            return False
    return not require_velocity or bool(np.all(causal['velocity_valid'][i, :len(CORE_STREAMS)]))


class HMBReader:
    """Read the signed E1 bank, including its frozen index and source contracts."""

    def __init__(self, manifest_path, plan_path, config_path, products, split, purpose):
        if split not in ('Train', 'Validation') or purpose not in PURPOSES:
            raise ValueError('Use one approved development split and purpose')
        self.archive = Path(products).resolve()
        self.purpose, self.split = purpose, split
        self.manifest = checked_json(Path(manifest_path), MANIFEST_SHA)
        if (self.manifest['scope'], self.manifest['schema'], self.manifest['allowed_use']) != (SCOPE, SCHEMA, USE):
            raise ValueError('Unexpected official HMB scope or schema')
        if self.manifest['execution_candidate_sha256'] != CONFIG_SHA or self.manifest['sync_sha256'] != SYNC_HASH:
            raise ValueError('Unexpected official HMB execution identity')
        plan = checked_json(Path(plan_path), self.manifest['original_plan_sha256'])
        if self.manifest['original_plan_sha256'] != PLAN_SHA:
            raise ValueError('The approved production plan hash differs')
        cfg = checked_json(Path(config_path), CONFIG_SHA)
        index = checked_json(self.archive/'build_index.json', self.manifest['index_sha256'])
        if index['scope'] != SCOPE or index['execution_candidate_sha256'] != CONFIG_SHA:
            raise ValueError('Production HMB index scope differs')
        if plan['scope'] != SCOPE or plan['config_sha256'] != CONFIG_SHA:
            raise ValueError('Production HMB plan scope differs')
        self._sources = {r['recording_id']: r for r in plan['sources']}
        if len({r['root'] for r in self._sources.values()}) != len(self._sources):
            raise ValueError('Source roots repeat in the approved plan')
        if [x['partition'] for x in index['partitions']] != plan['partitions']:
            raise ValueError('Production index and planned roots differ')
        # The immutable configuration binds all production sidecars.
        self.frozen_parameters = cfg['parameters']
        self._index = {x['partition']['key']: x for x in index['partitions']
                       if x['partition']['split'] == split and x['status'] == 'PUBLISHED'}
        self._resolved_sources = set()

    def keys(self):
        return tuple(self._index)

    def read(self, key, *, raw_assets=None):
        if key not in self._index:
            raise KeyError('Partition is outside the chosen split or unavailable')
        row = self._index[key]
        block, evidence = row['partition'], row['evidence']
        relative = Path(block['split'])/block['parent'].replace(':', '_')/block['label']
        target = (self.archive/relative).resolve()
        if (relative.as_posix() != evidence['path'] or not target.is_relative_to(self.archive) or
                file_sha(target/'manifest.json') != evidence['manifest_sha256']):
            raise ValueError('Partition identity or publication target differs')
        data, meta, sources = read_partition(target, PURPOSES[self.purpose])
        identity = meta['identity']
        raw = self._sources[block['recording_id']]
        if (identity['publication_scope'] != SCOPE or identity['execution_candidate_sha256'] != CONFIG_SHA or
                identity['verification'] != 'SIMULATED' or identity['allowed_use'] != USE or
                identity['development_split'] != self.split or
                identity['root_episode_id'] != raw['root'] or identity['source_manifest_sha256'] != raw['manifest_sha256'] or
                identity['parameters'] != self.frozen_parameters):
            raise ValueError('Partition source, schema or purpose differs')
        if raw_assets is not None:
            for src in sources:
                r = src['record']
                raw_path = Path(raw_assets[r['part_id']]).resolve()
                proof = (raw_path, r['asset_sha256'])
                if proof in self._resolved_sources:
                    continue
                if file_sha(raw_path) != r['asset_sha256']:
                    raise ValueError(f'Original MCAP SHA differs: {r["part_id"]}')
                self._resolved_sources.add(proof)
        return data, meta, sources, FRAMES.copy()

    def history(self, key, tick, *, require_velocity=False):
        if self.purpose != SIMULATED:
            raise ValueError('Actor history requires the simulated causal purpose')
        data, _, _, _ = self.read(key)
        if not eligible_history(data, tick, require_velocity=require_velocity):
            raise ValueError('Ineligible causal history window')
        return actor_features(data, tick)
