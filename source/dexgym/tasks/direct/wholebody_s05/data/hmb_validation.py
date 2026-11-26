"""Independent semantic checks at the HMB publication and recovery boundary."""
import re

import numpy as np

from .synchronize import CORE_STREAMS, SAFETY, STREAMS


SHA = re.compile(r'[0-9a-f]{64}')
E_SCOPES = {'E0_TRANSIENT_NON_PRODUCTION', 'E1_PRODUCTION'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _segments(data, meta, mode, provenance):
    n = len(data['tick_ns'])
    expected = np.full(n, -1, dtype=np.int32)
    occupied = 0
    cuts = set(meta.get('diagnostic_non_actor', {}).get('boundary_cut_indices', []))
    minimum = provenance.get('min_segment_ticks', 1)
    for sid, (lo, hi) in enumerate(meta['segments'][mode]):
        require(occupied <= lo < hi <= n and hi-lo >= minimum, 'Invalid or overlapping segment')
        require(np.all(data['joint_valid'][lo:hi] & data['safety_permission'][lo:hi]),
                'Segment includes invalid joint or safety')
        require(not any(lo < cut < hi for cut in cuts), 'Segment crosses a recorded validity barrier')
        for key in ('clock_generation', 'calibration_id', 'topology_id'):
            values = data[key][lo:hi, :len(CORE_STREAMS)]
            require(np.all(values == values[0]), f'Segment crosses {key} boundary')
        expected[lo:hi] = sid
        occupied = hi
    require(np.array_equal(expected, data['segment_id']), 'Segment indices differ from the manifest')


def _quality(data, mode, quality, reasons):
    pose, vel = data['pose_valid'], data['velocity_valid']
    materialized = data['node_mask']
    expanded = np.broadcast_to(pose[..., None], materialized.shape)
    require(np.all(~materialized | expanded) and np.array_equal(materialized.any(axis=-1), pose),
            'Node mask is inconsistent with pose validity')
    require(np.isfinite(data['pose'][materialized]).all(), 'Nonfinite valid pose node')
    require(np.isfinite(data['velocity_m_s'][materialized & vel[..., None]]).all(),
            'Nonfinite valid velocity node')
    raw = materialized[..., :].copy()
    raw[:, 2:] = False
    require(np.isfinite(data['sdk_common_pose'][raw]).all(), 'Nonfinite valid SDK node')
    expected = quality.index('SIMULATED' if mode == 'causal' else 'OFFLINE_PROXY')
    require(np.array_equal(data['clock_quality'], np.where(pose, expected, 0)) and
            np.array_equal(data['velocity_quality'], np.where(vel, expected, 0)),
            'Wrong historical clock quality')
    require(np.array_equal(np.isin(data['reason_code'], [reasons.index('VALID'), reasons.index('HELD')]), pose)
            and np.array_equal(data['velocity_reason_code'] == reasons.index('VALID'), vel),
            'Validity and reason code differ')


def _coordinates(data, meta, sources, mode, provenance):
    refs = data['source_index']
    pose = data['pose_valid']
    dictionaries = meta['dictionary']
    approved_assets = provenance['asset_sha256_by_part']
    approved_clocks = provenance['clock_generations']
    topologies = {(r['stream'], r['version']): r for r in meta['topologies']}
    for key, values in dictionaries.items():
        col = data[key]
        require(np.all((col >= -1) & (col < len(values))), f'Invalid {key} dictionary index')
    for r in sources:
        f = r['record']
        require(r['stream'] in (*STREAMS, SAFETY) and f.get('stream') == r['stream'],
                'Raw source stream identity differs')
        require(f.get('asset_sha256') == approved_assets.get(f.get('part_id')) and
                all(SHA.fullmatch(f.get(key) or '') for key in ('asset_sha256', 'cdr_sha256', 'schema_sha256',
                                                               'observation_schema_sha256', 'observation_cdr_sha256')),
                'Source or schema proof is outside the approved physical files')
        require(type(f.get('physical_message')) is int and f['physical_message'] >= 0 and
                f.get('observation_message') == f['physical_message'] + 1 and
                r['source_eligible'] and r['clock_generation'] in approved_clocks and
                r['calibration_id'] == provenance['calibration_sha256'],
                'Raw physical identity, clock generation, or calibration proof differs')
    for i, j in zip(*np.nonzero(pose)):
        s = STREAMS[j]
        endpoints = [sources[int(x)] for x in refs[i,j] if x>=0]
        require(len(endpoints)==(1 if mode=='causal' else 2) and
                all(r['stream']==s and r['raw_valid'] and r['fresh'] for r in endpoints),
                'Pose uses unauthenticated, stale-identity or unrelated endpoint')
        for key in ('clock_generation','calibration_id','topology_id'):
            col = int(data[key][i,j])
            value = dictionaries[key][col] if col >= 0 else None
            require(all(r[key] == value for r in endpoints), f'Endpoint {key} differs')
        if s.startswith('manus_'):
            tid = endpoints[0]['topology_id']
            require((s,tid) in topologies, 'Missing original MANUS topology')
            count = len(topologies[s,tid]['joints'])
        else:
            count = 1
        require(np.array_equal(data['node_mask'][i,j], np.arange(meta['N']) < count),
                'Node mask does not match paired topology')
        expected = endpoints[0]['capture_bounds_ns'] if mode=='causal' else (
            endpoints[0]['capture_bounds_ns'][1],endpoints[1]['capture_bounds_ns'][1])
        col = 'capture_bounds_ns' if mode=='causal' else 'bracket_ns'
        require(np.array_equal(data[col][i,j], expected), 'Pose bounds differ from endpoint')
        if mode=='causal':
            require(np.array_equal(data['arrival_bounds_ns'][i,j],endpoints[0]['arrival_bounds_ns']),
                    'Arrival bounds differ from issued physical endpoint')


def _time_and_joint(data, mode, params, sources):
    ticks = data['tick_ns']
    pose, vel = data['pose_valid'], data['velocity_valid']
    require(np.all(np.diff(ticks) == params['tick_ns']), 'Tick grid or clock restart inside shard')
    grid = np.broadcast_to(ticks[:, None], pose.shape)
    causal = mode == 'causal'
    bounds = data['capture_bounds_ns' if causal else 'bracket_ns']
    lo, hi = bounds[..., 0], bounds[..., 1]
    require(np.all(hi[pose]>=lo[pose]), 'Reversed capture or bracket')
    if causal:
        arrival = data['arrival_bounds_ns']
        age = data['age_ns']
        require(np.all(arrival[...,1][pose]>=arrival[...,0][pose]) and
                np.all(arrival[...,0][pose]>=hi[pose]) and
                np.array_equal(age[pose], (grid-lo)[pose]) and np.all(age[pose]<=params['age_ns']) and
                np.array_equal(data['velocity_age_ns'][vel], age[vel]),
                'Inconsistent causal bounds, age or velocity validity')
    else:
        span = hi-lo
        weight = data['interpolation_weight']
        require(np.all(grid[pose]>=lo[pose]) and np.all(grid[pose]<=hi[pose]) and
                np.array_equal(data['bracket_span_ns'][pose],span[pose]) and
                np.all(span[pose]<=params['gap_ns']) and np.all(data['velocity_age_ns']==-1),
                'Invalid offline time span or velocity age')
        expected = np.divide(grid-lo,span,out=np.zeros_like(weight),where=span>0)
        require(np.allclose(weight[pose],expected[pose],rtol=0,atol=1e-14),'Interpolation weight differs from stored endpoints')
    joint = pose[:,:len(CORE_STREAMS)].all(axis=1) & data['safety_permission']
    for a,b,key in ((0,1,'left_right_skew_ns'),(2,3,'left_right_skew_ns'),
                    (0,2,'wrist_hand_skew_ns'),(1,3,'wrist_hand_skew_ns')):
        joint &= np.maximum(hi[:,a],hi[:,b])-np.minimum(lo[:,a],lo[:,b])<=params[key]
    require(np.array_equal(joint,data['joint_valid']), 'Joint mask disagrees with clock/skew')
    for i in np.flatnonzero(data['safety_permission']):
        j = int(data['safety_source_index'][i])
        require(0<=j<len(sources), 'No safety provenance')
        record=sources[j]
        capture=record['capture_bounds_ns']
        require(record['stream']==SAFETY and record['safe'] is True and record['raw_valid'] and
                capture[1]<=ticks[i] and ticks[i]-capture[0]<=params['safety_age_ns'],
                'Safety came from unsafe, expired or future evidence')
        if causal:
            require(record['arrival_bounds_ns'][1]<=data['cutoff_ns'][i],'Safety arrived after cutoff')


def validate_semantics(data, meta, sources, mode, quality, reasons):
    _quality(data, mode, quality, reasons)
    identity = meta['identity']
    provenance = identity.get('publication_validation', {})
    _segments(data, meta, mode, provenance)
    if identity.get('publication_scope') in E_SCOPES:
        require(bool(provenance), 'Missing E0/E1 provenance')
        _coordinates(data, meta, sources, mode, provenance)
        _time_and_joint(data, mode, identity['parameters'], sources)
        tol = provenance['quaternion_tolerance']
        selected = data['node_mask']
        require(np.all(np.abs(np.linalg.norm(data['pose'][selected][:, 3:], axis=-1)-1) <= tol),
                'Valid exported quaternion is outside frozen tolerance')
