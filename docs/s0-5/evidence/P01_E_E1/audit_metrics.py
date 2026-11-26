"""Diagnostics over published E1 arrays and deduplicated in-body source references."""
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.data.hmb_build import REASONS_V1
from dexgym.tasks.direct.wholebody_s05.data.synchronize import CORE_STREAMS, STREAMS


def count(mask):
    return int(np.count_nonzero(mask))


class Distributions:
    def __init__(self):
        self.parts = defaultdict(list)

    def add(self, name, values):
        values = np.asarray(values).reshape(-1)
        if len(values):
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite diagnostic input: {name}')
            self.parts[name].append(values)

    def extend(self, other):
        for key, chunks in other.parts.items():
            self.parts[key].extend(chunks)

    def summary(self):
        result = {}
        for name, chunks in sorted(self.parts.items()):
            x = np.concatenate(chunks)
            p50, p95, p99 = np.percentile(x, [50, 95, 99])
            result[name] = dict(n=len(x), p50=float(p50), p95=float(p95), p99=float(p99), max=float(x.max()))
        return result


def mode_metrics(data, meta, mode, dist, qc, plan):
    stats = Counter(ticks=len(data['tick_ns']), joint_valid=count(data['joint_valid']),
                    joint_invalid=count(~data['joint_valid']), safety_permission=count(data['safety_permission']),
                    safety_unavailable=count(~data['safety_permission']), barriers=len(
                        meta['diagnostic_non_actor']['boundary_cut_indices']))
    usable = data['pose_valid']
    velocity = data['velocity_valid']
    segments = [hi-lo for lo,hi in meta['segments'][mode]]
    stats.update(segments=len(segments), segment_ticks=sum(segments),
                 valid_outside_segments=count(data['joint_valid'])-sum(segments))
    dist.add(f'{mode}.segment_length_ticks', segments)
    tip_columns = {t['stream']:[j for j,n in enumerate(t['joints']) if n['joint_type'].upper()=='TIP']
                   for t in meta['topologies']}
    for j, stream in enumerate(STREAMS):
        for field in ('pose_valid', 'velocity_valid'):
            stats[f'{stream}.{field}'] = count(data[field][:,j])
        for field in ('reason_code', 'velocity_reason_code'):
            keys, ns = np.unique(data[field][:,j], return_counts=True)
            stats.update({f'{stream}.{field}.{REASONS_V1[k]}':int(n) for k,n in zip(keys,ns,strict=True)})
        mask=data['node_mask'][:,j] & velocity[:,j,None]
        speeds=np.linalg.norm(data['velocity_m_s'][:,j][mask],axis=-1)
        dist.add(f'{mode}.{stream}.node_speed_m_s', speeds)
        stats[f'{stream}.speed_review_nodes'] = count(speeds>qc['speed_review']['mark_review_above_m_s'])
        p=data['pose'][:,j]
        dist.add(f'{mode}.{stream}.quaternion_norm_abs_error',
                 np.abs(np.linalg.norm(p[data['node_mask'][:,j]][:,3:],axis=-1)-1))
        if stream in tip_columns:
            top=next(t for t in meta['topologies'] if t['stream']==stream)
            xyz=data['sdk_common_pose'][usable[:,j],j,:,:3]
            if len(xyz):
                edges=np.asarray(top['edges'])
                lens=np.linalg.norm(xyz[:,edges[:,0]]-xyz[:,edges[:,1]],axis=-1)
                dev=np.abs(lens-np.median(lens,axis=0))
                dist.add(f'{mode}.{stream}.bone_deviation_from_partition_edge_median_m',dev)
                stats[f'{stream}.bone_review_edge_values']=count(dev>qc['bone_review_m'])
        if mode=='causal':
            dist.add(f'{mode}.{stream}.pose_age_ms',data['age_ns'][usable[:,j],j]/1e6)
            dist.add(f'{mode}.{stream}.velocity_age_ms',data['velocity_age_ns'][velocity[:,j],j]/1e6)
        else:
            dist.add(f'{mode}.{stream}.bracket_span_ms',data['bracket_span_ns'][usable[:,j],j]/1e6)
    time=data['capture_bounds_ns' if mode=='causal' else 'bracket_ns']
    for a,b,key in ((0,1,'hands'),(2,3,'wrists'),(0,2,'left_wrist_hand'),(1,3,'right_wrist_hand')):
        ok=usable[:,a]&usable[:,b]
        span=np.maximum(time[ok,a,1],time[ok,b,1])-np.minimum(time[ok,a,0],time[ok,b,0])
        dist.add(f'{mode}.skew_{key}_ms',span/1e6)
    if mode=='offline':
        stats.update(motion_metrics(data,meta,plan))
    return stats


def motion_metrics(data, meta, plan):
    # Match the frozen D diagnostic speed threshold. No motion label is a QC rejection or Actor feature.
    vel=data['velocity_m_s'][:,:4]
    node=data['node_mask'][:,:4] & data['velocity_valid'][:,:4,None]
    speed=np.linalg.norm(np.where(node[...,None],vel,0),axis=-1).max(axis=-1)
    moving=speed>plan['dynamic_threshold_m_s']
    valid=data['velocity_valid'][:,:4].all(axis=1)
    dynamic=moving.any(axis=1)
    both=lambda mask: count(mask & valid)
    stats=dict(static_ticks=both(~dynamic),dynamic_ticks=both(dynamic),unknown_motion_ticks=count(~valid),
               both_hands_shape_changing=both(moving[:,0]&moving[:,1]),
               both_registered_wrists_moving=both(moving[:,2]&moving[:,3]),
               both_sides_moving=both((moving[:,0]|moving[:,2])&(moving[:,1]|moving[:,3])))
    for j,stream in enumerate(CORE_STREAMS):
        stats[f'{stream}.moving_velocity_valid']=count(moving[:,j]&data['velocity_valid'][:,j])
    n=plan['stop_dwell_ticks']
    ids=data['segment_id']
    starts=stops=0
    for j in range(n,len(valid)-n+1):
        left,right=slice(j-n,j),slice(j,j+n)
        if ids[j]<0 or ids[j]!=ids[j-n] or ids[j]!=ids[j+n-1] or not valid[j-n:j+n].all():
            continue
        starts+=bool(not dynamic[left].any() and dynamic[right].all())
        stops+=bool(dynamic[left].all() and not dynamic[right].any())
    stats.update(starts=starts,stops=stops,same_partition_safety_release_edges=count(
        ~data['safety_permission'][:-1]&data['safety_permission'][1:]), same_partition_joint_recovery_edges=count(
        ~data['joint_valid'][:-1]&data['joint_valid'][1:]))
    return stats


def rotation_error(a,b):
    a=a/np.linalg.norm(a,axis=-1,keepdims=True)
    b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return 2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=-1)),0.,1.))


def pair_metrics(causal, offline, meta, dist):
    tips={t['stream']:[j for j,n in enumerate(t['joints']) if n['joint_type'].upper()=='TIP']
          for t in meta['topologies']}
    for j,stream in enumerate(STREAMS):
        common=causal['pose_valid'][:,j]&offline['pose_valid'][:,j]
        for field in ('pose','sdk_common_pose') if j<2 else ('pose',):
            left=causal[field][common,j]
            right=offline[field][common,j]
            # Both products use the original node order for the same approved topology.
            mask=causal['node_mask'][common,j]&offline['node_mask'][common,j]
            err=np.linalg.norm(left[...,:3]-right[...,:3],axis=-1)
            label='sdk_common' if field=='sdk_common_pose' else 'palm_local' if j<2 else 'tracker'
            dist.add(f'diagnostic.{stream}.{label}.translation_mm',err[mask]*1000)
            dist.add(f'diagnostic.{stream}.{label}.rotation_rad',rotation_error(left[mask,3:],right[mask,3:]))
            if j<2:
                cols=tips[stream]
                dist.add(f'diagnostic.{stream}.{label}.original_tip_node_mm',err[:,cols][mask[:,cols]]*1000)
        both=causal['velocity_valid'][:,j]&offline['velocity_valid'][:,j]
        mask=causal['node_mask'][both,j]&offline['node_mask'][both,j]
        diff=causal['velocity_m_s'][both,j]-offline['velocity_m_s'][both,j]
        dist.add(f'diagnostic.{stream}.velocity_difference_m_s',np.linalg.norm(diff[mask],axis=-1))
    both=(causal['pose_valid'][:,2:4]&offline['pose_valid'][:,2:4]).all(axis=1)
    if both.any():
        def rel(data):
            l,r=data['pose'][both,2,0],data['pose'][both,3,0]
            inv=Rotation.from_quat(l[:,3:]).inv()
            return inv.apply(r[:,:3]-l[:,:3]), (inv*Rotation.from_quat(r[:,3:])).as_quat()
        p,q=rel(causal)
        v,w=rel(offline)
        dist.add('diagnostic.relative_registered_wrists.translation_mm',1000*np.linalg.norm(p-v,axis=1))
        dist.add('diagnostic.relative_registered_wrists.rotation_rad',rotation_error(q,w))


def source_metrics(sources, block, dist):
    lo,hi=block['start_ns'],block['stop_ns']
    stats=defaultdict(Counter)
    seen={}
    intervals=defaultdict(list)
    for r in sources:
        stream, i, t = r['stream'], r['identity'], r['raw_clock']['collector_realtime']['value']
        prior,previous_time=seen.get(stream,(None,None))
        seen[stream]=(i,t)
        if not lo<=t<hi:
            continue
        s=stats[stream]
        s.update(body_publications=1, body_valid=int(r['raw_valid']), body_invalid=int(not r['raw_valid']))
        if prior and i['instance']==prior['instance']:
            gap=i['writer_sequence']-prior['writer_sequence']-1
            s['sequence_missing_between_observed']+=max(0,gap)
            if previous_time is not None and t>=previous_time:
                intervals[stream].append((t-previous_time)/1e6)
        if not stream.startswith('manus'):
            continue
        stamps=(i['sdk_timestamp_raw'],i['sdk_receive_raw'])
        if None in stamps:
            s['missing_sdk_identity']+=1
        if prior is None or i['instance']!=prior['instance'] or i['writer_sequence']!=prior['writer_sequence']+1:
            s['no_contiguous_predecessor']+=1
        elif None not in stamps and stamps==(prior['sdk_timestamp_raw'],prior['sdk_receive_raw']):
            s['same_sdk_pair_publications']+=1
        elif None not in stamps and r['raw_valid'] and r['fresh']:
            s['independent_sdk_pair_candidates']+=1
        else:
            s['unqualified_identity']+=1
    for stream,values in intervals.items():
        dist.add(f'raw.{stream}.publication_interval_ms',values)
    return {f'{name}.{k}':v for name,items in stats.items() for k,v in items.items()}
