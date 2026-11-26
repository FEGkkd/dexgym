"""Stratified bounded replay comparison. Each population keeps its own denominator."""
from collections import Counter,defaultdict

import numpy as np

from .geometry import compose,inverse,residual,rotation,summary
from .synchronize import CORE_STREAMS,REASONS,STREAMS


def valstats(values):
    values=np.asarray(values,dtype=float)
    return summary(values.ravel())


def paired_masks(legacy,legacy_off,causal,offline,events):
    out={}
    for j,stream in enumerate(STREAMS):
        old=legacy['mask'][:,j]
        old_v=np.any(np.isfinite(legacy['velocity_m_s'][:,j]),axis=(1,2))
        now=[a['streams'][stream] for a in causal]
        valid=np.array([x['valid'] for x in now])
        new_v=np.array([x['velocity_valid'] for x in now])
        age_diff=[(x['age_ns']-legacy['age_ns'][k,j])*1e-6 for k,x in enumerate(now) if valid[k] and old[k]]
        shifts=Counter(f"{REASONS[legacy['reason'][k,j]]}->{x['reason']}" for k,x in enumerate(now)
                       if REASONS[legacy['reason'][k,j]]!=x['reason'])
        out[stream]=dict(role='ONLINE' if stream in CORE_STREAMS else 'DIAGNOSTIC_ONLY',
                         pose_old=int(old.sum()),pose_new=int(valid.sum()),mask_lost=int(sum(old&~valid)),
                         mask_gained=int(sum(valid&~old)),age_delta_ms=summary(age_diff),
                         velocity_old=int(old_v.sum()),velocity_new=int(new_v.sum()),
                         velocity_lost=int(sum(old_v&~new_v)),velocity_gained=int(sum(new_v&~old_v)),
                         velocity_reason_counts=dict(Counter(x['velocity_reason'] for x in now)),
                         pose_reason_transitions=dict(shifts),
                         offline_pose_old=int(legacy_off['mask'][:,j].sum()),
                         offline_pose_new=sum(x['streams'][stream]['valid'] for x in offline))
    former=legacy['joint_valid']
    revised=np.array([x['joint_valid'] for x in causal])
    new_off=np.array([x['joint_valid'] for x in offline])
    safety_edges=[]
    last=None
    for event in events:
        if event.stream=='tracker_safety' and event.safe!=last:
            safety_edges.append(event.known_ns)
            last=event.safe
    near=np.array([any(abs(x['tick_ns']-v)<=40_000_000 for v in safety_edges) for x in causal])
    return dict(ticks=len(former),causal_old=int(former.sum()),causal_new=int(revised.sum()),
                offline_old=int(legacy_off['joint_valid'].sum()),offline_new=int(new_off.sum()),
                joint_lost=int(sum(former&~revised)),joint_gained=int(sum(~former&revised)),
                safety_neighborhood=dict(radius_ns=40_000_000,transition_count=len(safety_edges),
                    ticks=int(near.sum()),old_valid=int(sum(former&near)),new_valid=int(sum(revised&near)),
                    reasons=dict(Counter(x['safety_reason'] for x,z in zip(causal,near) if z))),
                streams=out)


def _comparisons(c,o,source,tips,root):
    distances=np.linalg.norm(c['pose'][:,:3]-o['pose'][:,:3],axis=1)
    rotations=(rotation(c['pose'][:,3:]).inv()*rotation(o['pose'][:,3:])).magnitude()
    if not tips:
        return {'wrist_translation_mm':1000*distances,'wrist_rotation_rad':rotations}
    live=source[c['ref']['raw_index_token']]
    a,b=(source[x['raw_index_token']] for x in o['refs'])
    raw_ref=(1-o['weight'])*a[:,:3]+o['weight']*b[:,:3]
    raw=np.linalg.norm(live[:,:3]-raw_ref,axis=1)*1000
    return {'raw_MANUS_landmarks_mm':raw,'palm_finger_landmarks_mm':1000*np.delete(distances,root),
            'raw_tip_nodes_mm':raw[tips],'palm_tip_nodes_mm':1000*distances[tips]}


def _relative(a,b,tips):
    sa,sb=a['streams'],b['streams']
    result={}
    wrists=('tracker_left','tracker_right')
    if all(x[s]['valid'] for x in (sa,sb) for s in wrists):
        live,offline=[compose(inverse(x[wrists[0]]['pose'][0]),x[wrists[1]]['pose'][0]) for x in (sa,sb)]
        translation,angle=residual(live,offline)
        result.update(relative_wrist_translation_mm=[1000*translation],relative_wrist_rotation_rad=[angle])
    hands=('manus_left','manus_right')
    if all(x[s]['valid'] and x[s]['version']==sa[s]['version'] for x in (sa,sb) for s in hands):
        def hand_shape(row,stream):
            pose=row[stream]['pose'][tips[(stream,row[stream]['version'][1])],:3]
            i,j=np.triu_indices(len(pose),k=1)
            return np.linalg.norm(pose[i]-pose[j],axis=1)
        profiles=[hand_shape(row,hands[0])-hand_shape(row,hands[1]) for row in (sa,sb)]
        result['left_minus_right_local_tip_pair_lengths_mm']=(1000*np.abs(profiles[0]-profiles[1])).tolist()
    return result


def spatial_metrics(causal,offline,prepared,topologies):
    definitions={(x['stream'],x['version']):x['joints'] for x in topologies}
    fingers=('Thumb','Index','Middle','Ring','Pinky')
    tips,roots={},{}
    for key,joints in definitions.items():
        tip_rows=[(node['chain_type'],i) for i,node in enumerate(joints) if node['joint_type']=='TIP']
        by_finger=dict(tip_rows)
        if len(tip_rows)!=5 or set(by_finger)!=set(fingers):
            raise ValueError('Missing unique MANUS tip per named finger')
        tips[key]=[by_finger[name] for name in fingers]
        roots[key]=next(i for i,n in enumerate(joints) if n['parent_node_id'] in {-1,n['node_id']})
    source={event.ref['raw_index_token']:r['raw_pose'] for event,r in prepared}
    groups=defaultdict(list)
    for a,b in zip(causal,offline):
        for stream in STREAMS:
            c,o=a['streams'][stream],b['streams'][stream]
            if o['valid']:
                low,high=o['time_interval']
                groups[stream,'offline_bracket_span_ms'].append((high-low)*1e-6)
            if c['valid'] and o['valid'] and c['version']==o['version']:
                key=stream,c['version'][1]
                for mode,value in _comparisons(c,o,source,tips.get(key),roots.get(key)).items():
                    groups[stream,mode].extend(value)
        for mode,value in _relative(a,b,tips).items():
            groups['relative',mode].extend(value)
        for source_name,row in (('causal_capture_span',a),('offline_bracket_span',b)):
            for name,bound in row['skew_ns'].items():
                if bound is not None:
                    groups[source_name,name].append(bound*1e-6)
    stats=defaultdict(dict)
    for (group,name),values in groups.items():
        stats[group][name]=valstats(values)
    nodes=[dict(stream=s,topology=h,node_ids=[definitions[s,h][i]['node_id'] for i in selection])
           for (s,h),selection in tips.items()]
    return dict(metrics=dict(stats),identified_tip_nodes=nodes,finger_order=list(fingers),
                frames='SDK common, palm-local, and registered tracker kept separate. No retarget or robot world tips.')
