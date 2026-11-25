"""Quantify bounded CDR samples without inferring causality or training coverage."""
from collections import Counter, defaultdict
from hashlib import sha256
from math import dist
from statistics import median

from .decode_inputs import MANUS
from .raw_events import CLOCKS, xyz
from .source_registry import stable_json
from .telemetry import observed_roles, eligibility


def clock_stats(events, key):
    values = [row['clock'][key]['value'] for row in events]
    steps = [b-a for a,b in zip(values,values[1:]) if a is not None and b is not None]
    pos = [x for x in steps if x>0]
    center = median(pos) if pos else None
    return {
        'count':len(values),'missing':sum(x is None for x in values),'zero_values':values.count(0),
        'repeat_steps':steps.count(0),'backward_steps':sum(x<0 for x in steps),
        'step_median_raw':median(steps) if steps else None,'step_median_positive_raw':center,
        'min_step_raw':min(steps) if steps else None,'max_step_raw':max(steps) if steps else None,
        'large_forward_steps_diagnostic_only':sum(x>5*center for x in pos) if center else 0,
        'domain':next((r['clock'][key]['domain'] for r in events if r['clock'][key]['value'] is not None),None),
        'unit':next((r['clock'][key]['unit'] for r in events if r['clock'][key]['value'] is not None),None),
    }


def duplicate_class(a,b):
    stamp = a['clock']['sdk_source']['value']
    if stamp is None or stamp != b['clock']['sdk_source']['value']:
        return None
    if a['cdr_sha256']==b['cdr_sha256']:
        return 'IDENTICAL_ROS_PAYLOAD_RECORDED_AGAIN'
    seen = a['clock']['sdk_receive']['value']
    same = (seen is not None and seen==b['clock']['sdk_receive']['value'] and
            a['fields'].get('raw_nodes')==b['fields'].get('raw_nodes') and
            a['fields'].get('glove_id')==b['fields'].get('glove_id'))
    if same and a['clock']['ros_header']['value']!=b['clock']['ros_header']['value']:
        return 'SAME_CACHED_CAPTURE_CANDIDATE_WITH_NEW_HEADER'
    return 'SAME_SOURCE_VALUE_CAPTURE_IDENTITY_UNKNOWN'


def field_report(events, telemetry):
    observations = defaultdict(list)
    for row in events:
        observations[(row['recording_id'],row['index']['topic'],row['index']['schema_key'],row['segment'])].append(row)
    output, timestamp, topology = [], [], []
    for (rec,topic,key,segment),group in sorted(observations.items(),key=lambda item:str(item[0])):
        success = [r for r in group if r['decode_status']=='PASS']
        known = Counter(k for r in success for k in r['fields'])
        array_lengths = defaultdict(Counter)
        for r in success:
            for name,value in r['fields'].items():
                if isinstance(value,list):
                    array_lengths[name][str(len(value))] += 1
        ref = dict(recording_id=rec,topic=topic,schema_key=key,segment=segment)
        output.append(dict(**ref,total=len(group),decoded=len(success),field_present=dict(known),
                           array_lengths={k:dict(v) for k,v in array_lengths.items()},
                           fields_missing_counts={k:len(success)-v for k,v in known.items()},
                           clock_missing_counts={k:sum(r['clock'][k]['value'] is None for r in group) for k in CLOCKS},
                           nonfinite_messages=sum(bool(r.get('nonfinite')) for r in group),
                           record_examples=[r['index'] for r in success[:2]]))
        if topic not in MANUS:
            continue
        valid = [r for r in success if 'manus' in r]
        changes = [dict(from_topology=a['manus']['topology_sha256'],to_topology=b['manus']['topology_sha256'],
                        new_index=b['index']) for a,b in zip(valid,valid[1:])
                   if a['manus']['topology_sha256']!=b['manus']['topology_sha256']]
        examples = {}
        for r in valid:
            g = r['manus']
            if g['topology_sha256'] not in examples:
                examples[g['topology_sha256']]=dict(
                    record_index=r['index'],node_count=g['node_count'],frame_id=g['frame_id'],
                    node_topology=[{k:n[k] for k in ('node_id','parent_node_id','joint_type','chain_type','original_order')}
                                   for n in g['nodes']])
        topology.append(dict(**ref,observed_node_counts=dict(Counter(str(r['manus']['node_count']) for r in valid)),
                             samples=len(valid),topology_changes=changes,known_topologies=examples,
                             geometry_issue_counts=dict(Counter(i for r in valid for i in r['manus']['issues'])),
                             measured_validity_unknown=sum(r['manus']['validity']['sensor_status']=='UNKNOWN' for r in valid)))
        timestamps = {clock:clock_stats(group,clock) for clock in CLOCKS}
        timestamp.append(dict(**ref,clock_statistics=timestamps,
                              repeated_source_candidates=dict(Counter(c for a,b in zip(valid,valid[1:])
                                  if (c:=duplicate_class(a,b)) is not None))))
    absences = defaultdict(Counter)
    for r in telemetry:
        stats = absences[r['role']]
        stats['observed'] += 1
        if r['expected_dimension'] is None:
            continue
        for name,vec in r['field_vectors'].items():
            stats[f'{name}_absent'] += int(vec['field'] is None)
            stats[f'{name}_empty'] += int(vec['length']==0)
            stats[f'{name}_wrong_size'] += int(vec['length'] not in {r['expected_dimension'],None,0})
        stats['feedback_clock_unknown'] += int(r['clock']['robot_feedback']['status']=='UNKNOWN')
        stats['joint_order_or_axes_invalid'] += int(not r['axes_match_recorded_schema'])
    return output,timestamp,topology,{key:dict(count) for key,count in sorted(absences.items())}


def semantic_candidates(events):
    """Equal short prefixes of original capture values warrant a review, never a merge."""
    captures = defaultdict(list)
    for row in events:
        if row.get('manus') and row['segment']!='INITIAL_STATE':
            fields = row['fields']
            if fields.get('receive_timestamp_us') is not None and fields.get('source_timestamp_us') is not None:
                captures[(row['recording_id'],row['index']['topic'])].append(
                    [fields.get('glove_id'),fields['source_timestamp_us'],
                     fields['receive_timestamp_us'],fields.get('raw_nodes')])
    by_digest = defaultdict(list)
    for (rec,topic), samples in captures.items():
        if len(samples)>=8:
            key = sha256(stable_json(samples[:8]).encode()).hexdigest()
            by_digest[topic,key].append(rec)
    return [{'topic':topic,'prefix_raw_frames':8,'fingerprint':key,'recording_ids':sorted(ids),
             'status':'PENDING_REVIEW_NOT_MERGED'} for (topic,key),ids in sorted(by_digest.items()) if len(ids)>1]


def native_coverage(recordings, events, verified_ids):
    by_rec = defaultdict(lambda:defaultdict(list))
    for row in events:
        if (row['recording_id'] in verified_ids and row.get('pairing',{}).get('status')=='VERIFIED_WRITER_PAIR'
                and not row['index']['topic'].startswith('/dex/collection/observations/') and
                row['decode_status']=='PASS' and not row.get('nonfinite')):
            by_rec[row['recording_id']][row['index']['topic']].append(row)
    results = []
    for rec in recordings:
        if rec['id'] not in by_rec:
            continue
        fields = by_rec[rec['id']]
        health = {'captured_bimanual_tag':rec.get('task')=='bi_hand','task':rec.get('task')}
        widths = []
        for topic in sorted(MANUS|{'/vive/tracker/left_wrist','/vive/tracker/right_wrist'}):
            group = fields.get(topic,[])
            if group:
                start = group[0]['clock']['collector_realtime']['value']
                end = group[-1]['clock']['collector_realtime']['value']
                widths.append((start,end))
        health['four_stream_overlap_ns'] = max(0,min(end for _,end in widths)-max(t for t,_ in widths)) if len(widths)==4 else None
        for side in ('left','right'):
            group = fields.get(f'/vive/tracker/{side}_wrist',[])
            coords = [xyz((r['fields'].get('pose') or {}).get('position'),'xyz') for r in group]
            coords = [x for x in coords if x is not None]
            health[f'{side}_wrist_peak_displacement_raw_unit'] = max((dist(coords[0],x) for x in coords),default=None)
            glove = fields.get(f'/manus_glove_{side}',[])
            geom = [r['manus'] for r in glove if r.get('manus') and not r['manus']['issues']]
            if geom:
                initial = {x['node_id']:x['node_position'] for x in geom[0]['nodes']}
                hand_range = [dist(initial[x['node_id']],x['node_position']) for g in geom
                              if g['topology_sha256']==geom[0]['topology_sha256'] for x in g['nodes']]
                health[f'{side}_manus_max_local_node_motion_raw_unit'] = max(hand_range,default=None)
        results.append(dict(recording_id=rec['id'],**health))
    return results
